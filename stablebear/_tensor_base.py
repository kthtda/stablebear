from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from typing import Union

from . import _sb_cpp as cpp

Shape = cpp.Shape

ShapeLike = Shape | tuple[int, ...]


def _unpickle_tensor(data: bytes):
    import io as _io
    from .io import _load
    return _load(_io.BytesIO(data))

CppTensor = Union[
    cpp.Float32Tensor,
    cpp.Float64Tensor,
    cpp.Int32Tensor,
    cpp.Int64Tensor,
    cpp.Uint32Tensor,
    cpp.Uint64Tensor,
    cpp.Pcf32Tensor,
    cpp.Pcf64Tensor,
    cpp.Pcf32iTensor,
    cpp.Pcf64iTensor,
    cpp.PointCloud32Tensor,
    cpp.PointCloud64Tensor,
    cpp.BoolTensor,
]


def _infer_shape_and_flatten(data):
    """Walk a nested list/tuple structure and return (shape, flat_elements).

    Recursion stops at any element that is not a list or tuple.
    Validates that the structure is rectangular.
    """
    shape: list[int] = []

    def _probe(obj, depth):
        if not isinstance(obj, (list, tuple)):
            return
        if depth == len(shape):
            shape.append(len(obj))
        elif shape[depth] != len(obj):
            raise ValueError(
                f"Ragged nested list: expected length {shape[depth]} at depth {depth}, got {len(obj)}"
            )
        if obj:
            _probe(obj[0], depth + 1)

    _probe(data, 0)

    flat: list = []

    def _collect(obj, depth):
        if depth == len(shape):
            flat.append(obj)
        else:
            for item in obj:
                _collect(item, depth + 1)

    _collect(data, 0)
    return tuple(shape), flat


def _tensor_from_nested(data, elem_to_tensor, default_ctor=None):
    """Build a C++ tensor from a nested list/tuple of elements with ``._data``.

    *elem_to_tensor* maps C++ element type to C++ tensor constructor,
    e.g. ``{cpp.Pcf_f32_f32: cpp.Pcf32Tensor}``.
    *default_ctor* is used when the list is empty (no element to infer from).
    """
    shape, flat = _infer_shape_and_flatten(data)
    if not flat:
        if default_ctor is None:
            default_ctor = next(iter(elem_to_tensor.values()))
        return default_ctor(cpp.Shape(list(shape or (0,))))
    tensor_ctor = elem_to_tensor.get(type(flat[0]._data))
    if tensor_ctor is None:
        raise TypeError(f"Unsupported element type {type(flat[0])}")
    t = tensor_ctor(cpp.Shape([len(flat)]))
    for i, elem in enumerate(flat):
        t._set_element([i], elem._data)
    if shape != (len(flat),):
        t = t.reshape(list(shape))
    return t


def _slice_to_cpp(s):
    """Convert a Python ``slice`` to a C++ ``SliceRange`` (negatives resolved in C++)."""
    return cpp.slice_range(s.start, s.stop, s.step)


def _is_bool_scalar(obj):
    """True for Python ``bool`` and 0-d numpy booleans (treated as scalar-bool indices)."""
    import numpy as np
    return isinstance(obj, (bool, np.bool_))


def _as_int_index(obj):
    """Return ``obj`` as a Python ``int`` if it is integer-like (via ``__index__``).

    Returns ``None`` for booleans (handled separately) and non-integer objects.
    Covers Python ``int`` and numpy integer scalars (``np.int64`` etc.).
    """
    if _is_bool_scalar(obj):
        return None
    try:
        return operator.index(obj)
    except TypeError:
        return None


def _is_full_slice(s):
    """True for ``slice(None, None, None)`` (a bare ``:``)."""
    return isinstance(s, slice) and s.start is None and s.stop is None and s.step is None


def _resolve_negative_indices(index_tensor, axis_size):
    """Resolve negative indices and bounds-check."""
    import numpy as np
    from .base_tensor import IntTensor
    arr = np.asarray(index_tensor).astype(np.int64).copy()
    neg = arr < 0
    arr[neg] += axis_size
    if np.any((arr < 0) | (arr >= axis_size)):
        raise IndexError(f"Index out of bounds for axis with size {axis_size}")
    return IntTensor(arr)


def _resolve_axis(axis: int, ndim: int) -> int:
    axis = operator.index(axis)
    resolved = axis + ndim if axis < 0 else axis
    if not 0 <= resolved < ndim:
        raise IndexError(
            f"axis {axis} is out of range for tensor with {ndim} dimension(s)"
        )
    return resolved


class Tensor(ABC):
    _data: CppTensor

    __array_ufunc__ = None

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            f"np.asarray() is not supported for {type(self).__name__}. "
            f"Only numeric tensors (FloatTensor, IntTensor, BoolTensor) "
            f"can be converted to NumPy arrays."
        )

    @staticmethod
    def _coerce_ndarray_index(arr):
        """Classify a numpy index array into a normalized index entry.

        Returns ``('int', v)`` for a 0-d integer array, ``('adv', tensor)`` for
        an integer (->IntTensor) or boolean (->BoolTensor) array. Raises an
        ``IndexError`` (matching NumPy) for non-integer/boolean arrays.
        """
        import numpy as np
        from .base_tensor import BoolTensor, IntTensor
        if arr.dtype == np.bool_:
            return ('adv', BoolTensor(arr))
        if np.issubdtype(arr.dtype, np.integer):
            if arr.ndim == 0:
                return ('int', int(arr))
            return ('adv', IntTensor(arr))
        raise IndexError(
            "arrays used as indices must be of integer (or boolean) type")

    def _normalize_index(self, raw):
        """Normalize an arbitrary NumPy-style index into axis entries + inserts.

        Returns ``(entries, inserts)`` where:

        * ``entries`` is a list of axis-consuming index objects — Python ``int``,
          ``slice``, ``IntTensor`` or ``BoolTensor`` — with a single Ellipsis
          expanded and trailing axes padded with full slices.
        * ``inserts`` is a list of ``(output_position, length)`` describing
          ``None``/``np.newaxis`` (length 1) and scalar-boolean (length 1 for
          ``True``, 0 for ``False``) axes to insert after the core indexing.
        """
        import numpy as np
        from .base_tensor import BoolTensor, IntTensor

        ndim = self.ndim

        items = list(raw) if isinstance(raw, tuple) else [raw]

        # Pass 1: classify every item into a tagged, coerced form.
        coerced = []
        n_ellipsis = 0
        for s in items:
            if s is Ellipsis:
                coerced.append(('ellipsis',))
                n_ellipsis += 1
            elif s is None:
                coerced.append(('newaxis', 1))
            elif _is_bool_scalar(s):
                coerced.append(('newaxis', 1 if bool(s) else 0))
            elif isinstance(s, slice):
                coerced.append(('slice', s))
            elif isinstance(s, (BoolTensor, IntTensor)):
                coerced.append(('adv', s))
            elif isinstance(s, np.ndarray):
                coerced.append(self._coerce_ndarray_index(s))
            elif isinstance(s, list):
                coerced.append(self._coerce_ndarray_index(np.asarray(s)))
            else:
                iv = _as_int_index(s)
                if iv is not None:
                    coerced.append(('int', iv))
                else:
                    raise IndexError(
                        "only integers, slices (`:`), ellipsis (`...`), "
                        "numpy.newaxis (`None`) and integer or boolean arrays "
                        f"are valid indices, not {type(s).__name__}")

        if n_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        def _consumes(c):
            tag = c[0]
            if tag in ('int', 'slice'):
                return 1
            if tag == 'adv':
                return c[1].ndim if isinstance(c[1], BoolTensor) else 1
            return 0  # newaxis / ellipsis consume no input axis

        n_consume = sum(_consumes(c) for c in coerced if c[0] != 'ellipsis')

        if n_consume > ndim:
            raise IndexError(
                f"too many indices for tensor: tensor is {ndim}-dimensional, "
                f"but {n_consume} were indexed")

        # Pass 2: expand a single Ellipsis into the right number of full slices.
        expanded = []
        for c in coerced:
            if c[0] == 'ellipsis':
                expanded.extend([('slice', slice(None))] * max(0, ndim - n_consume))
            else:
                expanded.append(c)

        # Pad trailing axes with full slices when there is no Ellipsis.
        if n_ellipsis == 0 and n_consume < ndim:
            expanded.extend([('slice', slice(None))] * (ndim - n_consume))

        # Pass 3: split into axis-consuming entries and newaxis/scalar-bool inserts.
        entries = []
        inserts = []
        out_pos = 0
        for c in expanded:
            tag = c[0]
            if tag == 'newaxis':
                inserts.append((out_pos, c[1]))
                out_pos += 1
            elif tag == 'int':
                entries.append(c[1])
            elif tag == 'slice':
                entries.append(c[1])
                out_pos += 1
            else:  # 'adv'
                entries.append(c[1])
                out_pos += 1
        return entries, inserts

    def _resolve_axis_int(self, idx, axis):
        """Resolve a (possibly negative) integer against ``self.shape[axis]``."""
        n = self.shape[axis]
        resolved = idx + n if idx < 0 else idx
        if resolved < 0 or resolved >= n:
            raise IndexError(
                f"index {idx} is out of bounds for axis {axis} with size {n}")
        return resolved

    def __getitem__(self, slices):
        """Index the tensor NumPy-style, returning an element or a sub-tensor.

        Supports the common NumPy index objects, in any combination via a tuple:

        * integers, including negative integers (which count from the end);
        * slices, including negative bounds and negative steps;
        * ``Ellipsis`` (``...``) and ``None`` / ``np.newaxis``;
        * integer arrays (``numpy.ndarray``, a Python ``list``, or an
          ``IntTensor``) for gather-style "advanced" indexing;
        * boolean masks (``BoolTensor`` or a boolean ``ndarray``), either
          full-shape, per-axis, or matching the leading axes.

        A full-integer index returns a single element — a Python scalar for
        numeric tensors, or the element wrapper (e.g. ``Pcf``) otherwise.
        Anything else returns a tensor. Basic (int/slice) indexing returns a
        view that shares storage with the original; advanced (array/mask)
        indexing returns a copy.

        When two or more advanced indices appear together they use outer
        (``numpy.ix_``-style) semantics rather than NumPy's vectorized
        broadcasting; see the "Indexing and Masking" guide for details.

        Parameters
        ----------
        slices : int, slice, Ellipsis, None, tuple, array-like, or BoolTensor
            The index. A tuple combines per-axis indices.

        Returns
        -------
        element or Tensor
            A single element for a full-integer index, otherwise a tensor.

        Raises
        ------
        IndexError
            If an integer index is out of range, too many indices are given,
            or a non-index object (such as a float) is used as an index.
        ValueError
            If a slice step is zero.
        """
        entries, inserts = self._normalize_index(slices)
        result = self._getitem_entries(entries, allow_scalar=not inserts)
        for pos, length in inserts:
            result = result.expand_dims(pos)
            if length == 0:
                result = self._zero_len_axis(result, pos)
        return result

    @staticmethod
    def _zero_len_axis(t, pos):
        """Return a view of ``t`` with axis ``pos`` truncated to length 0."""
        cpp_slices = [cpp.slice_range(None, None, None) for _ in range(t.ndim)]
        cpp_slices[pos] = cpp.slice_range(0, 0, None)
        return t._to_py_tensor(t._data[cpp_slices])

    def _getitem_entries(self, entries, allow_scalar=True):
        """Index using normalized entries (int/slice/IntTensor/BoolTensor)."""
        from .base_tensor import BoolTensor, IntTensor

        advanced = [(i, s) for i, s in enumerate(entries)
                    if isinstance(s, (BoolTensor, IntTensor))]

        # A single boolean mask spanning the leading axes (or the full shape),
        # with no other non-trivial index, collapses those axes (NumPy parity).
        if (len(advanced) == 1 and isinstance(advanced[0][1], BoolTensor)
                and advanced[0][0] == 0
                and all(_is_full_slice(e) for i, e in enumerate(entries) if i != 0)):
            return self._bool_mask_getitem(advanced[0][1])

        if not advanced:
            return self._basic_getitem(entries, allow_scalar)

        # Mixed/multiple advanced indices: apply slices first, then each advanced
        # index sequentially (outer / np.ix_-style semantics).
        basic = [slice(None) if isinstance(s, (BoolTensor, IntTensor)) else s
                 for s in entries]
        result = self._basic_getitem(basic, allow_scalar=False)
        for orig_pos, idx in advanced:
            dims_dropped = sum(1 for j in range(orig_pos) if isinstance(entries[j], int))
            axis = orig_pos - dims_dropped
            if isinstance(idx, BoolTensor):
                result = self._to_py_tensor(result._data.axis_select(axis, idx._data))  # type: ignore[arg-type]
            else:
                result = self._index_select_nd(result, axis, idx)
        return result

    def _basic_getitem(self, entries, allow_scalar=True):
        """Index using only ints and slices (negatives resolved, bounds checked)."""
        if (allow_scalar and len(entries) == self.ndim
                and all(isinstance(s, int) for s in entries)):
            idx = [self._resolve_axis_int(v, k) for k, v in enumerate(entries)]
            return self._represent_element(self._data._get_element(idx))

        cpp_slices = []
        for k, s in enumerate(entries):
            if isinstance(s, int):
                cpp_slices.append(cpp.slice_index(self._resolve_axis_int(s, k)))
            else:
                cpp_slices.append(_slice_to_cpp(s))
        return self._to_py_tensor(self._data[cpp_slices])

    def _bool_mask_getitem(self, mask):
        """Select with a boolean mask matching the full shape or leading axes."""
        import numpy as np
        from .base_tensor import IntTensor

        tshape = tuple(self.shape[i] for i in range(self.ndim))
        mshape = tuple(mask.shape[i] for i in range(mask.ndim))

        if mshape == tshape:
            return self._to_py_tensor(self._data.masked_select(mask._data))  # type: ignore[arg-type]

        if mask.ndim <= self.ndim and mshape == tshape[:mask.ndim]:
            trailing = list(tshape[mask.ndim:])
            true_idx = np.flatnonzero(np.asarray(mask).reshape(-1)).astype(np.int64)
            collapsed = self._data.reshape([-1] + trailing)
            return self._to_py_tensor(collapsed.index_select(0, IntTensor(true_idx)._data))

        raise ValueError(
            f"boolean index of shape {mshape} does not match the indexed "
            f"tensor of shape {tshape} along the leading axes")

    def _index_select_nd(self, result, axis, idx):
        """Gather along ``axis`` using an integer index array of any rank."""
        import numpy as np
        from .base_tensor import IntTensor

        axis_size = result.shape[axis]
        arr = np.asarray(idx).astype(np.int64)
        resolved = arr.copy()
        resolved[resolved < 0] += axis_size
        if np.any((resolved < 0) | (resolved >= axis_size)):
            raise IndexError(
                f"index out of bounds for axis {axis} with size {axis_size}")

        if arr.ndim <= 1:
            return self._to_py_tensor(
                result._data.index_select(axis, IntTensor(resolved)._data))

        selected = result._data.index_select(axis, IntTensor(resolved.reshape(-1))._data)
        rshape = [result.shape[i] for i in range(result.ndim)]
        new_shape = rshape[:axis] + list(arr.shape) + rshape[axis + 1:]
        return self._to_py_tensor(selected.reshape(new_shape))

    @abstractmethod
    def _get_valid_setitem_dtypes(self):
        raise NotImplementedError()

    def _validate_setitem_dtype(self, val):
        valid_dtypes = self._get_valid_setitem_dtypes()
        if not any(isinstance(val, dt) for dt in valid_dtypes):
            raise TypeError(
                f"Tried to set an item of a tensor of type {type(self)} to a value of type {type(val)}. Only {valid_dtypes} are accepted."
            )

    def _coerce_rhs(self, val):
        """Cast a tensor RHS to ``self``'s dtype on assignment (NumPy-style).

        Only numeric tensors are cross-cast (e.g. int -> float); other tensor
        families are returned unchanged.
        """
        from .base_tensor import NumericTensor
        if (isinstance(self, NumericTensor) and isinstance(val, NumericTensor)
                and getattr(val, "dtype", None) is not getattr(self, "dtype", None)):
            return val.astype(self.dtype)
        return val

    def __setitem__(self, slices, val):
        """Assign into the tensor using NumPy-style indexing.

        Accepts the same index objects as ``__getitem__``. The right-hand side
        may be:

        * a scalar or element, broadcast across the whole selected region; or
        * a tensor, broadcast to the selected shape. A numeric tensor whose
          dtype differs from this tensor's is cast (e.g. int to float).

        Basic-slice and single-integer targets are views, so the assignment
        writes through to the original tensor.

        Parameters
        ----------
        slices : int, slice, Ellipsis, None, tuple, array-like, or BoolTensor
            The index identifying the region to assign to (see ``__getitem__``).
        val : scalar, element, or Tensor
            The value(s) to assign.

        Raises
        ------
        IndexError
            If an integer index is out of range or too many indices are given.
        ValueError
            If a slice step is zero, or the right-hand side is not
            broadcast-compatible with the selected region.
        TypeError
            If ``val`` has a type that cannot be assigned to this tensor.
        """
        entries, inserts = self._normalize_index(slices)
        # A scalar-boolean ``False`` (or any zero-length newaxis) selects nothing.
        if any(length == 0 for _, length in inserts):
            return
        self._validate_setitem_dtype(val)
        self._setitem_entries(entries, val)

    def _setitem_entries(self, entries, val):
        """Assign using normalized entries (int/slice/IntTensor/BoolTensor)."""
        from .base_tensor import BoolTensor, IntTensor

        # Single full-shape boolean mask: flat masked assign/fill.
        if len(entries) == 1 and isinstance(entries[0], BoolTensor):
            if isinstance(val, Tensor):
                self._data.masked_assign(entries[0]._data, self._coerce_rhs(val)._data)  # type: ignore[arg-type]
            else:
                self._data.masked_fill(entries[0]._data, self._decay_value(val))  # type: ignore[arg-type]
            return

        advanced = [(i, s) for i, s in enumerate(entries)
                    if isinstance(s, (BoolTensor, IntTensor))]

        if not advanced:
            self._basic_setitem(entries, val)
            return

        basic = [slice(None) if isinstance(s, (BoolTensor, IntTensor)) else s
                 for s in entries]
        view = self._basic_getitem(basic, allow_scalar=False)

        selectors = []
        for orig_pos, idx in advanced:
            dims_dropped = sum(1 for j in range(orig_pos) if isinstance(entries[j], int))
            axis = orig_pos - dims_dropped
            if isinstance(idx, BoolTensor):
                selectors.append((axis, idx._data, True))
            else:
                resolved = _resolve_negative_indices(idx, view.shape[axis])
                selectors.append((axis, resolved._data, False))

        if len(selectors) == 1:
            axis, sel_data, is_bool = selectors[0]
            if is_bool:
                if isinstance(val, Tensor):
                    view._data.axis_assign(axis, sel_data, self._coerce_rhs(val)._data)  # type: ignore[arg-type]
                else:
                    view._data.axis_fill(axis, sel_data, self._decay_value(val))  # type: ignore[arg-type]
            else:
                if isinstance(val, Tensor):
                    view._data.index_assign(axis, sel_data, self._coerce_rhs(val)._data)  # type: ignore[arg-type]
                else:
                    view._data.index_fill(axis, sel_data, self._decay_value(val))  # type: ignore[arg-type]
        else:
            sel_pairs = [(axis, data) for axis, data, _ in selectors]
            if isinstance(val, Tensor):
                view._data.outer_assign(sel_pairs, self._coerce_rhs(val)._data)  # type: ignore[arg-type]
            else:
                view._data.outer_fill(sel_pairs, self._decay_value(val))  # type: ignore[arg-type]

    def _basic_setitem(self, entries, val):
        """Assign using only ints and slices (negatives resolved, bounds checked)."""
        import numpy as np
        from .base_tensor import NumericTensor

        if (len(entries) == self.ndim
                and all(isinstance(s, int) for s in entries)):
            idx = [self._resolve_axis_int(v, k) for k, v in enumerate(entries)]
            self._data._set_element(idx, self._decay_value(val))
            return

        cpp_slices = []
        for k, s in enumerate(entries):
            if isinstance(s, int):
                cpp_slices.append(cpp.slice_index(self._resolve_axis_int(s, k)))
            else:
                cpp_slices.append(_slice_to_cpp(s))

        if isinstance(val, Tensor):
            self._data[cpp_slices] = self._coerce_rhs(val)._data
        elif isinstance(val, np.ndarray) and isinstance(self, NumericTensor):
            # Element-wise array RHS: wrap as a same-dtype tensor and broadcast.
            self._data[cpp_slices] = self._coerce_rhs(type(self)(val))._data
        else:
            # Scalar (or single-element) RHS: broadcast-fill the selected view.
            view = self._to_py_tensor(self._data[cpp_slices])
            shp = [view.shape[i] for i in range(view.ndim)] if view.ndim > 0 else [1]
            filler = type(self._data)(cpp.Shape(shp), self._decay_value(val))
            self._data[cpp_slices] = filler

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(self.shape[0]):
            yield self[i]

    def __eq__(self, rhs):
        if not isinstance(rhs, Tensor):
            return NotImplemented
        from .base_tensor import BoolTensor
        return BoolTensor(self._data == rhs._data)  # type: ignore[arg-type]

    def __ne__(self, rhs):
        if not isinstance(rhs, Tensor):
            return NotImplemented
        from .base_tensor import BoolTensor
        return BoolTensor(self._data != rhs._data)  # type: ignore[arg-type]

    def _compare(self, rhs, op, symbol):
        """Elementwise ordering comparison returning a ``BoolTensor``.

        Accepts another ``Tensor`` (compared in C++) or a scalar / ``numpy``
        array on the right-hand side. Scalars and arrays are broadcast against
        ``self`` (NumPy semantics), so idioms such as ``t[t > 3]`` work.
        """
        from .base_tensor import BoolTensor
        if isinstance(rhs, Tensor):
            return BoolTensor(op(self._data, rhs._data))
        import numpy as np
        try:
            arr = np.asarray(self)
        except TypeError:
            raise TypeError(
                f"'{symbol}' not supported between instances of "
                f"'{type(self).__name__}' and '{type(rhs).__name__}'"
            ) from None
        return BoolTensor(op(arr, rhs))

    def __lt__(self, rhs):
        return self._compare(rhs, operator.lt, "<")

    def __le__(self, rhs):
        return self._compare(rhs, operator.le, "<=")

    def __gt__(self, rhs):
        return self._compare(rhs, operator.gt, ">")

    def __ge__(self, rhs):
        return self._compare(rhs, operator.ge, ">=")

    def array_equal(self, rhs) -> bool:
        """Test whether two tensors have the same shape and all equal elements.

        Parameters
        ----------
        rhs : Tensor
            The tensor to compare with.

        Returns
        -------
        bool
            True if the tensors are elementwise equal, False otherwise.
        """
        return self._data.array_equal(rhs._data)

    def __reduce__(self):
        import io as _io
        from .io import _save, _load
        buf = _io.BytesIO()
        _save(self, buf)
        return _unpickle_tensor, (buf.getvalue(),)

    def __deepcopy__(self, memodict=None):
        return self._to_py_tensor(self._data.copy())

    def copy(self):
        return self.__deepcopy__()

    @abstractmethod
    def _to_py_tensor(self, data):
        """
        Convert C++/Python tensor type to a Python tensor. Normally, it is enough to just return Datatype(data) where
        Datatype is the current class's type.
        """
        raise NotImplementedError()

    @abstractmethod
    def _decay_value(self, val):
        """
        Convert a Python value into one that can be used by the corresponding C++ class. For example, if `X` is a Python
        `Tensor`, `_decay_value` should convert a Python value `val` so that the following (pseudocode) works:

        `X[1,2,3] = val`

        `X._data._set_element("1,2,3", self._decay_value(val))
        """
        raise NotImplementedError()

    @abstractmethod
    def _represent_element(self, element):
        """
        Casts a single C++ element into the corresponding Python type. For example, a C++ function may return a
        stablebear._sb_cpp.Pcf32, and then we would like to cast it into a stablebear.Pcf by simply returning Pcf(element).
        """
        raise NotImplementedError()

    def broadcast_to(self, shape):
        """Return a broadcast view of this tensor with the given target shape.

        Dimensions of size 1 are expanded (stride set to 0); prepended
        dimensions also get stride 0. No data is copied — the result shares
        the underlying storage.

        Parameters
        ----------
        shape : tuple of int
            The target shape. Must be broadcast-compatible with the current
            shape.

        Returns
        -------
        Tensor
            A non-contiguous view of this tensor with the target shape.

        Raises
        ------
        ValueError
            If the shapes are not broadcast-compatible.
        """
        return self._to_py_tensor(self._data.broadcast_to(list(shape)))

    def flatten(self):
        return self._to_py_tensor(self._data.flatten())

    def reshape(self, shape):
        return self._to_py_tensor(self._data.reshape(list(shape)))

    def transpose(self, axes=None):
        if axes is None:
            return self._to_py_tensor(self._data.transpose([]))
        resolved_axes = [_resolve_axis(axis, self.ndim) for axis in axes]
        return self._to_py_tensor(self._data.transpose(resolved_axes))

    @property
    def T(self):
        return self.transpose()

    def swapaxes(self, axis1, axis2):
        axis1 = _resolve_axis(axis1, self.ndim)
        axis2 = _resolve_axis(axis2, self.ndim)
        return self._to_py_tensor(self._data.swapaxes(axis1, axis2))

    def squeeze(self, axis=None):
        if axis is None:
            return self._to_py_tensor(self._data.squeeze())
        return self._to_py_tensor(self._data.squeeze(_resolve_axis(axis, self.ndim)))

    def expand_dims(self, axis):
        axis = _resolve_axis(axis, self.ndim + 1)
        return self._to_py_tensor(self._data.expand_dims(axis))

    # Implementation: looks up a C++ cast function by naming convention:
    # cpp.cast_{src_tag}_{dst_tag}. To add a new cast:
    #   1. Ensure the C++ target type is constructible from the source
    #      (add a converting constructor if needed).
    #   2. Add a tensor_cast binding in py_tensor.cpp named cast_{src}_{dst}.
    #   3. Add entries to _DTYPE_TAG and _DTYPE_TO_WRAPPER in typing.py
    #      if the dtype is new.
    def astype(self, dtype):
        """Return a new tensor with elements cast to the given dtype.

        Supported casts are same-family precision changes (e.g. float32 → float64,
        pcf32 → pcf64) and numeric cross-family (e.g. float → int).
        """
        if dtype == self.dtype:
            return self.copy()
        from . import _sb_cpp as cpp
        from .typing import _DTYPE_TAG, _DTYPE_TO_WRAPPER, _init_dtype_wrappers
        _init_dtype_wrappers()
        src_tag = _DTYPE_TAG.get(self.dtype)
        dst_tag = _DTYPE_TAG.get(dtype)
        if src_tag is None or dst_tag is None:
            raise TypeError(
                f"Cannot cast from {self.dtype.__name__} to {dtype.__name__}")
        cast_fn = getattr(cpp, f"cast_{src_tag}_{dst_tag}", None)
        if cast_fn is None:
            raise TypeError(
                f"Cannot cast from {self.dtype.__name__} to {dtype.__name__}")
        return _DTYPE_TO_WRAPPER[dtype](cast_fn(self._data))

    @property
    def shape(self) -> Shape:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return len(self._data.shape)

    @property
    def size(self) -> int:
        s = self._data.shape
        result = 1
        for i in range(len(s)):
            result *= s[i]
        return result

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self._data.shape[0]

    def __bool__(self) -> bool:
        """Return the truth value of a single-element tensor (NumPy semantics).

        A tensor with exactly one element (including a 0-d/scalar tensor)
        returns that element's truthiness; a tensor with any other number of
        elements has an ambiguous truth value and raises ``ValueError``.
        Without this, ``bool()`` would fall back to ``__len__`` and report the
        first-dim length rather than the value.
        """
        if self.size == 1:
            idx = [0] * self.ndim
            return bool(self._data._get_element(idx))
        raise ValueError(
            f"The truth value of a tensor with {self.size} elements is ambiguous. "
            "Use array_equal() for whole-tensor comparison."
        )

    @property
    def strides(self):
        return self._data.strides

    @property
    def offset(self):
        return self._data.offset

    @staticmethod
    def _validate_constructor_arg(tensor, arg, valid_types):
        if not any(type(arg) is tp for tp in valid_types):
            raise TypeError(
                f"Tried to construct tensor of type {type(tensor)} from argument of type {type(arg)}. Only the following type(s) are allowed: {valid_types}."
            )


class FunctionTensorMixin:
    """Mixin for tensors whose elements can be evaluated at domain points.

    Delegates to C++ ``__call__`` overloads which handle scalar, numpy array,
    and Tensor inputs. Tensor inputs are unwrapped, evaluated in C++,
    and the result is re-wrapped via the input's ``_to_py_tensor``.
    For scalars, lists, and numpy arrays the result is returned as an ndarray.
    """

    def __call__(self, t):
        """Evaluate every element of the tensor at the given domain point(s).

        Parameters
        ----------
        t : scalar, list, numpy.ndarray, or NumericTensor
            A single domain value or a collection of values.

        Returns
        -------
        numpy.ndarray or NumericTensor
            For scalar *t*, the result has shape ``self.shape``.
            For array-like *t* of shape ``t_shape``, the result has shape
            ``self.shape + t_shape``.
            A ``NumericTensor`` input produces a ``NumericTensor`` output;
            all other inputs produce a ``numpy.ndarray``.
        """
        import numpy as np

        if isinstance(t, Tensor):
            return t._to_py_tensor(self._data(t._data))
        if isinstance(t, list):
            t = np.asarray(t)
        if isinstance(t, (int, float, np.generic, np.ndarray)):
            return np.asarray(self._data(t))
        raise TypeError(f"Cannot evaluate tensor at argument of type {type(t)}")


class ArithmeticTensorMixin:
    """Mixin providing elementwise arithmetic operators for tensors.

    Operators accept either a scalar or another tensor of the same type.
    When both operands are tensors, NumPy-style broadcasting is applied:
    shapes are compared right-to-left, dimensions match when equal or one
    is 1, and missing leading dimensions are treated as size 1.

    In-place operators (``+=``, ``-=``, ``*=``, ``/=``) require that the
    broadcast output shape equals the shape of the left-hand operand
    (the left-hand side is never expanded).
    """

    def _decay_operand(self, val):
        if hasattr(val, "_data"):
            return val._data
        import numpy as np
        if isinstance(val, (np.ndarray, list, tuple)):
            # Wrap an array-like RHS as a same-type tensor (cast to this
            # tensor's dtype where applicable) so that ``tensor + array``
            # mirrors ``array + tensor``. See issue #62.
            wrapped = self._coerce_rhs(type(self)(val))
            return wrapped._data
        return val

    def __add__(self, rhs):
        return self._to_py_tensor(self._data + self._decay_operand(rhs))

    def __radd__(self, lhs):
        return self._to_py_tensor(self._decay_operand(lhs) + self._data)

    def __iadd__(self, rhs):
        self._data += self._decay_operand(rhs)
        return self

    def __sub__(self, rhs):
        return self._to_py_tensor(self._data - self._decay_operand(rhs))

    def __rsub__(self, lhs):
        return self._to_py_tensor(self._decay_operand(lhs) - self._data)

    def __isub__(self, rhs):
        self._data -= self._decay_operand(rhs)
        return self

    def __mul__(self, rhs):
        return self._to_py_tensor(self._data * self._decay_operand(rhs))

    def __rmul__(self, lhs):
        return self._to_py_tensor(self._decay_operand(lhs) * self._data)

    def __imul__(self, rhs):
        self._data *= self._decay_operand(rhs)
        return self

    def __neg__(self):
        return self._to_py_tensor(-self._data)

    def __truediv__(self, rhs):
        return self._to_py_tensor(self._data / self._decay_operand(rhs))

    def __rtruediv__(self, lhs):
        return self._to_py_tensor(self._decay_operand(lhs) / self._data)

    def __itruediv__(self, rhs):
        self._data /= self._decay_operand(rhs)
        return self

    def __pow__(self, exponent):
        """Raise every element of the tensor to a power.

        Returns a new tensor whose elements are each raised to
        ``exponent``. A ``RuntimeWarning`` is emitted if the result
        contains NaN or infinity.

        Parameters
        ----------
        exponent : float or int
            The exponent.

        Returns
        -------
        Tensor
            A new tensor with transformed elements.
        """
        return self._to_py_tensor(self._data.__pow__(exponent))

    def __ipow__(self, exponent):
        """Raise every element of the tensor to a power in place.

        Parameters
        ----------
        exponent : float or int
            The exponent.

        Returns
        -------
        self
        """
        self._data.__ipow__(exponent)
        return self
