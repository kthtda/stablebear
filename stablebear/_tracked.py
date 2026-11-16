"""Tensors and single objects that carry gradients of PyTorch.

A tensor type whose elements are blocks of rows of a fixed width, such as a
BarcodeTensor with its (birth, death) bars, opts in by setting a ``_tracking``
description on its class. ``tracked_class(T)`` is then the private subclass
of ``T`` for tensors that carry gradients. A single object, such as a Pcf
with its (time, value) breakpoints, opts in the same way, as the case of one
element without a shape.

The only state of a tracked object is its serial content, given as groups of
columns that are each a framework array or a constant NumPy array, and its
enumeration. Its C++ object is derived from them, so the two cannot disagree,
and kept until a framework array is changed in place. ``detach()`` gives the
untracked object of the current values. A type whose elements have a size
that their number of rows does not tell, such as a distance matrix, which
has no entries when it has no points and when it has one, also keeps the
size of each element; its serial content is then (content, enumeration,
sizes).

Indexing, reshaping, and copying a tracked tensor keep the gradients, and
give a copy, not a view. Reading ``_data`` raises, so any operation that
was not written for gradients raises instead of dropping them; code that
handles them takes the C++ tensor from ``_cpp()``. Comparisons, saving, and
pickling use the values. A single element, as from an integer index,
carries the gradients of its rows: a tracked Pcf or Barcode, or the
framework array of the points of a point cloud. A matrix is an untracked
element of the current values.

A write to a tracked tensor is recorded in a table of pending writes, and
the next use of the framework arrays applies all of them in one pass, so a
loop of n writes costs one pass and not n. Reading an element looks in the
table first.
"""

from __future__ import annotations

import inspect
import itertools
from typing import Any, Callable, NamedTuple

import numpy as np

from . import _sb_cpp as cpp
from ._tensor_base import Tensor, _infer_shape_and_flatten


class Tracking(NamedTuple):
    """How a type is tracked.

    ``groups`` lists, in column order, the groups of columns of the serial
    content that are tracked together: for each the name of the attribute
    that gives its framework array (None while it is a constant) and its
    number of columns. A group of one column is an (m,) array, a wider one
    (m, columns). A single group may have None for its number of columns:
    all columns of the content, however many, as for the coordinates of
    point clouds. ``from_serial(content, enumeration)`` builds an untracked
    object, and ``attributes`` gives the attributes that the type's
    constructor sets, such as ``dtype``, from the NumPy dtype of the
    content. For a tensor type, ``element(rows)`` builds an element from its
    rows. For a type that keeps sizes, these are ``from_serial(content,
    enumeration, sizes)`` and ``element(rows, size)``.
    ``tracked_element(parts)`` builds, from the parts of the rows of one
    element, the element that carries their gradients; without it, a single
    element is untracked.

    ``value_members`` names the methods and properties of the type whose
    results cannot carry gradients, such as ``to_numpy`` and ``__eq__``: a
    tracked object answers them with its values. ``dropping_members`` names
    methods that would give an object without the gradients by a way other
    than reading ``_data``, and so raise.
    """

    groups: tuple[tuple[str, int | None], ...]
    from_serial: Callable[..., Any]
    attributes: Callable[[Any], dict[str, Any]]
    element: Callable[..., Any] | None = None
    tracked_element: Callable[[list], Any] | None = None
    value_members: tuple[str, ...] = ()
    dropping_members: tuple[str, ...] = ()


def _is_framework(x: Any) -> bool:
    from . import _autodiff

    return _autodiff._framework(x) is not None


def _split_columns(content: np.ndarray, groups: tuple) -> list:
    """The serial content as one array per group of columns."""
    parts, start = [], 0
    for width in _widths(groups, content.shape[1]):
        parts.append(content[:, start] if width == 1 else content[:, start:start + width])
        start += width
    return parts


def _widths(groups: tuple, columns: int) -> list[int]:
    """The number of columns of each group, for content of the given number
    of columns."""
    return [columns if width is None else width for _, width in groups]


def _concrete_content(parts: list, groups: tuple, np_dtype: Any) -> np.ndarray:
    """The (m, k) serial content of the parts."""
    from . import _autodiff

    columns = []
    for part, (_, width) in zip(parts, groups):
        values = part
        if _is_framework(part):
            values = _autodiff._framework(part).to_numpy(part)
        values = np.asarray(values, dtype=np_dtype)
        columns.append(values if width is None else values.reshape(len(values), width))
    return columns[0] if len(columns) == 1 else np.concatenate(columns, axis=1)


class _TrackedObject:
    """The state of ``tracked_class(T)`` and what derives from it; see the
    module docstring."""

    _tracking: Tracking
    _base: type
    _parts: list
    _enumeration: np.ndarray
    _np_dtype: Any
    _plain: Any
    _plain_versions: Any
    _sizes: np.ndarray | None

    def __new__(cls, parts: list, enumeration: np.ndarray, plain: Any = None, sizes: Any = None):
        """``plain`` is the untracked object of these parts and enumeration,
        from an operation that already has it; otherwise the first use
        derives it. ``sizes`` gives the size of each element, for a type
        that keeps them."""
        # The state is set here and __init__ does nothing, so that the
        # constructor of the base type can return a tracked object.
        self = object.__new__(cls)
        self._sizes = None if sizes is None else np.asarray(sizes, dtype=np.int64)
        self._set_state(parts, enumeration)
        self._plain = plain
        self._plain_versions = self._versions() if plain is not None else None
        return self

    def _set_state(self, parts: list, enumeration: np.ndarray) -> None:
        self._parts = list(parts)
        self._enumeration = np.asarray(enumeration, dtype=np.int64)
        all_float32 = all("float32" in str(part.dtype) for part in parts)
        self._np_dtype = np.float32 if all_float32 else np.float64
        for name, value in self._tracking.attributes(self._np_dtype).items():
            setattr(self, name, value)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _versions(self) -> list:
        """What tells, per part, whether its values changed in place."""
        from . import _autodiff

        return [_autodiff._framework(part).version(part) if _is_framework(part) else None for part in self._parts]

    def _concrete(self) -> np.ndarray:
        """The (m, k) serial content."""
        return _concrete_content(self._parts, self._tracking.groups, self._np_dtype)

    def _untracked(self) -> Any:
        """The untracked object of the current values, shared between calls."""
        versions = self._versions()
        if self._plain is None or versions != self._plain_versions:
            self._plain = self._tracking.from_serial(*self._serial())
            self._plain_versions = versions
        return self._plain

    def _cpp(self) -> Any:
        """The C++ object of the current values."""
        return self._untracked()._data

    def _drops_gradients(self) -> NotImplementedError:
        name = self._base.__name__
        return NotImplementedError(
            f"This operation is not supported for a {name} that carries gradients: it would drop them. "
            f"Use .detach() for a {name} of its values, without gradients.")

    @property
    def _data(self):
        # Code that was not written for gradients reads _data, and so raises
        # instead of dropping them; code that handles them calls _cpp().
        raise self._drops_gradients()

    def detach(self) -> Any:
        """The untracked object of the current values, without gradients."""
        return self._tracking.from_serial(*self._serial())

    def _binary_io_data(self):
        return self._cpp()

    def _serial(self) -> tuple:
        sizes = () if self._sizes is None else (self._sizes,)
        return (self._concrete(), self._enumeration, *sizes)

    def to_serial_content(self) -> tuple:
        """The serial content with the content as one framework array, which
        carries the gradients; see the method of the base type."""
        from . import _autodiff

        like = next((part for part in self._parts if _is_framework(part)), None)
        if like is None:
            return self._serial()
        adapter = _autodiff._framework(like)
        columns = [part if _is_framework(part) else adapter.as_tensor(part, like=like) for part in self._parts]
        if len(columns) > 1:
            # Groups of one column each, as the times and values of PCFs.
            content = adapter.stack_columns(*columns)
        elif self._tracking.groups[0][1] == 1:
            content = adapter.reshape(columns[0], (-1, 1))
        else:
            content = columns[0]
        sizes = () if self._sizes is None else (self._sizes.copy(),)
        return (content, self._enumeration.copy(), *sizes)

    def _tracked_parts(self):
        from . import _autodiff

        tracked = [i for i, part in enumerate(self._parts) if _is_framework(part)]
        content, enumeration, *sizes = self._serial()
        starts = np.cumsum([0] + _widths(self._tracking.groups, content.shape[1]))
        versions = self._versions()

        def rebuild(arrays):
            arrays = [np.asarray(array).reshape(len(content), starts[i + 1] - starts[i]) for i, array in zip(tracked, arrays)]
            # Whether these are the values the content was read from.
            current = all(np.array_equal(array, content[:, starts[i]:starts[i + 1]]) for i, array in zip(tracked, arrays))
            if current and self._plain is not None and self._plain_versions == versions:
                return self._plain
            c = content.copy()
            for i, array in zip(tracked, arrays):
                c[:, starts[i]:starts[i + 1]] = array
            plain = self._tracking.from_serial(c, enumeration, *sizes)
            if current:
                self._plain, self._plain_versions = plain, versions
            return plain

        def split(grad):
            grad = np.asarray(grad)
            return [grad[:, starts[i]:starts[i + 1]].reshape(tuple(self._parts[i].shape)) for i in tracked]

        return _autodiff._Tracked([self._parts[i] for i in tracked], rebuild, split)

    def copy(self):
        # The state cannot change, so the copy can share it.
        return type(self)(
            self._parts, self._enumeration, plain=self._untracked() if self._plain is not None else None,
            sizes=self._sizes)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict=None):
        return self.copy()


class _TrackedMixin(_TrackedObject):
    """The behavior of ``tracked_class(T)`` for a tensor type; see the
    module docstring."""

    def detach(self) -> Any:
        """The untracked tensor of the current values, without gradients."""
        return self._base(self._cpp())

    # The state, stored as _p and _e, is read through _parts and
    # _enumeration, which first apply the pending writes: for the flat index
    # of each written element, the (parts, flat enumeration) of the value
    # written and the index of its element there.
    _p: list
    _e: np.ndarray
    _pending: dict[int, tuple[tuple[list, np.ndarray], int]] = {}

    @property
    def _parts(self) -> list:  # pyright: ignore[reportIncompatibleVariableOverride]
        self._flush()
        return self._p

    @_parts.setter
    def _parts(self, parts: list) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
        self._p = parts

    @property
    def _enumeration(self) -> np.ndarray:  # pyright: ignore[reportIncompatibleVariableOverride]
        self._flush()
        return self._e

    @_enumeration.setter
    def _enumeration(self, enumeration: np.ndarray) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
        self._e = enumeration

    def _source(self, val: Any) -> tuple[list, np.ndarray]:
        """The (parts, flat enumeration) of a value to write: a tensor or an
        element, with or without gradients, or the rows of an element."""
        if is_tracked(val):
            return val._parts, val._enumeration.reshape(-1, 2)
        if isinstance(val, Tensor):
            content, enumeration = val._serial()  # pyright: ignore[reportAttributeAccessIssue]
        else:
            content = np.asarray(val if isinstance(val, np.ndarray) else val.to_numpy())
            enumeration = np.array([0, len(content)])
        return _split_columns(content, self._tracking.groups), np.asarray(enumeration).reshape(-1, 2)

    def __setitem__(self, slices, val):
        if self._sizes is not None:
            raise NotImplementedError(
                f"Writing to a {self._base.__name__} that carries gradients is not supported.")
        self._validate_setitem_dtype(val)  # pyright: ignore[reportAttributeAccessIssue]
        ids = self._ids()[slices]
        source = self._source(val)
        if isinstance(val, Tensor):
            if not isinstance(ids, Tensor):
                raise ValueError("Cannot write a tensor to a single element.")
            elements = np.arange(len(source[1])).reshape(tuple(val.shape))
            elements = np.broadcast_to(elements, np.asarray(ids).shape).ravel()
        else:
            elements = itertools.repeat(0)
        if not self._pending:
            self._pending = {}
        for target, element in zip(np.asarray(ids).ravel() if isinstance(ids, Tensor) else [ids], elements):
            self._pending[int(target)] = (source, int(element))

    def _flush(self) -> None:
        """Apply the pending writes: per group of columns, one concatenation
        of this tensor's rows with those of the written values, and one
        gather of the rows of each element from where it now lives."""
        from . import _autodiff

        if not self._pending:
            return
        pending, self._pending = self._pending, {}
        flat = self._e.reshape(-1, 2)
        sources = [self._p]
        number = {}
        source_of = np.zeros(len(flat), dtype=np.int64)
        starts, stops = flat[:, 0].copy(), flat[:, 1].copy()
        for target, ((parts, enumeration), element) in pending.items():
            if id(parts) not in number:
                number[id(parts)] = len(sources)
                sources.append(parts)
            source_of[target] = number[id(parts)]
            starts[target], stops[target] = enumeration[element]
        lengths = [int(parts[0].shape[0]) for parts in sources]
        offsets = np.cumsum([0] + lengths)
        counts = stops - starts
        ends = np.cumsum(counts)
        rows = np.arange(int(counts.sum()), dtype=np.int64) + np.repeat(
            offsets[source_of] + starts - (ends - counts), counts)

        new_parts = [
            _autodiff._framework(part).gather(part, rows) if _is_framework(part) else part[rows]
            for part in _concatenated(sources)]
        self._set_state(new_parts, np.stack([ends - counts, ends], axis=-1).reshape(self._e.shape))
        self._plain = None

    @property
    def shape(self):
        return cpp.Shape(list(self._e.shape[:-1]))

    @property
    def ndim(self) -> int:
        return self._e.ndim - 1

    @property
    def size(self) -> int:
        return int(np.prod(self._e.shape[:-1]))

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return int(self._e.shape[0])

    def _ids(self):
        """An IntTensor of the flat indices of the elements, in this shape.
        Indexing and reshaping it as this tensor selects the elements."""
        from .base_tensor import IntTensor

        return IntTensor(np.arange(self.size, dtype=np.int64).reshape(self._e.shape[:-1]))

    def _select(self, ids: Any) -> Any:
        """The elements with flat indices ids: for an IntTensor, a tracked
        tensor of its shape that gathers their rows; for one index, the
        element, which carries the gradients of its rows where the type has
        a tracked element."""
        from . import _autodiff

        if not isinstance(ids, Tensor):
            # Without applying the pending writes: a written element is read
            # from the value written.
            (parts, enumeration), element = self._pending.get(int(ids), ((self._p, self._e.reshape(-1, 2)), int(ids)))
            start, stop = enumeration[element]
            if self._tracking.tracked_element is not None and any(_is_framework(part) for part in parts):
                rows = np.arange(start, stop)
                return self._tracking.tracked_element([
                    _autodiff._framework(part).gather(part, rows) if _is_framework(part) else part[start:stop]
                    for part in parts])
            assert self._tracking.element is not None
            size = () if self._sizes is None else (int(self._sizes.reshape(-1)[element]),)
            return self._tracking.element(
                _concrete_content(parts, self._tracking.groups, self._np_dtype)[start:stop], *size)
        flat = self._enumeration.reshape(-1, 2)
        ids = np.asarray(ids, dtype=np.int64)
        starts = flat[ids.ravel(), 0]
        counts = flat[ids.ravel(), 1] - starts
        ends = np.cumsum(counts)
        rows = np.arange(int(counts.sum()), dtype=np.int64) + np.repeat(starts - (ends - counts), counts)
        parts = [
            _autodiff._framework(part).gather(part, rows) if _is_framework(part) else part[rows]
            for part in self._parts
        ]
        sizes = None if self._sizes is None else self._sizes.reshape(-1)[ids.ravel()].reshape(ids.shape)
        return type(self)(parts, np.stack([ends - counts, ends], axis=-1).reshape(*ids.shape, 2), sizes=sizes)

    def __getitem__(self, slices):
        return self._select(self._ids()[slices])

    def astype(self, dtype):
        raise NotImplementedError(f"astype of a {self._base.__name__} that carries gradients is not supported.")

    def _to_py_tensor(self, data):
        raise self._drops_gradients()


def _shape_method(name: str):
    def method(self, *args, **kwargs):
        return self._select(getattr(self._ids(), name)(*args, **kwargs))

    method.__name__ = name
    return method


for _name in ("reshape", "flatten", "transpose", "swapaxes", "squeeze", "expand_dims", "broadcast_to"):
    setattr(_TrackedMixin, _name, _shape_method(_name))


def _values_method(name: str):
    """A method whose result cannot carry gradients, on the values."""
    def method(self, *args, **kwargs):
        return getattr(self._untracked(), name)(*(arg._untracked() if is_tracked(arg) else arg for arg in args), **kwargs)

    method.__name__ = name
    return method


def _dropping_method(name: str):
    def method(self, *args, **kwargs):
        raise self._drops_gradients()

    method.__name__ = name
    return method


for _name in ("__eq__", "__ne__", "array_equal"):
    setattr(_TrackedMixin, _name, _values_method(_name))
del _name


def _part_property(index: int):
    return property(lambda self: self._parts[index] if _is_framework(self._parts[index]) else None)


_TRACKED_CLASSES: dict[type, Any] = {}


def tracked_class(base: type) -> Any:
    """The subclass of the type ``base`` for objects that carry gradients.
    Its constructor takes the parts, one framework or NumPy array per group
    of ``base._tracking``, and the enumeration."""
    if base not in _TRACKED_CLASSES:
        tracking = base._tracking  # pyright: ignore[reportAttributeAccessIssue]
        mixin = _TrackedMixin if issubclass(base, Tensor) else _TrackedObject
        namespace: dict[str, Any] = {"_base": base, "__doc__": mixin.__doc__}
        for index, (name, _) in enumerate(tracking.groups):
            namespace[name] = _part_property(index)
        for name in tracking.value_members:
            if isinstance(inspect.getattr_static(base, name), property):
                namespace[name] = property(lambda self, name=name: getattr(self._untracked(), name))
            else:
                namespace[name] = _values_method(name)
        for name in tracking.dropping_members:
            namespace[name] = _dropping_method(name)
        _TRACKED_CLASSES[base] = type(f"_Tracked{base.__name__}", (mixin, base), namespace)
    return _TRACKED_CLASSES[base]


def is_tracked(x: Any) -> bool:
    return isinstance(x, _TrackedObject)


def _concatenated(sources: list) -> list:
    """The parts of several sources, each a list of one array per group of
    columns, as one array per group: a framework array if a source has one
    for the group."""
    from . import _autodiff

    combined = []
    for arrays in zip(*sources):
        like = next((array for array in arrays if _is_framework(array)), None)
        if like is None:
            combined.append(np.concatenate(arrays))
            continue
        adapter = _autodiff._framework(like)
        combined.append(adapter.concat([
            array if _is_framework(array) else adapter.as_tensor(array, like=like) for array in arrays]))
    return combined


def joined(tensors: Any, join: Any) -> Any:
    """Tensors of one type, of which at least one carries gradients, joined
    into one that carries them.

    ``join`` joins IntTensors of the shapes of the tensors, as it would the
    tensors: concatenation or stacking along an axis. It is called on the
    indices of the elements, numbered through the tensors, and the result
    takes the rows of the elements at those indices: one concatenation and
    one gather per group of columns.
    """
    from .base_tensor import IntTensor

    base = next(tensor for tensor in tensors if is_tracked(tensor))._base
    groups = base._tracking.groups
    sources, enumerations, sizes, ids = [], [], [], []
    n_rows = n_elements = 0
    for tensor in tensors:
        if not isinstance(tensor, base) or tensor.dtype != tensors[0].dtype:
            raise TypeError(
                f"All tensors must have one type and dtype to be joined (got {type(tensors[0]).__name__} of "
                f"{tensors[0].dtype.__name__} and {type(tensor).__name__} of {tensor.dtype.__name__}).")
        if is_tracked(tensor):
            parts, enumeration, size = tensor._parts, tensor._enumeration, tensor._sizes
        else:
            content, enumeration, *rest = tensor._serial()
            parts, size = _split_columns(content, groups), rest[0] if rest else None
        flat = enumeration.reshape(-1, 2)
        ids.append(IntTensor(np.arange(len(flat), dtype=np.int64).reshape(enumeration.shape[:-1]) + n_elements))
        sources.append(parts)
        enumerations.append(flat + n_rows)
        sizes.append(None if size is None else size.reshape(-1))
        n_rows += int(parts[0].shape[0])
        n_elements += len(flat)
    every = tracked_class(base)(
        _concatenated(sources), np.concatenate(enumerations),
        sizes=None if sizes[0] is None else np.concatenate(sizes))
    return every._select(join(ids))


def from_elements(base: type, data: Any) -> Any:
    """The tracked tensor of type ``base`` of a nested list of elements, if
    an element carries gradients, and None otherwise. Every element is
    written to an empty tensor, so the tensor is built in one pass."""
    shape, flat = _infer_shape_and_flatten(data)
    if not any(is_tracked(element) for element in flat):
        return None
    groups = base._tracking.groups  # pyright: ignore[reportAttributeAccessIssue]
    tensor = tracked_class(base)(
        [np.zeros(0) if width == 1 else np.zeros((0, width)) for _, width in groups], np.zeros(shape + (2,)))
    tensor._pending = {index: (tensor._source(element), 0) for index, element in enumerate(flat)}
    tensor._flush()
    return tensor
