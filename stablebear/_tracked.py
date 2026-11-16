"""Tensors and single objects that carry gradients of PyTorch, JAX, or
TensorFlow.

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
untracked object of the current values.

Indexing, reshaping, and copying a tracked tensor keep the gradients, and
give a copy, not a view. Reading ``_data`` raises, so any operation that
was not written for gradients raises instead of dropping them; code that
handles them takes the C++ tensor from ``_cpp()``. Comparisons, saving, and
pickling use the values. A single element, as from an integer index, is an
untracked element object of the current values.

A write to a tracked tensor is recorded in a table of pending writes, and
the next use of the framework arrays applies all of them in one pass, so a
loop of n writes costs one pass and not n. Reading an element looks in the
table first.
"""

from __future__ import annotations

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
    rows.
    """

    groups: tuple[tuple[str, int | None], ...]
    from_serial: Callable[[np.ndarray, np.ndarray], Any]
    attributes: Callable[[Any], dict[str, Any]]
    element: Callable[[np.ndarray], Any] | None = None


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
    """The (m, k) serial content of the parts, with NaN where a framework
    traces it abstractly."""
    from . import _autodiff

    columns = []
    for part, (_, width) in zip(parts, groups):
        values = part
        if _is_framework(part):
            values = _autodiff._framework(part).concrete(part)
            if values is None:
                values = np.full(tuple(part.shape), np.nan)
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

    def __new__(cls, parts: list, enumeration: np.ndarray, plain: Any = None):
        """``plain`` is the untracked object of these parts and enumeration,
        from an operation that already has it; otherwise the first use
        derives it."""
        # The state is set here and __init__ does nothing, so that the
        # constructor of the base type can return a tracked object.
        self = object.__new__(cls)
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
        """The (m, k) serial content, with NaN where a framework traces it abstractly."""
        return _concrete_content(self._parts, self._tracking.groups, self._np_dtype)

    def _untracked(self) -> Any:
        """The untracked object of the current values, shared between calls."""
        versions = self._versions()
        if self._plain is None or versions != self._plain_versions:
            self._plain = self._tracking.from_serial(self._concrete(), self._enumeration)
            self._plain_versions = versions
        return self._plain

    def _cpp(self) -> Any:
        """The C++ object of the current values."""
        return self._untracked()._data

    @property
    def _data(self):
        return self._cpp()

    def detach(self) -> Any:
        """The untracked object of the current values, without gradients."""
        return self._tracking.from_serial(self._concrete(), self._enumeration)

    def _binary_io_data(self):
        return self._cpp()

    def _serial(self) -> tuple[np.ndarray, np.ndarray]:
        return self._concrete(), self._enumeration

    def _tracked_parts(self):
        from . import _autodiff

        tracked = [i for i, part in enumerate(self._parts) if _is_framework(part)]
        content, enumeration = self._serial()
        starts = np.cumsum([0] + _widths(self._tracking.groups, content.shape[1]))
        versions = self._versions()

        def rebuild(arrays):
            arrays = [np.asarray(array).reshape(len(content), starts[i + 1] - starts[i]) for i, array in zip(tracked, arrays)]
            # Whether these are the values the content was read from, as
            # when the framework is not tracing. The parts are not read
            # here: TensorFlow forbids reading a variable inside its
            # operations.
            current = all(np.array_equal(array, content[:, starts[i]:starts[i + 1]]) for i, array in zip(tracked, arrays))
            if current and self._plain is not None and self._plain_versions == versions:
                return self._plain
            c = content.copy()
            for i, array in zip(tracked, arrays):
                c[:, starts[i]:starts[i + 1]] = array
            plain = self._tracking.from_serial(c, enumeration)
            if current:
                self._plain, self._plain_versions = plain, versions
            return plain

        def split(grad):
            grad = np.asarray(grad)
            return [grad[:, starts[i]:starts[i + 1]].reshape(tuple(self._parts[i].shape)) for i in tracked]

        return _autodiff._Tracked([self._parts[i] for i in tracked], rebuild, split)

    def copy(self):
        # The state cannot change, so the copy can share it.
        return type(self)(self._parts, self._enumeration, plain=self._untracked() if self._plain is not None else None)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict=None):
        return self.copy()


class _TrackedMixin(_TrackedObject):
    """The behavior of ``tracked_class(T)`` for a tensor type; see the
    module docstring."""

    def _drops_gradients(self) -> NotImplementedError:
        name = self._base.__name__
        return NotImplementedError(
            f"This operation is not supported for a {name} that carries gradients: it would drop them. "
            f"Use .detach() for a {name} of its values, without gradients.")

    @property
    def _data(self):
        raise self._drops_gradients()

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

        new_parts = []
        for group in range(len(self._tracking.groups)):
            arrays = [parts[group] for parts in sources]
            like = next((array for array in arrays if _is_framework(array)), None)
            if like is None:
                new_parts.append(np.concatenate(arrays)[rows])
                continue
            adapter = _autodiff._framework(like)
            combined = adapter.concat([
                array if _is_framework(array) else adapter.as_tensor(array, like=like) for array in arrays])
            new_parts.append(adapter.gather(combined, rows))
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
        tensor of its shape that gathers their rows; for one index, an
        untracked element of its current rows."""
        from . import _autodiff

        if not isinstance(ids, Tensor):
            # Without applying the pending writes: a written element is read
            # from the value written.
            (parts, enumeration), element = self._pending.get(int(ids), ((self._p, self._e.reshape(-1, 2)), int(ids)))
            start, stop = enumeration[element]
            assert self._tracking.element is not None
            return self._tracking.element(
                _concrete_content(parts, self._tracking.groups, self._np_dtype)[start:stop])
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
        return type(self)(parts, np.stack([ends - counts, ends], axis=-1).reshape(*ids.shape, 2))

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
    def method(self, rhs):
        return getattr(self.detach(), name)(rhs.detach() if is_tracked(rhs) else rhs)

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
        _TRACKED_CLASSES[base] = type(f"_Tracked{base.__name__}", (mixin, base), namespace)
    return _TRACKED_CLASSES[base]


def is_tracked(x: Any) -> bool:
    return isinstance(x, _TrackedObject)


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
