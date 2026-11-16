"""Tensors that carry gradients of PyTorch, JAX, or TensorFlow.

A tensor type whose elements are blocks of rows of a fixed width, such as a
BarcodeTensor with its (birth, death) bars, opts in by setting a ``_tracking``
description on its class. ``tracked_class(T)`` is then the private subclass
of ``T`` for tensors that carry gradients.

The only state of such a tensor is its serial content, given as groups of
columns that are each a framework array or a constant NumPy array, and its
enumeration. Its C++ tensor is built from them on each use, so the two cannot
disagree. It cannot be changed in place. Indexing, reshaping, and copying
keep the gradients; other operations that would drop them raise. A single
element, as from an integer index, is an untracked element object of the
current values.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np

from . import _sb_cpp as cpp
from ._tensor_base import Tensor


class Tracking(NamedTuple):
    """How a tensor type is tracked.

    ``groups`` lists, in column order, the groups of columns of the serial
    content that are tracked together: for each the name of the attribute
    that gives its framework array (None while it is a constant) and its
    number of columns. A group of one column is an (m,) array, a wider one
    (m, columns). ``from_serial(content, enumeration)`` builds an untracked
    tensor, ``element(rows)`` an element from its rows, and ``dtype`` gives
    the tensor's dtype from the NumPy dtype of its content.
    """

    groups: tuple[tuple[str, int], ...]
    from_serial: Callable[[np.ndarray, np.ndarray], Any]
    element: Callable[[np.ndarray], Any]
    dtype: Callable[[Any], Any]


def _is_framework(x: Any) -> bool:
    from . import _autodiff

    return _autodiff._framework(x) is not None


class _TrackedMixin:
    """The behavior of ``tracked_class(T)``; see the module docstring."""

    _tracking: Tracking
    _base: type

    def __init__(self, parts: list, enumeration: np.ndarray):
        # Not the constructor of the tensor type: _data is derived, not stored.
        self._parts = list(parts)
        self._enumeration = np.asarray(enumeration, dtype=np.int64)
        all_float32 = all("float32" in str(part.dtype) for part in self._parts)
        self._np_dtype = np.float32 if all_float32 else np.float64
        self.dtype = self._tracking.dtype(self._np_dtype)

    def _concrete(self) -> np.ndarray:
        """The (m, k) serial content, with NaN where a framework traces it abstractly."""
        from . import _autodiff

        columns = []
        for part in self._parts:
            values = part
            if _is_framework(part):
                values = _autodiff._framework(part).concrete(part)
                if values is None:
                    values = np.full(tuple(part.shape), np.nan)
            values = np.asarray(values, dtype=self._np_dtype)
            columns.append(values.reshape(len(values), -1))
        return columns[0] if len(columns) == 1 else np.concatenate(columns, axis=1)

    @property
    def _data(self):
        return self._tracking.from_serial(self._concrete(), self._enumeration)._data

    def _serial(self) -> tuple[np.ndarray, np.ndarray]:
        return self._concrete(), self._enumeration

    def _tracked_parts(self):
        from . import _autodiff

        tracked = [i for i, part in enumerate(self._parts) if _is_framework(part)]
        starts = np.cumsum([0] + [columns for _, columns in self._tracking.groups])
        content, enumeration = self._serial()

        def rebuild(arrays):
            c = content.copy()
            for i, array in zip(tracked, arrays):
                c[:, starts[i]:starts[i + 1]] = np.asarray(array).reshape(len(c), -1)
            return self._tracking.from_serial(c, enumeration)

        def split(grad):
            grad = np.asarray(grad)
            return [grad[:, starts[i]:starts[i + 1]].reshape(tuple(self._parts[i].shape)) for i in tracked]

        return _autodiff._Tracked([self._parts[i] for i in tracked], rebuild, split)

    @property
    def shape(self):
        return cpp.Shape(list(self._enumeration.shape[:-1]))

    @property
    def ndim(self) -> int:
        return self._enumeration.ndim - 1

    @property
    def size(self) -> int:
        return int(np.prod(self._enumeration.shape[:-1]))

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return int(self._enumeration.shape[0])

    def _ids(self):
        """An IntTensor of the flat indices of the elements, in this shape.
        Indexing and reshaping it as this tensor selects the elements."""
        from .base_tensor import IntTensor

        return IntTensor(np.arange(self.size, dtype=np.int64).reshape(self._enumeration.shape[:-1]))

    def _select(self, ids: Any) -> Any:
        """The elements with flat indices ids: for an IntTensor, a tracked
        tensor of its shape that gathers their rows; for one index, an
        untracked element of its current rows."""
        from . import _autodiff

        flat = self._enumeration.reshape(-1, 2)
        if not isinstance(ids, Tensor):
            start, stop = flat[ids]
            return self._tracking.element(self._concrete()[start:stop])
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

    def copy(self):
        # Immutable, so the copy can share the parts.
        return type(self)(self._parts, self._enumeration)

    def __deepcopy__(self, memodict=None):
        return self.copy()

    def __setitem__(self, slices, val):
        raise TypeError(f"A {self._base.__name__} that carries gradients cannot be changed in place.")

    def astype(self, dtype):
        raise NotImplementedError(f"astype of a {self._base.__name__} that carries gradients is not supported.")

    def _to_py_tensor(self, data):
        raise NotImplementedError(
            f"This operation is not supported for a {self._base.__name__} that carries gradients: "
            "it would drop them.")


def _shape_method(name: str):
    def method(self, *args, **kwargs):
        return self._select(getattr(self._ids(), name)(*args, **kwargs))

    method.__name__ = name
    return method


for _name in ("reshape", "flatten", "transpose", "swapaxes", "squeeze", "expand_dims", "broadcast_to"):
    setattr(_TrackedMixin, _name, _shape_method(_name))
del _name


def _part_property(index: int):
    return property(lambda self: self._parts[index] if _is_framework(self._parts[index]) else None)


_TRACKED_CLASSES: dict[type, Any] = {}


def tracked_class(base: type) -> Any:
    """The subclass of the tensor type ``base`` for tensors that carry
    gradients. Its constructor takes the parts, one framework or NumPy array
    per group of ``base._tracking``, and the enumeration."""
    if base not in _TRACKED_CLASSES:
        tracking = base._tracking  # pyright: ignore[reportAttributeAccessIssue]
        namespace: dict[str, Any] = {"_base": base, "__doc__": _TrackedMixin.__doc__}
        for index, (name, _) in enumerate(tracking.groups):
            namespace[name] = _part_property(index)
        _TRACKED_CLASSES[base] = type(f"_Tracked{base.__name__}", (_TrackedMixin, base), namespace)
    return _TRACKED_CLASSES[base]


def is_tracked(x: Any) -> bool:
    return isinstance(x, _TrackedMixin)
