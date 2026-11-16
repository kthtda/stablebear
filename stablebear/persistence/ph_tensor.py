from __future__ import annotations

from typing import Any

import numpy as np

from .. import _sb_cpp as cpp
from .._tensor_base import Tensor, _tensor_from_nested
from ..typing import barcode32, barcode64
from .barcode import Barcode

cpp_p = cpp.persistence

_BARCODE_CPP_TO_DTYPE = {
    cpp_p.Barcode32Tensor: barcode32,
    cpp_p.Barcode64Tensor: barcode64,
}


class BarcodeTensor(Tensor):
    def __init__(self, data):
        super().__init__()
        if isinstance(data, _TrackedBarcodeTensor):
            raise TypeError(
                "This BarcodeTensor carries gradients, which a copy into a plain BarcodeTensor would drop. "
                "Use .copy() to keep them.")
        if isinstance(data, BarcodeTensor):
            data = data._data
        elif isinstance(data, (list, tuple)):
            data = _tensor_from_nested(data, {
                cpp_p.Barcode32: cpp_p.Barcode32Tensor,
                cpp_p.Barcode64: cpp_p.Barcode64Tensor,
            })
        elif not isinstance(data, (cpp_p.Barcode32Tensor, cpp_p.Barcode64Tensor)):
            raise TypeError(f"Cannot create BarcodeTensor from {type(data)}")
        self._data = data
        self.dtype = _BARCODE_CPP_TO_DTYPE[type(self._data)]

    # The framework bars of a _TrackedBarcodeTensor; None for any other.
    _tvalues: Any = None

    def _serial(self) -> tuple[np.ndarray, np.ndarray]:
        """(content, enumeration): the (m, 2) bars of all barcodes in C order,
        and for each barcode the [start, stop) rows of its bars, with shape
        self.shape + (2,)."""
        return cpp.barcodes_to_serial_content(self._data)

    @staticmethod
    def _from_serial(content: np.ndarray, enumeration: np.ndarray) -> BarcodeTensor:
        """The inverse of _serial."""
        content = np.asarray(content)
        enumeration = np.ascontiguousarray(enumeration, dtype=np.int64)
        if content.dtype == np.float32:
            return BarcodeTensor(cpp.barcodes_from_serial_content_32(np.ascontiguousarray(content), enumeration))
        return BarcodeTensor(cpp.barcodes_from_serial_content_64(
            np.ascontiguousarray(content, dtype=np.float64), enumeration))

    def _to_py_tensor(self, data):
        return BarcodeTensor(data)

    def _decay_value(self, val):
        if isinstance(val, np.ndarray):
            return Barcode(val)._data
        return val._data

    def _represent_element(self, element):
        return Barcode(element)

    def _get_valid_setitem_dtypes(self):
        return [BarcodeTensor, Barcode, np.ndarray]

    def is_isomorphic_to(
        self, other: BarcodeTensor, atol: float = 1e-8, rtol: float = 1e-5
    ) -> bool:
        """Check aligned barcodes for isomorphism across the whole tensor.

        The tensors must have the same outer shape and barcode dtype. Each
        aligned pair of barcodes is compared as an order-independent multiset
        of bars using the supplied endpoint tolerances.

        Parameters
        ----------
        other : BarcodeTensor
            The barcode tensor to compare against.
        atol : float, optional
            Absolute endpoint tolerance, by default ``1e-8``. For an endpoint
            ``a`` in this tensor and the corresponding endpoint ``b`` in
            *other*, the absolute contribution permits a fixed difference of
            up to ``atol``.
        rtol : float, optional
            Relative endpoint tolerance, by default ``1e-5``. The endpoints
            match when ``abs(a - b) <= atol + rtol * abs(b)``. Infinite
            endpoints must match exactly. Pass ``atol=0, rtol=0`` for exact
            endpoint comparison.

        Returns
        -------
        bool
            Whether every aligned pair of barcodes is isomorphic.

        Raises
        ------
        TypeError
            If *other* is not a ``BarcodeTensor`` or has a different dtype.
        ValueError
            If the outer tensor shapes differ.
        """
        if not isinstance(other, BarcodeTensor):
            raise TypeError(
                "BarcodeTensor.is_isomorphic_to expects another BarcodeTensor"
            )
        if self.dtype != other.dtype:
            raise TypeError(
                "Barcode tensors must have the same dtype, got "
                f"{self.dtype.__name__} and {other.dtype.__name__}"
            )
        return cpp_p.barcode_tensors_are_isomorphic(
            self._data, other._data, atol, rtol
        )


class _TrackedBarcodeTensor(BarcodeTensor):
    """A BarcodeTensor whose bars depend on PyTorch, JAX, or TensorFlow arrays.

    Its only state is the framework bars, an (m, 2) tensor of the bars of all
    barcodes in C order, and the enumeration of _serial. Its C++ tensor is
    built from them on each use, so the two cannot disagree. It cannot be
    changed in place. Indexing, reshaping, and copying keep the gradients;
    other operations that would drop them raise. A single barcode, as from
    an integer index, is a plain Barcode of the current values.
    """

    def __init__(self, bars: Any, enumeration: np.ndarray):
        # Not BarcodeTensor.__init__: _data is derived, not stored.
        self._tvalues = bars
        self._enumeration = np.asarray(enumeration, dtype=np.int64)
        self.dtype = barcode32 if "float32" in str(bars.dtype) else barcode64

    def _concrete(self) -> np.ndarray:
        """The bars, or NaN while a framework traces them abstractly."""
        from .. import _autodiff

        concrete = _autodiff._framework(self._tvalues).concrete(self._tvalues)
        if concrete is None:
            concrete = np.full(tuple(self._tvalues.shape), np.nan)
        return np.asarray(concrete, dtype=np.float32 if self.dtype == barcode32 else np.float64)

    @property
    def _data(self):  # pyright: ignore[reportIncompatibleVariableOverride]
        return BarcodeTensor._from_serial(self._concrete(), self._enumeration)._data

    def _serial(self) -> tuple[np.ndarray, np.ndarray]:
        return self._concrete(), self._enumeration

    def _tracked_parts(self):
        from .. import _autodiff

        enumeration = self._enumeration
        return _autodiff._Tracked(
            [self._tvalues], lambda arrays: BarcodeTensor._from_serial(arrays[0], enumeration), lambda grad: [grad])

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
        """An IntTensor of the flat indices of the barcodes, in this shape.
        Indexing and reshaping it as this tensor selects the barcodes."""
        from ..base_tensor import IntTensor

        return IntTensor(np.arange(self.size, dtype=np.int64).reshape(self._enumeration.shape[:-1]))

    def _select(self, ids: Any) -> Any:
        """The barcodes with flat indices ids: for an IntTensor, a tracked
        tensor of its shape that gathers their framework bars; for one
        index, a Barcode of its current bars."""
        from .. import _autodiff

        flat = self._enumeration.reshape(-1, 2)
        if not isinstance(ids, Tensor):
            start, stop = flat[ids]
            return Barcode(self._concrete()[start:stop])
        ids = np.asarray(ids, dtype=np.int64)
        starts = flat[ids.ravel(), 0]
        counts = flat[ids.ravel(), 1] - starts
        ends = np.cumsum(counts)
        rows = np.arange(int(counts.sum()), dtype=np.int64) + np.repeat(starts - (ends - counts), counts)
        bars = _autodiff._framework(self._tvalues).gather(self._tvalues, rows)
        return _TrackedBarcodeTensor(bars, np.stack([ends - counts, ends], axis=-1).reshape(*ids.shape, 2))

    def __getitem__(self, slices):
        return self._select(self._ids()[slices])

    def copy(self):
        # Immutable, so the copy can share the framework bars.
        return _TrackedBarcodeTensor(self._tvalues, self._enumeration)

    def __deepcopy__(self, memodict=None):
        return self.copy()

    def __setitem__(self, slices, val):
        raise TypeError("A BarcodeTensor that carries gradients cannot be changed in place.")

    def astype(self, dtype):
        raise NotImplementedError("astype of a BarcodeTensor that carries gradients is not supported.")

    def _to_py_tensor(self, data):
        raise NotImplementedError(
            "This operation is not supported for a BarcodeTensor that carries gradients: it would drop them.")


def _shape_method(name: str):
    def method(self, *args, **kwargs):
        return self._select(getattr(self._ids(), name)(*args, **kwargs))

    method.__name__ = name
    return method


for _name in ("reshape", "flatten", "transpose", "swapaxes", "squeeze", "expand_dims", "broadcast_to"):
    setattr(_TrackedBarcodeTensor, _name, _shape_method(_name))
del _name
