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

    # Set when the bars depend on a PyTorch, JAX, or TensorFlow array: all
    # bars, barcode after barcode in C order, as an (m, 2) framework tensor.
    _tvalues: Any = None

    def _tracked_parts(self):
        from .. import _autodiff
        from .barcode import Barcode

        if self._tvalues is None:
            return None
        # ponytail: only the (1,) tensors that the tracked kernel returns; serialize bars for more
        assert self.shape == (1,)
        return _autodiff._Tracked(
            [self._tvalues], lambda arrays: BarcodeTensor([Barcode(arrays[0])]), lambda grad: [grad])

    @classmethod
    def _from_tracked_bars(cls, bars: Any) -> BarcodeTensor:
        """A (1,) BarcodeTensor that carries (m, 2) framework bars. The barcode
        holds the bars when they are concrete and NaN while they are abstract."""
        from .. import _autodiff

        concrete = _autodiff._framework(bars).concrete(bars)
        if concrete is None:
            concrete = np.full(tuple(bars.shape), np.nan)
        out = cls([Barcode(np.asarray(concrete))])
        out._tvalues = bars
        return out

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
