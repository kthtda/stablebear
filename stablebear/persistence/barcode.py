from __future__ import annotations

import numpy as np

from .. import _sb_cpp as cpp
from .._binary_io import _BinaryIoMixin
from ..typing import barcode32, barcode64

cpp_p = cpp.persistence


def _as_float64_bars(bc):
    try:
        arr = np.asarray(bc)
    except ValueError as e:
        raise TypeError(f"Barcode cannot be constructed from {type(bc)}: {e}") from None
    if arr.dtype.kind not in "iuf":
        raise TypeError(
            f"Barcode cannot be constructed from {type(bc)} with dtype {arr.dtype}; "
            "expected real-valued (birth, death) pairs"
        )
    if arr.shape == (0,):
        return np.zeros((0, 2), dtype=np.float64)
    return arr.astype(np.float64, copy=False)


class Barcode(_BinaryIoMixin):
    """A persistence barcode, i.e. a collection of bars (birth-death intervals).

    A barcode is represented as an (n, 2) array where each row is a bar
    given as a ``(birth, death)`` pair from a persistent homology computation.

    Parameters
    ----------
    bc : Barcode or array_like
        An existing ``Barcode`` to copy, or (n, 2) ``(birth, death)`` pairs,
        such as a nested list or an array. A ``float32`` array gives a
        ``barcode32``; any other real-valued input gives a ``barcode64``.
    """

    def _binary_io_data(self):
        return self._data

    def __init__(self, bc):
        if isinstance(bc, Barcode):
            self._data = bc._data
        elif isinstance(bc, cpp_p.Barcode32 | cpp_p.Barcode64):
            self._data = bc
        elif isinstance(bc, np.ndarray) and bc.dtype == np.float32:
            self._data = cpp_p.Barcode32(bc)
        else:
            self._data = cpp_p.Barcode64(_as_float64_bars(bc))

        if isinstance(self._data, cpp_p.Barcode32):
            self.dtype = barcode32
        else:
            self.dtype = barcode64

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__repr__()

    def to_numpy(self):
        """Return the bars as an (n, 2) NumPy array of ``(birth, death)`` pairs."""
        return np.asarray(self._data).copy()

    def __len__(self):
        return len(self._data)

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self._data)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def is_isomorphic_to(self, bc: Barcode, atol: float = 1e-8, rtol: float = 1e-5):
        """Check whether two barcodes are isomorphic (same multiset of bars).

        Bar endpoints are compared with a numerical tolerance so that barcodes
        computed via different but mathematically equivalent routes (for example,
        from a point cloud versus a precomputed distance matrix) still compare
        equal despite floating-point rounding. Two endpoints ``a`` and ``b`` are
        considered equal when ``abs(a - b) <= atol + rtol * abs(b)``; infinite
        endpoints must match exactly.

        Parameters
        ----------
        bc : Barcode
            The barcode to compare against.
        atol : float, optional
            Absolute tolerance for endpoint comparison, by default ``1e-8``.
        rtol : float, optional
            Relative tolerance for endpoint comparison, by default ``1e-5``.
            Pass ``atol=0, rtol=0`` for a bitwise comparison.
        """
        return self._data.is_isomorphic_to(bc._data, atol, rtol)
