#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import numpy as np

from .. import _sb_cpp as cpp
from ..typing import barcode32, barcode64

cpp_p = cpp.persistence


class Barcode:
    """A persistence barcode, i.e. a collection of bars (birth-death intervals).

    A barcode is represented as an (n, 2) array where each row is a bar
    given as a ``(birth, death)`` pair from a persistent homology computation.

    Parameters
    ----------
    bc : Barcode or numpy.ndarray
        An existing ``Barcode`` to copy, or an (n, 2) NumPy array of
        ``(birth, death)`` pairs (``float32`` or ``float64``).
    """

    def __init__(self, bc):
        fail = False
        if isinstance(bc, Barcode):
            self._data = bc._data
        elif isinstance(bc, cpp_p.Barcode32 | cpp_p.Barcode64):
            self._data = bc
        elif isinstance(bc, np.ndarray):
            if bc.dtype == np.float32:
                self._data = cpp_p.Barcode32(bc)
            elif bc.dtype == np.float64:
                self._data = cpp_p.Barcode64(bc)
            else:
                fail = True
        else:
            fail = True

        if isinstance(self._data, cpp_p.Barcode32):
            self.dtype = barcode32
        elif isinstance(self._data, cpp_p.Barcode64):
            self.dtype = barcode64
        else:
            fail = True

        if fail:
            raise TypeError(
                f"Barcode cannot be constructed with object of type {type(bc)}"
            )

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

    def __reduce__(self):
        import io as _io
        from ..io import _save_object, _unpickle_object
        buf = _io.BytesIO()
        _save_object(self, buf)
        return _unpickle_object, (buf.getvalue(),)

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
