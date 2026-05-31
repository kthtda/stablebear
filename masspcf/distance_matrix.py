#    Copyright 2024-2026 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _mpcf_cpp as cpp
from ._tensor_base import Tensor
from .typing import float32, float64, distmat32, distmat64

if TYPE_CHECKING:
    CppDistanceMatrix = cpp.DistanceMatrix_f32 | cpp.DistanceMatrix_f64

_dtype_to_cpp = {
    float32: cpp.DistanceMatrix_f32,
    float64: cpp.DistanceMatrix_f64,
}

_cpp_types = (cpp.DistanceMatrix_f32, cpp.DistanceMatrix_f64)

_CPP_TO_DTYPE = {
    cpp.DistanceMatrix_f32: float32,
    cpp.DistanceMatrix_f64: float64,
}

_DISTMAT_CPP_TO_DTYPE = {
    cpp.DistanceMatrix32Tensor: distmat32,
    cpp.DistanceMatrix64Tensor: distmat64,
}


class DistanceMatrix:
    """Compressed distance matrix (symmetric, zero diagonal, nonnegative).

    Stores only n*(n-1)/2 elements for an n×n distance matrix.
    Supports subscript access with ``matrix[i, j]``.

    Parameters
    ----------
    n_or_data : int | DistanceMatrix | CppDistanceMatrix
        If an int, creates a zero-initialized matrix of that size.
        If a DistanceMatrix or C++ distance matrix, wraps it directly.
    dtype : float32 | float64 | None, optional
        Element precision. ``float32`` stores entries as 32-bit floats,
        ``float64`` as 64-bit floats. Defaults to ``float64`` when
        ``n_or_data`` is an int. Ignored otherwise.
    """

    def __init__(
        self,
        n_or_data: int | DistanceMatrix | CppDistanceMatrix,
        dtype: float32 | float64 | None = None,
    ):
        if isinstance(n_or_data, DistanceMatrix):
            self._data = n_or_data._data
        elif isinstance(n_or_data, _cpp_types):
            self._data = n_or_data
        elif isinstance(n_or_data, int):
            if dtype is None:
                dtype = float64
            if dtype not in _dtype_to_cpp:
                raise TypeError(f"Unsupported dtype {dtype}; use float32 or float64")
            self._data = _dtype_to_cpp[dtype](n_or_data)
        else:
            raise TypeError(f"Expected int, DistanceMatrix, or C++ DistanceMatrix; got {type(n_or_data)}")

    @property
    def dtype(self):
        """Element precision (``float32`` or ``float64``)."""
        return _CPP_TO_DTYPE[type(self._data)]

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def storage_count(self) -> int:
        return self._data.storage_count

    def _resolve_ij(self, ij):
        i, j = ij
        n = self._data.size
        if i < 0:
            i += n
        if j < 0:
            j += n
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(
                f"index ({ij[0]}, {ij[1]}) is out of bounds for a {n}x{n} matrix")
        return i, j

    def __getitem__(self, ij):
        i, j = self._resolve_ij(ij)
        return self._data[i, j]

    def __setitem__(self, ij, value):
        i, j = self._resolve_ij(ij)
        self._data[i, j] = value

    def to_dense(self) -> np.ndarray:
        """Return the full n×n distance matrix as a numpy array."""
        return self._data.to_dense()

    def __reduce__(self):
        import io as _io
        from .io import _save_object, _unpickle_object
        buf = _io.BytesIO()
        _save_object(self, buf)
        return _unpickle_object, (buf.getvalue(),)

    @classmethod
    def from_dense(cls, array):
        """Create a DistanceMatrix from a dense n×n numpy array."""
        if array.dtype == np.float32:
            return cls(cpp.DistanceMatrix_f32.from_dense(array))
        elif array.dtype == np.float64:
            return cls(cpp.DistanceMatrix_f64.from_dense(array))
        else:
            raise TypeError(f"Unsupported dtype {array.dtype}")

    def __repr__(self):
        return repr(self._data)


class DistanceMatrixTensor(Tensor):
    def __init__(self, data: cpp.DistanceMatrix32Tensor | cpp.DistanceMatrix64Tensor):
        super().__init__()
        if isinstance(data, DistanceMatrixTensor):
            data = data._data
        elif not isinstance(data, (cpp.DistanceMatrix32Tensor, cpp.DistanceMatrix64Tensor)):
            raise TypeError(f"Cannot create DistanceMatrixTensor from {type(data)}")
        self._data = data
        self.dtype = _DISTMAT_CPP_TO_DTYPE[type(self._data)]

    def _to_py_tensor(self, data):
        return DistanceMatrixTensor(data)

    def _decay_value(self, val):
        return val._data

    def _represent_element(self, element):
        return DistanceMatrix(element)

    def _get_valid_setitem_dtypes(self):
        return [DistanceMatrix, DistanceMatrixTensor]
