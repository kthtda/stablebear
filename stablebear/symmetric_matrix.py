from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _sb_cpp as cpp
from ._tensor_base import Tensor
from .typing import float32, float64, symmat32, symmat64

if TYPE_CHECKING:
    CppSymmetricMatrix = cpp.SymmetricMatrix_f32 | cpp.SymmetricMatrix_f64

_dtype_to_cpp = {
    float32: cpp.SymmetricMatrix_f32,
    float64: cpp.SymmetricMatrix_f64,
}

_cpp_types = (cpp.SymmetricMatrix_f32, cpp.SymmetricMatrix_f64)

_CPP_TO_DTYPE = {
    cpp.SymmetricMatrix_f32: float32,
    cpp.SymmetricMatrix_f64: float64,
}

_SYMMAT_CPP_TO_DTYPE = {
    cpp.SymmetricMatrix32Tensor: symmat32,
    cpp.SymmetricMatrix64Tensor: symmat64,
}


class SymmetricMatrix:
    """Compressed symmetric matrix using lower-triangular storage.

    Stores only n*(n+1)/2 elements for an n×n symmetric matrix.
    Supports subscript access with ``matrix[i, j]``.

    Parameters
    ----------
    n_or_data : int | SymmetricMatrix | CppSymmetricMatrix
        If an int, creates a zero-initialized matrix of that size.
        If a SymmetricMatrix or C++ symmetric matrix, wraps it directly.
    dtype : float32 | float64 | None, optional
        Element precision. ``float32`` stores entries as 32-bit floats,
        ``float64`` as 64-bit floats. Defaults to ``float64`` when
        ``n_or_data`` is an int. Ignored otherwise.
    """

    def __init__(
        self,
        n_or_data: int | SymmetricMatrix | CppSymmetricMatrix,
        dtype: float32 | float64 | None = None,
    ):
        if isinstance(n_or_data, SymmetricMatrix):
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
            raise TypeError(f"Expected int, SymmetricMatrix, or C++ SymmetricMatrix; got {type(n_or_data)}")

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
        """Return the entry at ``(i, j)``.

        Access is symmetric (``m[i, j] == m[j, i]``). Negative indices count
        from the end, as in NumPy.

        Parameters
        ----------
        ij : tuple of int
            A ``(row, column)`` pair.

        Returns
        -------
        float
            The stored value at ``(i, j)``.

        Raises
        ------
        IndexError
            If ``i`` or ``j`` is out of range for the matrix size.
        """
        i, j = self._resolve_ij(ij)
        return self._data[i, j]

    def __setitem__(self, ij, value):
        """Set the entry at ``(i, j)`` (and, symmetrically, ``(j, i)``).

        Negative indices count from the end, as in NumPy.

        Parameters
        ----------
        ij : tuple of int
            A ``(row, column)`` pair.
        value : float
            The value to store.

        Raises
        ------
        IndexError
            If ``i`` or ``j`` is out of range for the matrix size.
        """
        i, j = self._resolve_ij(ij)
        self._data[i, j] = value

    def to_dense(self) -> np.ndarray:
        """Return the full n×n symmetric matrix as a numpy array."""
        return self._data.to_dense()

    def to_numpy(self) -> np.ndarray:
        """Return the full n×n symmetric matrix as a numpy array.

        Alias for :meth:`to_dense`, provided for naming consistency with the
        rest of the library.
        """
        return self.to_dense()

    def __array__(self, dtype=None, copy=None):
        """Return the dense n×n matrix so ``np.asarray(matrix)`` works.

        Without this, ``np.asarray``/``np.array`` would silently wrap the
        object in a 0-d ``object`` array instead of materializing the matrix
        (see issue #75).
        """
        arr = self.to_dense()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __reduce__(self):
        import io as _io
        from .io import _save_object, _unpickle_object
        buf = _io.BytesIO()
        _save_object(self, buf)
        return _unpickle_object, (buf.getvalue(),)

    @classmethod
    def from_dense(cls, array):
        """Create a SymmetricMatrix from a dense n×n numpy array."""
        if array.dtype == np.float32:
            return cls(cpp.SymmetricMatrix_f32.from_dense(array))
        elif array.dtype == np.float64:
            return cls(cpp.SymmetricMatrix_f64.from_dense(array))
        else:
            raise TypeError(f"Unsupported dtype {array.dtype}")

    def __repr__(self):
        return repr(self._data)


class SymmetricMatrixTensor(Tensor):
    """Tensor whose elements are :class:`SymmetricMatrix` objects.

    Parameters
    ----------
    data : ndarray, SymmetricMatrixTensor, or C++ tensor
        An ndarray of shape ``(*tensor_shape, n, n)`` whose trailing two axes
        form each n×n symmetric matrix, or an existing tensor.
    dtype : symmat32 | symmat64 | None, optional
        Element precision. Inferred from the array dtype when ``None``.
    """

    def __init__(self, data, dtype=None):
        super().__init__()
        if isinstance(data, SymmetricMatrixTensor):
            data = data._data
        elif isinstance(data, np.ndarray):
            from .distance_matrix import _matrix_tensor_cpp_from_array
            data = _matrix_tensor_cpp_from_array(
                data, dtype, SymmetricMatrix, symmat32, symmat64)
        elif not isinstance(data, (cpp.SymmetricMatrix32Tensor, cpp.SymmetricMatrix64Tensor)):
            raise TypeError(f"Cannot create SymmetricMatrixTensor from {type(data)}")
        self._data = data
        self.dtype = _SYMMAT_CPP_TO_DTYPE[type(self._data)]

    @classmethod
    def from_numpy(cls, array, dtype=None):
        """Build a tensor of symmetric matrices from an ``(*tensor_shape, n, n)`` array.

        The trailing two axes of *array* form each n×n symmetric matrix; the
        leading axes form the tensor shape.
        """
        return cls(np.asarray(array), dtype=dtype)

    def _to_py_tensor(self, data):
        return SymmetricMatrixTensor(data)

    def _decay_value(self, val):
        return val._data

    def _represent_element(self, element):
        return SymmetricMatrix(element)

    def _get_valid_setitem_dtypes(self):
        return [SymmetricMatrix, SymmetricMatrixTensor]

    def _validate_setitem_dtype(self, val):
        super()._validate_setitem_dtype(val)
        from .distance_matrix import _check_matrix_tensor_setitem
        _check_matrix_tensor_setitem(
            self, val, SymmetricMatrixTensor, SymmetricMatrix,
            {symmat32: float32, symmat64: float64})
