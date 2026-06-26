from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _sb_cpp as cpp
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
        """Return the entry at ``(i, j)``.

        Access is symmetric (``m[i, j] == m[j, i]``) and the diagonal is always
        zero. Negative indices count from the end, as in NumPy.

        Parameters
        ----------
        ij : tuple of int
            A ``(row, column)`` pair.

        Returns
        -------
        float
            The distance between items ``i`` and ``j``.

        Raises
        ------
        IndexError
            If ``i`` or ``j`` is out of range for the matrix size.
        """
        i, j = self._resolve_ij(ij)
        return self._data[i, j]

    def __setitem__(self, ij, value):
        """Set the entry at ``(i, j)`` (and, symmetrically, ``(j, i)``).

        Negative indices count from the end. Writes to the diagonal are
        rejected unless the value is zero, and entries must be nonnegative.

        Parameters
        ----------
        ij : tuple of int
            A ``(row, column)`` pair.
        value : float
            The distance to store.

        Raises
        ------
        IndexError
            If ``i`` or ``j`` is out of range for the matrix size.
        """
        i, j = self._resolve_ij(ij)
        self._data[i, j] = value

    def to_dense(self) -> np.ndarray:
        """Return the full n×n distance matrix as a numpy array."""
        return self._data.to_dense()

    def to_numpy(self) -> np.ndarray:
        """Return the full n×n distance matrix as a numpy array.

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
        """Create a DistanceMatrix from a dense n×n numpy array."""
        if array.dtype == np.float32:
            return cls(cpp.DistanceMatrix_f32.from_dense(array))
        elif array.dtype == np.float64:
            return cls(cpp.DistanceMatrix_f64.from_dense(array))
        else:
            raise TypeError(f"Unsupported dtype {array.dtype}")

    def __repr__(self):
        return repr(self._data)


def _matrix_dtype_for(arr_dtype, dtype, dtype32, dtype64, name):
    if dtype is not None:
        if dtype not in (dtype32, dtype64):
            raise TypeError(
                f"dtype must be {dtype32.__name__} or {dtype64.__name__}, "
                f"got {getattr(dtype, '__name__', dtype)}")
        return dtype
    return dtype32 if np.dtype(arr_dtype) == np.float32 else dtype64


def _matrix_tensor_to_dense(tensor) -> np.ndarray:
    """Densify a distance/symmetric matrix tensor to ``(*tensor_shape, n, n)``.

    Each element is materialized via its ``.to_dense()``; all matrices must
    share the same size ``n`` or a ``ValueError`` is raised. An empty tensor
    (a zero-length axis) has no matrix to infer ``n`` from and also raises
    ``ValueError``.
    """
    shape = tuple(tensor.shape)
    mats = [
        tensor._represent_element(tensor._data._get_element(list(idx))).to_dense()
        for idx in np.ndindex(*shape)
    ]
    if not mats:
        raise ValueError(
            f"cannot densify a matrix tensor with an empty axis (shape {shape}): "
            "there is no matrix to infer the size from")
    n = mats[0].shape[0]
    if any(m.shape != (n, n) for m in mats):
        raise ValueError(
            "cannot densify a matrix tensor whose matrices have differing sizes; "
            "index the tensor and convert each matrix separately instead")
    return np.stack(mats).reshape(shape + (n, n))


def _matrix_tensor_cpp_from_array(arr, dtype, scalar_cls, dtype32, dtype64):
    """Build a C++ matrix tensor from an ``(*tensor_shape, n, n)`` ndarray.

    The trailing two axes of *arr* form each n×n matrix; the leading axes
    form the tensor shape.
    """
    from .tensor_create import zeros
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError(
            "array must have at least 2 dimensions (the trailing two form each matrix)")
    if arr.shape[-1] != arr.shape[-2]:
        raise ValueError(
            f"the trailing two axes must be square, got {tuple(arr.shape[-2:])}")
    dt = _matrix_dtype_for(arr.dtype, dtype, dtype32, dtype64, scalar_cls.__name__)
    np_float = np.float32 if dt == dtype32 else np.float64
    arr = np.ascontiguousarray(arr, dtype=np_float)
    tensor_shape = arr.shape[:-2]
    t = zeros(tensor_shape, dtype=dt)
    for idx in np.ndindex(*tensor_shape):
        t[idx] = scalar_cls.from_dense(arr[idx])
    return t._data


class DistanceMatrixTensor(Tensor):
    """Tensor whose elements are :class:`DistanceMatrix` objects.

    Parameters
    ----------
    data : ndarray, DistanceMatrixTensor, or C++ tensor
        An ndarray of shape ``(*tensor_shape, n, n)`` whose trailing two axes
        form each n×n distance matrix, or an existing tensor.
    dtype : distmat32 | distmat64 | None, optional
        Element precision. Inferred from the array dtype when ``None``.
    """

    def __init__(self, data, dtype=None):
        super().__init__()
        if isinstance(data, DistanceMatrixTensor):
            data = data._data
        elif isinstance(data, np.ndarray):
            data = _matrix_tensor_cpp_from_array(
                data, dtype, DistanceMatrix, distmat32, distmat64)
        elif not isinstance(data, (cpp.DistanceMatrix32Tensor, cpp.DistanceMatrix64Tensor)):
            raise TypeError(f"Cannot create DistanceMatrixTensor from {type(data)}")
        self._data = data
        self.dtype = _DISTMAT_CPP_TO_DTYPE[type(self._data)]

    @classmethod
    def from_numpy(cls, array, dtype=None):
        """Build a tensor of distance matrices from an ``(*tensor_shape, n, n)`` array.

        The trailing two axes of *array* form each n×n distance matrix (which
        must be symmetric with a zero diagonal and nonnegative entries); the
        leading axes form the tensor shape.
        """
        return cls(np.asarray(array), dtype=dtype)

    @classmethod
    def from_dense(cls, array, dtype=None):
        """Build a tensor of distance matrices from a dense array.

        Alias for :meth:`from_numpy`, completing the ``from_dense``/``to_dense``
        pair.
        """
        return cls.from_numpy(array, dtype=dtype)

    def to_dense(self) -> np.ndarray:
        """Return the matrices as a dense ``(*tensor_shape, n, n)`` array.

        All matrices must share the same size ``n``; otherwise (or for an
        empty tensor) a ``ValueError`` is raised.
        """
        return _matrix_tensor_to_dense(self)

    def to_numpy(self) -> np.ndarray:
        """Return the dense ``(*tensor_shape, n, n)`` array. Alias for :meth:`to_dense`."""
        return self.to_dense()

    def __array__(self, dtype=None, copy=None):
        """Return the dense ``(*tensor_shape, n, n)`` array so ``np.asarray`` works."""
        arr = self.to_dense()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def _to_py_tensor(self, data):
        return DistanceMatrixTensor(data)

    def _decay_value(self, val):
        return val._data

    def _represent_element(self, element):
        return DistanceMatrix(element)

    def _get_valid_setitem_dtypes(self):
        return [DistanceMatrix, DistanceMatrixTensor]
