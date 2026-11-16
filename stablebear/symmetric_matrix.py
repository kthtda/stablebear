from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from . import _autodiff
from . import _sb_cpp as cpp
from ._binary_io import _BinaryIoMixin
from ._tensor_base import Tensor, _array_for_protocol
from ._tracked import Tracking, tracked_class
from .typing import Dtype, float32, float64, symmat32, symmat64

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


class SymmetricMatrix(_BinaryIoMixin):
    """Compressed symmetric matrix using lower-triangular storage.

    Stores only n*(n+1)/2 elements for an n×n symmetric matrix.
    Supports subscript access with ``matrix[i, j]``.

    Parameters
    ----------
    n_or_data : int, numpy.ndarray, or SymmetricMatrix
        If an int, creates a zero-initialized matrix of that size.
        A two-dimensional NumPy array supplies the full square matrix. A
        one-dimensional array supplies the upper triangle, including the
        diagonal, in row-major order: ``(0, 0), (0, 1), ..., (1, 1), ...``.
        If a SymmetricMatrix, wraps it directly.
    dtype : float32 or float64, optional
        Element precision. ``float32`` stores entries as 32-bit floats,
        ``float64`` as 64-bit floats. Defaults to ``float64`` when
        ``n_or_data`` is an int. For array input, ``float32`` and ``float64``
        are inferred; other array dtypes require an explicit dtype. An
        explicit dtype converts the array before construction. Ignored for
        existing matrix objects.
    """

    # A native SymmetricMatrix_f32 or _f64.
    _data: Any

    def _binary_io_data(self):
        return self._data

    def __init__(
        self,
        n_or_data: int | np.ndarray | SymmetricMatrix,
        dtype: Dtype | None = None,
    ):
        n_or_data = _autodiff.dense(n_or_data, "A SymmetricMatrix")
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
        elif isinstance(n_or_data, np.ndarray):
            self._data = _symmetric_matrix_from_array(n_or_data, dtype)
        else:
            raise TypeError(
                "Expected int, numpy.ndarray, or SymmetricMatrix; "
                f"got {type(n_or_data)}")

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
        return _array_for_protocol(self.to_dense(), dtype, copy, copied=True)

    @classmethod
    def from_dense(cls, array):
        """Create a SymmetricMatrix from a dense n×n NumPy array.

        .. deprecated:: 0.4.7
           Pass the array to ``SymmetricMatrix(array)`` instead.
        """
        warnings.warn(
            "SymmetricMatrix.from_dense() is deprecated since 0.4.7; pass the array to "
            "SymmetricMatrix(array) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if array.dtype not in (np.float32, np.float64):
            raise TypeError(f"Unsupported dtype {array.dtype}")
        return cls(array)

    def __repr__(self):
        return repr(self._data)


def _symmetric_matrix_from_array(array, dtype):
    from .distance_matrix import _array_dtype

    dt = _array_dtype(array, dtype)
    np_dtype = np.float32 if dt is float32 else np.float64
    array = np.asarray(array, dtype=np_dtype)
    return _dtype_to_cpp[dt].from_numpy(array)


class SymmetricMatrixTensor(Tensor):
    """Tensor whose elements are :class:`SymmetricMatrix` objects.

    Parameters
    ----------
    data : numpy.ndarray or SymmetricMatrixTensor
        An ndarray of shape ``(*tensor_shape, n, n)`` whose trailing two axes
        form each n×n symmetric matrix, or an existing tensor.
    dtype : symmat32 or symmat64, optional
        Element precision. Inferred from the array dtype when ``None``.
    """

    def __init__(self, data, dtype=None):
        super().__init__()
        data = _autodiff.dense(data, "A SymmetricMatrixTensor")
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

    # The framework entries of a tensor that carries gradients (see
    # _tracked); None for any other.
    _tvalues: Any = None

    def to_serial_content(self) -> tuple[Any, np.ndarray, np.ndarray]:
        """The entries of all matrices as one array, with the rows and the
        size of each matrix.

        If the tensor carries gradients, the content is a PyTorch tensor
        that carries them, and a loss can be built from it.

        Returns
        -------
        content : numpy.ndarray or torch.Tensor
            The entries on and below the diagonal of all matrices, of shape
            ``(m, 1)``, in row-major order of the matrices, and for each
            matrix by rows: ``(0, 0)``, ``(1, 0)``, ``(1, 1)``, and so on.
        enumeration : numpy.ndarray
            The rows ``[start, stop)`` of the entries of each matrix, of
            shape ``self.shape + (2,)``.
        sizes : numpy.ndarray
            The number of rows of each matrix, of shape ``self.shape``.
        """
        return self._serial()

    def _serial(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(content, enumeration, sizes): as the (m, 1) content, the entries
        on and below the diagonal of all matrices in C order, each by rows:
        (0, 0), (1, 0), (1, 1), and so on; for each matrix the [start, stop)
        rows of its entries, with shape self.shape + (2,); and the number of
        rows of each matrix, with shape self.shape."""
        return cpp.symmetric_matrices_to_serial_content(self._data)

    @staticmethod
    def _from_serial(content: np.ndarray, enumeration: np.ndarray, sizes: np.ndarray) -> SymmetricMatrixTensor:
        """The inverse of _serial."""
        from .distance_matrix import _matrix_serial_content

        content, enumeration, sizes = _matrix_serial_content(
            content, enumeration, sizes, lambda n: n * (n + 1) // 2)
        if content.dtype == np.float32:
            return SymmetricMatrixTensor(cpp.symmetric_matrices_from_serial_content_32(content, enumeration, sizes))
        return SymmetricMatrixTensor(cpp.symmetric_matrices_from_serial_content_64(content, enumeration, sizes))

    def _decay_value(self, val):
        return val._data

    def _represent_element(self, element):
        return SymmetricMatrix(element)

    def _get_valid_setitem_dtypes(self):
        return [SymmetricMatrix, SymmetricMatrixTensor]


SymmetricMatrixTensor._tracking = Tracking(  # pyright: ignore[reportAttributeAccessIssue]
    groups=(("_tvalues", 1),),
    from_serial=SymmetricMatrixTensor._from_serial,
    attributes=lambda np_dtype: {"dtype": symmat32 if np_dtype == np.float32 else symmat64},
    element=lambda rows, size: SymmetricMatrixTensor._from_serial(
        rows, np.array([[0, len(rows)]]), np.array([size]))[0],
)

# A SymmetricMatrixTensor whose entries depend on PyTorch
# tensors, built from [entries], the enumeration, and the sizes of
# _serial; see _tracked.
_TrackedSymmetricMatrixTensor = tracked_class(SymmetricMatrixTensor)
