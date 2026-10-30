from __future__ import annotations

import numpy as np

from .base_tensor import BoolTensor, FloatTensor, NumericTensor
from .distance_matrix import DistanceMatrix
from .symmetric_matrix import SymmetricMatrix
from .typing import float64


def _as_float64_tensor(x) -> FloatTensor:
    if isinstance(x, FloatTensor) and x.dtype == float64:
        return x
    return FloatTensor(np.asarray(x, dtype=np.float64))


def allclose(a, b, atol=1e-8, rtol=1e-5) -> bool:
    r"""Test whether two objects are element-wise equal within a tolerance.

    Returns ``True`` when, for every pair of corresponding elements
    :math:`a_i` and :math:`b_i`,

    .. math::

        |a_i - b_i| \leq \texttt{atol} + \texttt{rtol} \cdot |b_i|.

    An infinite element is close only to an equal infinity, and NaN is never
    close to anything.

    Parameters
    ----------
    a, b : FloatTensor, IntTensor, BoolTensor, DistanceMatrix, SymmetricMatrix, or array_like
        The objects to compare. Numeric and boolean tensors are compared as
        ``float64``, so their dtypes may differ, and their shapes are
        broadcast together. Either one may be an array or a list, as long as
        the other is a tensor. Integers above :math:`2^{53}` lose precision
        in the conversion. A ``DistanceMatrix`` or ``SymmetricMatrix`` must
        be compared with a matrix of the same type.
    atol : float, optional
        Absolute tolerance, by default 1e-8.
    rtol : float, optional
        Relative tolerance, by default 1e-5.

    Returns
    -------
    bool

    Raises
    ------
    TypeError
        If neither input is a tensor or matrix, or a matrix is compared with
        an object of a different type.
    ValueError
        If the shapes of two tensors cannot be broadcast together.
    """
    matrix_types = (DistanceMatrix, SymmetricMatrix)
    if isinstance(a, matrix_types) or isinstance(b, matrix_types):
        if type(a) is not type(b):
            matrix_type = type(a if isinstance(a, matrix_types) else b).__name__
            raise TypeError(
                f"allclose compares a {matrix_type} only with another {matrix_type}, "
                f"got {type(a).__name__} and {type(b).__name__}"
            )
        if a.dtype != b.dtype:
            dense_a, dense_b = np.asarray(a), np.asarray(b)
            if dense_a.shape != dense_b.shape:
                return False
            return bool(np.allclose(dense_a, dense_b, atol=atol, rtol=rtol))
        if isinstance(a, DistanceMatrix):
            return a._current_matrix().allclose(b._current_matrix(), atol=atol, rtol=rtol)
        return a._data.allclose(b._data, atol=atol, rtol=rtol)

    tensor_types = (NumericTensor, BoolTensor)
    if not isinstance(a, tensor_types) and not isinstance(b, tensor_types):
        raise TypeError(
            f"allclose requires at least one tensor or matrix, "
            f"got {type(a).__name__} and {type(b).__name__}"
        )
    fa, fb = _as_float64_tensor(a), _as_float64_tensor(b)
    return fa._data.allclose(fb._data, atol=atol, rtol=rtol)
