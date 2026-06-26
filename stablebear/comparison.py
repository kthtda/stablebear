from __future__ import annotations

import numpy as np

from .base_tensor import BoolTensor, FloatTensor, NumericTensor
from .distance_matrix import DistanceMatrix
from .symmetric_matrix import SymmetricMatrix
from .typing import float64

_MATRIX_TYPES = (DistanceMatrix, SymmetricMatrix)


def _to_float64_tensor(x) -> FloatTensor:
    """Coerce a numeric/boolean tensor or array-like to a float64 FloatTensor."""
    if isinstance(x, FloatTensor) and x.dtype is float64:
        return x
    return FloatTensor(np.asarray(x, dtype=np.float64))


def allclose(a, b, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    r"""Test whether two objects are element-wise equal within a tolerance.

    Returns ``True`` when, for every pair of corresponding elements
    :math:`a_i` and :math:`b_i`,

    .. math::

        |a_i - b_i| \leq \texttt{atol} + \texttt{rtol} \cdot |b_i|.

    Infinities are handled as in NumPy: equal (same-sign) infinities are close,
    while a finite-vs-infinite or opposite-sign pair is not, and ``NaN`` is
    never close.

    Parameters
    ----------
    a, b : FloatTensor | IntTensor | BoolTensor | ndarray | DistanceMatrix | SymmetricMatrix
        The two objects to compare. Numeric and boolean tensors (and NumPy
        arrays) are compared as ``float64`` and may differ in dtype; their
        shapes are broadcast together (as in NumPy). A ``DistanceMatrix`` /
        ``SymmetricMatrix`` must be compared against the same matrix type.
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
        If neither operand is a stablebear tensor or matrix, or a matrix is
        compared against a different type.
    ValueError
        If the operands' shapes are not broadcast-compatible.
    """
    # Compressed matrices keep their own compact comparison.
    if isinstance(a, _MATRIX_TYPES) or isinstance(b, _MATRIX_TYPES):
        if type(a) is not type(b):
            raise TypeError(
                f"allclose on a {type(a).__name__} requires the same matrix "
                f"type, got {type(b).__name__}"
            )
        if a.dtype is b.dtype:
            return a._data.allclose(b._data, atol=float(atol), rtol=float(rtol))
        # Differing precision has no shared C++ overload; compare dense forms.
        return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol))

    # Numeric / boolean tensors (and array-likes paired with one). Require at
    # least one genuine stablebear tensor so allclose(42, 42) still raises.
    if not (isinstance(a, (NumericTensor, BoolTensor))
            or isinstance(b, (NumericTensor, BoolTensor))):
        raise TypeError(
            f"allclose requires at least one stablebear tensor or matrix operand, "
            f"got {type(a).__name__} and {type(b).__name__}"
        )

    fa = _to_float64_tensor(a)
    fb = _to_float64_tensor(b)
    return fa._data.allclose(fb._data, atol=float(atol), rtol=float(rtol))
