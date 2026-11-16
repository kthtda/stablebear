from __future__ import annotations

from typing import Any

import numpy as np

from .. import _autodiff
from .._tensor_base import _array_for_protocol


class _PiecewiseFunction:
    """A piecewise function on [0, inf), the base of ``Pcf`` and ``Plf``."""

    # A native piecewise function with an (n, 2) array view of its
    # breakpoints and __call__ overloads for a scalar and an array.
    _data: Any

    def __call__(self, t):
        """Evaluate the function at the given time(s).

        Parameters
        ----------
        t : scalar, list, numpy.ndarray, NumericTensor, or torch.Tensor
            Any time or times the function accepts, as a single scalar or an
            array-like of any shape.

        Returns
        -------
        scalar, numpy.ndarray, NumericTensor, or torch.Tensor
            The value(s) at ``t``. Their kind follows ``t``: a scalar gives a
            scalar, a list or ndarray gives an ndarray of the same shape, and
            a ``NumericTensor`` gives a tensor of the same shape. Their type
            follows the function. If ``t`` is a PyTorch tensor, or the
            function carries gradients, the values are a PyTorch tensor of
            the shape of ``t`` that carries gradients.

        Raises
        ------
        ValueError
            If any time is negative (the function is defined on
            :math:`[0, \\infty)`).

        Examples
        --------
        >>> import numpy as np
        >>> import stablebear as sb
        >>> f = sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.5]], dtype=np.float32))
        >>> f(0.5)
        1.0
        >>> f(np.array([0.5, 1.5, 4.0]))
        array([1. , 2. , 0.5], dtype=float32)
        """
        from ..base_tensor import NumericTensor
        from ..tensor_create import tensor

        if _autodiff._tracked(self) is not None or _autodiff._framework(t) is not None:
            # With gradients: the values as a PyTorch tensor.
            return self._framework_values(np.asarray(t) if isinstance(t, NumericTensor) else t)

        if isinstance(t, int | float):
            return self._data(t)

        if isinstance(t, NumericTensor | list):
            # Times and values share a type, so the breakpoints give the time type.
            result = self._data(np.asarray(t, dtype=np.asarray(self._data).dtype))
            return tensor(result) if isinstance(t, NumericTensor) else result

        return self._data(t)

    def _framework_values(self, t):
        """The values at ``t`` as a framework array that carries gradients."""
        raise NotImplementedError

    def __array__(self, dtype=None, copy=None):
        return _array_for_protocol(np.asarray(self._data), dtype, copy, copied=False)

    def __eq__(self, other):
        return np.array_equal(self.__array__(), np.asarray(other))
