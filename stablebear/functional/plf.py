from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import _sb_cpp as cpp
from .._tensor_base import _array_for_protocol
from ..typing import Dtype, _assert_valid_dtype, float32, float64


class Plf:
    r"""A piecewise linear function (PLF).

    A PLF is defined by breakpoints
    :math:`(t_0, v_0), (t_1, v_1), \ldots, (t_n, v_n)` with :math:`t_0 = 0`
    and :math:`t_0 < t_1 < \cdots < t_n`. On :math:`[t_i, t_{i+1}]` it
    interpolates linearly between :math:`v_i` and :math:`v_{i+1}`. A single
    breakpoint :math:`(0, v_0)` gives the constant function :math:`v_0`.

    .. important::

       After its last breakpoint, a PLF continues its last segment, with
       slope :math:`(v_n - v_{n-1}) / (t_n - t_{n-1})`. It does not stay at
       :math:`v_n`, as a :class:`Pcf` does. To keep a PLF constant after its
       data ends, end it with a flat segment by repeating the last value at
       a later time.

    Parameters
    ----------
    arr : numpy.ndarray, list, or Plf
        The breakpoints, as an ``(n, 2)`` array or a list of
        ``(time, value)`` pairs, or a ``Plf`` to copy.
    dtype : dtype, optional
        ``float32`` or ``float64``, the type of the times and values.
        Without it, a ``numpy.float32`` array gives ``float32``, any other
        real array gives ``float64``, a list gives ``float64``, and a
        ``Plf`` keeps its dtype.

    Raises
    ------
    ValueError
        If there are no breakpoints, the first time is not 0, the times are
        not strictly increasing, or a time is not finite.
    TypeError
        If ``dtype`` is not ``float32`` or ``float64``, or an array does not
        hold real numbers.

    Examples
    --------
    >>> import stablebear as sb
    >>> f = sb.Plf([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    >>> f.size
    3
    """

    _DTYPE_TO_NP = {float32: np.float32, float64: np.float64}

    _DTYPE_TO_CPP = {float32: cpp.Plf_f32_f32, float64: cpp.Plf_f64_f64}

    # A native Plf_f32_f32 or Plf_f64_f64.
    _data: Any

    def __init__(
        self,
        arr: np.ndarray | Plf | Sequence[Sequence[float]],
        dtype: Dtype | None = None,
    ):
        if dtype is not None:
            _assert_valid_dtype(dtype, [float32, float64])

        if isinstance(arr, Plf):
            if dtype is None:
                dtype = arr.dtype
            arr = arr.to_numpy()

        if isinstance(arr, np.ndarray):
            if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
                raise TypeError(f"PLF breakpoints must be real numbers, not {arr.dtype}.")
            if dtype is None:
                dtype = float32 if arr.dtype == np.float32 else float64
        elif dtype is None:
            dtype = float64

        data = np.asarray(arr, dtype=self._DTYPE_TO_NP[dtype])
        self._data = self._DTYPE_TO_CPP[dtype](data)
        self._dtype = dtype

    @property
    def dtype(self) -> Dtype:
        """The type of the times and values, ``float32`` or ``float64``."""
        return self._dtype

    @property
    def size(self) -> int:
        """Number of breakpoints (time-value pairs) in this PLF."""
        return self._data.size()

    def to_numpy(self) -> np.ndarray:
        """Convert the PLF to a numpy array of shape (n, 2) with (time, value) rows."""
        return np.asarray(self._data)

    def astype(self, dtype: Dtype) -> Plf:
        """Return a copy of the PLF cast to the given dtype (``float32`` or ``float64``)."""
        _assert_valid_dtype(dtype, [float32, float64])
        return Plf(self.to_numpy(), dtype=dtype)

    def __str__(self):
        return f"<PLF size={self.size}, dtype={self._dtype}>"

    def __array__(self, dtype=None, copy=None):
        return _array_for_protocol(np.asarray(self._data), dtype, copy, copied=False)

    def __eq__(self, other):
        return np.array_equal(self.__array__(), np.asarray(other))
