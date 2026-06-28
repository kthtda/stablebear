from __future__ import annotations

import numpy as np

from .. import _sb_cpp as cpp
from ..typing import _assert_valid_dtype, _SB_TO_NP, _NP_TO_SB, float32, float64, int32, int64, pcf32, pcf32i, pcf64, pcf64i


class Pcf:
    r"""A piecewise constant function (PCF).

    A PCF is defined by a sequence of (time, value) pairs
    :math:`(t_0, v_0), (t_1, v_1), \ldots, (t_{n-1}, v_{n-1})` with
    :math:`t_0 = 0` and :math:`t_0 < t_1 < \cdots < t_{n-1}`. The function
    takes the value :math:`v_i` on the interval :math:`[t_i, t_{i+1})` for
    :math:`0 \leq i < n-1`, and :math:`v_{n-1}` on :math:`[t_{n-1}, \infty)`.

    The breakpoints must be given in non-decreasing time order; a ``ValueError``
    is raised otherwise. The first time must be :math:`t_0 = 0` (PCFs are
    defined on :math:`[0, \infty)`): a ``ValueError`` is raised if the first
    time is not zero — in particular if any time is negative.

    Parameters
    ----------
    arr : numpy.ndarray or Pcf or list
        Input data. If an ndarray or list, should have shape (n, 2) where each
        row is a (time, value) pair. Can also be an existing ``Pcf`` to copy.
    dtype : type, optional
        Data type for the PCF (``pcf32``, ``pcf64``, ``pcf32i``, or ``pcf64i``).
        If ``None``, the dtype is inferred from the input array (e.g. a
        ``numpy.float32`` array produces a 32-bit PCF, a ``numpy.int32`` array
        produces a 32-bit integer PCF).

    Examples
    --------
    >>> import numpy as np
    >>> import stablebear as sb
    >>> f = sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.0]], dtype=np.float32))
    >>> f.size
    3
    """

    _DTYPE_TO_NP = {
        pcf32: np.float32,
        pcf64: np.float64,
        pcf32i: np.int32,
        pcf64i: np.int64,
    }

    _NP_TO_CPP = {
        np.float32: lambda arr: cpp.Pcf_f32_f32(arr),
        np.float64: lambda arr: cpp.Pcf_f64_f64(arr),
        np.int32: lambda arr: cpp.Pcf_i32_i32(arr),
        np.int64: lambda arr: cpp.Pcf_i64_i64(arr),
    }

    _CPP_TYPE_INFO = {
        cpp.Pcf_f32_f32: (float32, float32),
        cpp.Pcf_f64_f64: (float64, float64),
        cpp.Pcf_i32_i32: (int32, int32),
        cpp.Pcf_i64_i64: (int64, int64),
    }

    def __init__(self, arr: np.ndarray | Pcf | list[list[float | int] | tuple[float | int, ...]], dtype=None):
        if isinstance(arr, Pcf):
            self._data = arr._data
            self.ttype = arr.ttype
            self.vtype = arr.vtype

        elif isinstance(arr, np.ndarray):
            if dtype is not None:
                np_dtype = self._DTYPE_TO_NP.get(dtype, dtype)
                if arr.dtype != np_dtype:
                    arr = arr.astype(np_dtype)

            constructor = self._NP_TO_CPP.get(arr.dtype.type)
            if constructor is None:
                raise ValueError(
                    "Unsupported array type (must be np.float32/64 or np.int32/64)"
                )
            self._data = constructor(arr)
            self.ttype = _NP_TO_SB[arr.dtype.type]
            self.vtype = _NP_TO_SB[arr.dtype.type]

        elif isinstance(arr, tuple(self._CPP_TYPE_INFO.keys())):
            self._data = arr
            self.ttype, self.vtype = self._CPP_TYPE_INFO[type(arr)]

        elif isinstance(arr, list):
            if dtype is None:
                dtype = pcf64
            np_dtype = self._DTYPE_TO_NP.get(dtype, dtype)
            data = np.array(arr, dtype=np_dtype)
            constructor = self._NP_TO_CPP.get(np_dtype)
            if constructor is None:
                raise ValueError("Unsupported dtype")
            self._data = constructor(data)
            self.ttype = _NP_TO_SB[np_dtype]
            self.vtype = _NP_TO_SB[np_dtype]

        else:
            raise ValueError(
                f"Tried to create PCF from unsupported input data of type {type(arr)}."
            )

    def _get_time_type(self):
        return self._data.get_time_type()

    def _get_value_type(self):
        return self._data.get_value_type()

    def _get_time_value_type(self):
        return self._get_time_type() + "_" + self._get_value_type()

    def to_numpy(self):
        """Convert the PCF to a numpy array of shape (n, 2) with (time, value) rows."""
        return np.asarray(self._data)

    def _debug_print(self):
        self._data.debug_print()

    def astype(self, dtype):
        """Return a copy of the PCF cast to the given dtype (``pcf32``, ``pcf64``, ``pcf32i``, or ``pcf64i``)."""
        _assert_valid_dtype(
            dtype,
            [pcf32, pcf64, pcf32i, pcf64i, np.float32, np.float64, np.int32, np.int64],
        )
        np_dtype = self._DTYPE_TO_NP.get(dtype, dtype)
        return Pcf(self.to_numpy().astype(np_dtype))

    def _binop(self, rhs, op):
        if isinstance(rhs, Pcf):
            if not _has_matching_types(self, rhs):
                raise TypeError("Mismatched PCF types")
            return Pcf(op(self._data, rhs._data))
        return Pcf(op(self._data, rhs))

    def __add__(self, rhs):
        return self._binop(rhs, type(self._data).__add__)

    def __sub__(self, rhs):
        return self._binop(rhs, type(self._data).__sub__)

    def __mul__(self, rhs):
        return self._binop(rhs, type(self._data).__mul__)

    def __radd__(self, lhs):
        return Pcf(self._data.__radd__(lhs))

    def __rsub__(self, lhs):
        return Pcf(self._data.__rsub__(lhs))

    def __rmul__(self, lhs):
        return Pcf(self._data.__rmul__(lhs))

    def __truediv__(self, rhs):
        return self._binop(rhs, type(self._data).__truediv__)

    def __rtruediv__(self, lhs):
        return Pcf(self._data.__rtruediv__(lhs))

    def __neg__(self):
        return Pcf(-self._data)

    def __pow__(self, exponent):
        """Raise every value of the PCF to a power.

        Returns a new PCF whose value at each breakpoint is raised to
        ``exponent``. The domain (time coordinates) is unchanged.

        A ``RuntimeWarning`` is emitted if the result contains NaN or
        infinity (e.g. negative base with fractional exponent).

        Parameters
        ----------
        exponent : float or int
            The exponent.

        Returns
        -------
        Pcf
            A new PCF with transformed values.

        Examples
        --------
        >>> import numpy as np
        >>> import stablebear as sb
        >>> f = sb.Pcf(np.array([[0.0, 4.0], [1.0, 9.0]], dtype=np.float64))
        >>> g = f ** 0.5
        >>> g(0.5)
        2.0
        """
        return Pcf(self._data.__pow__(exponent))

    def __ipow__(self, exponent):
        """Raise every value of the PCF to a power in place.

        Parameters
        ----------
        exponent : float or int
            The exponent.

        Returns
        -------
        self
        """
        self._data = self._data.__pow__(exponent)
        return self

    def __call__(self, t):
        """Evaluate the PCF at the given time(s).

        Parameters
        ----------
        t : float, int, list, numpy.ndarray, or FloatTensor
            The time(s) at which to evaluate the PCF. Can be a single scalar
            or an array-like of times with arbitrary shape.

        Returns
        -------
        float, numpy.ndarray, or FloatTensor
            The PCF value(s) at the given time(s). A scalar input returns a
            Python float. A list or ndarray input returns an ndarray of the
            same shape. A ``FloatTensor`` input returns a tensor of the same
            type and shape.

        Raises
        ------
        ValueError
            If any time is negative (PCFs are defined on :math:`[0, \\infty)`).

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
        from ..base_tensor import FloatTensor

        if isinstance(t, int | float):
            return self._data(t)

        return_tensor = None
        if isinstance(t, FloatTensor):
            return_tensor = type(t)
            t = np.asarray(t)
        elif isinstance(t, list):
            t = np.asarray(t, dtype=_SB_TO_NP[self.ttype])

        result = self._data(t)

        if return_tensor is not None:
            return return_tensor(result)
        return result

    @property
    def size(self):
        """Number of breakpoints (time-value pairs) in this PCF."""
        return self._data.size()

    _VTYPE_NAMES = {
        float32: "float32",
        float64: "float64",
        int32: "int32",
        int64: "int64",
    }

    def __str__(self):
        dtname = self._VTYPE_NAMES.get(self.vtype, str(self.vtype))
        return f"<PCF size={self._data.size()}, dtype={dtname}>"

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self._data)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __reduce__(self):
        import io as _io
        from ..io import _save_object, _load_object
        buf = _io.BytesIO()
        _save_object(self, buf)
        from ..io import _unpickle_object
        return _unpickle_object, (buf.getvalue(),)

    def __eq__(self, other):
        return np.array_equal(self.__array__(), np.asarray(other))


_BACKEND_MAP = {
    cpp.Pcf_f32_f32: cpp.Backend_f32_f32,
    cpp.Pcf_f64_f64: cpp.Backend_f64_f64,
    cpp.Pcf_i32_i32: cpp.Backend_i32_i32,
    cpp.Pcf_i64_i64: cpp.Backend_i64_i64,
}


def _has_matching_types(f: Pcf, g: Pcf):
    return type(f._data) is type(g._data)


class Rectangle:
    """A rectangle produced by iterating over a pair of PCFs."""

    __slots__ = ("_data",)

    def __init__(self, data):
        self._data = data

    @property
    def left(self):
        """Left time boundary."""
        return self._data.left

    @property
    def right(self):
        """Right time boundary."""
        return self._data.right

    @property
    def f_value(self):
        """Value of the first PCF on this interval."""
        return self._data.f_value

    @property
    def g_value(self):
        """Value of the second PCF on this interval."""
        return self._data.g_value

    def __eq__(self, other):
        if isinstance(other, Rectangle):
            return (self.left == other.left and self.right == other.right
                    and self.f_value == other.f_value and self.g_value == other.g_value)
        if isinstance(other, tuple) and len(other) == 4:
            return (self.left, self.right, self.f_value, self.g_value) == other
        return NotImplemented

    def __repr__(self):
        return (f"Rectangle(left={self.left}, right={self.right}, "
                f"f_value={self.f_value}, g_value={self.g_value})")


def _coerce_int_bound(x, np_t):
    """Coerce an integration bound for an integer-time PCF.

    A ``float`` infinity (the "no bound" sentinel) maps to the dtype's max/min;
    any other value is truncated to a Python ``int``.
    """
    if isinstance(x, float) and np.isinf(x):
        info = np.iinfo(np_t)
        return int(info.max if x > 0 else info.min)
    return int(x)


def iterate_rectangles(f: Pcf, g: Pcf, a=None, b=None):
    """Iterate over the rectangles formed by two PCFs.

    Parameters
    ----------
    f, g : Pcf
        The two piecewise constant functions (which must share a dtype).
    a : int or float, optional
        Left integration bound. Defaults to 0 in the PCF's time dtype.
    b : int or float, optional
        Right integration bound. Defaults to "no upper bound": ``inf`` for a
        float-typed PCF and the integer dtype's maximum for an integer-typed
        PCF. For an integer-typed PCF an explicit ``inf`` bound is mapped to the
        dtype's maximum and any other value is truncated to ``int``.

    Returns
    -------
    list[Rectangle]
        Rectangles in chronological order.
    """
    if not isinstance(f, Pcf) or not isinstance(g, Pcf):
        raise TypeError("Both f and g must be Pcf objects.")
    if not _has_matching_types(f, g):
        raise TypeError("f and g must have the same dtype.")
    backend = _BACKEND_MAP.get(type(f._data))
    if backend is None:
        raise TypeError(
            "iterate_rectangles is not supported for this PCF type."
        )

    # The C++ overload is typed on the PCF's time type, so integer PCFs need
    # integer bounds; the float defaults (0.0 / inf) would not bind (issue #13).
    np_t = _SB_TO_NP[f.ttype]
    if np.issubdtype(np_t, np.integer):
        a = 0 if a is None else _coerce_int_bound(a, np_t)
        b = int(np.iinfo(np_t).max) if b is None else _coerce_int_bound(b, np_t)
    else:
        a = 0.0 if a is None else float(a)
        b = float('inf') if b is None else float(b)

    cpp_rects = backend.iterate_rectangles(f._data, g._data, a, b)
    return [Rectangle(r) for r in cpp_rects]
