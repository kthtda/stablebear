from __future__ import annotations

import operator
from typing import Any

import numpy as np

from .. import _autodiff
from .. import _sb_cpp as cpp
from .._binary_io import _BinaryIoMixin
from .._tracked import Tracking, tracked_class
from ._piecewise import _PiecewiseFunction
from .plf import Plf
from ..typing import (_assert_valid_dtype, _NP_TO_SB, _sized_type, float32, float64, int32, int64, pcf32, pcf32i, pcf64,
                      pcf64i)


class Pcf(_PiecewiseFunction, _BinaryIoMixin):
    r"""A piecewise constant function (PCF).

    A PCF is defined by a sequence of (time, value) pairs
    :math:`(t_0, v_0), (t_1, v_1), \ldots, (t_{n-1}, v_{n-1})` with
    :math:`t_0 = 0` and :math:`t_0 < t_1 < \cdots < t_{n-1}`. The function
    takes the value :math:`v_i` on the interval :math:`[t_i, t_{i+1})` for
    :math:`0 \leq i < n-1`, and :math:`v_{n-1}` on :math:`[t_{n-1}, \infty)`.

    The breakpoint times must be strictly increasing; a ``ValueError`` is
    raised otherwise. The first time must be :math:`t_0 = 0` (PCFs are
    defined on :math:`[0, \infty)`): a ``ValueError`` is raised if the first
    time is not zero — in particular if any time is negative.

    Parameters
    ----------
    arr : numpy.ndarray or Pcf or list
        Input data. If an ndarray or list, should have shape (n, 2) where each
        row is a (time, value) pair. Can also be an existing ``Pcf`` to copy.
        If `times` is given, the values :math:`v_i` instead, with shape (n,).
        The values and the times may be PyTorch, JAX, or TensorFlow arrays,
        and the PCF then carries gradients with respect to them through
        arithmetic and norms.
    dtype : type, optional
        Data type for the PCF (``pcf32``, ``pcf64``, ``pcf32i``, or ``pcf64i``).
        If ``None``, the dtype is inferred from the input array (e.g. a
        ``numpy.float32`` array produces a 32-bit PCF, a ``numpy.int32`` array
        produces a 32-bit integer PCF).
    times : array_like, optional
        The breakpoint times :math:`t_i`, with shape (n,), when `arr` holds
        only the values. Times traced abstractly, as by ``jax.grad`` or
        ``tf.function``, cannot be checked for order, and the PCF then
        supports norms but not arithmetic.

    Examples
    --------
    >>> import numpy as np
    >>> import stablebear as sb
    >>> f = sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.0]], dtype=np.float32))
    >>> f.size
    3
    """

    def _binary_io_data(self):
        return self._data

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

    # A native Pcf_f32_f32, Pcf_f64_f64, Pcf_i32_i32, or Pcf_i64_i64.
    _data: Any

    _CPP_TYPE_INFO = {
        cpp.Pcf_f32_f32: (float32, float32),
        cpp.Pcf_f64_f64: (float64, float64),
        cpp.Pcf_i32_i32: (int32, int32),
        cpp.Pcf_i64_i64: (int64, int64),
    }

    def __new__(cls, arr: Any = None, dtype: Any = None, *, times: Any = None) -> Pcf:
        if cls is Pcf and times is not None:
            content, ttimes, tvalues = _autodiff.content_from_values(times, arr)
            if ttimes is not None or tvalues is not None:
                # A PCF that carries gradients; see _tracked. Times traced
                # abstractly are NaN here, and their order cannot be checked.
                checked = None if np.isnan(content[:, 0]).any() else Pcf(content)
                return _TrackedPcf(
                    [ttimes if ttimes is not None else content[:, 0],
                     tvalues if tvalues is not None else content[:, 1]],
                    np.array([0, len(content)]), plain=checked)
        return super().__new__(cls)

    def __init__(
        self,
        arr: np.ndarray | Pcf | list[list[float | int] | tuple[float | int, ...]] | Any,
        dtype=None,
        *,
        times: Any = None,
    ):
        if times is not None:
            arr = _autodiff.content_from_values(times, arr)[0]
        elif _autodiff._framework(arr) is not None:
            raise TypeError(
                "To build a PCF from a PyTorch, JAX, or TensorFlow array, pass the values "
                "and the times separately: Pcf(values, times=times)."
            )

        if isinstance(arr, Pcf):
            self._data = arr._data
            self.ttype = arr.ttype
            self.vtype = arr.vtype

        elif isinstance(arr, np.ndarray):
            if dtype is not None:
                np_dtype = self._DTYPE_TO_NP.get(dtype, dtype)
                if arr.dtype != np_dtype:
                    arr = arr.astype(np_dtype)

            constructor = self._NP_TO_CPP.get(_sized_type(arr.dtype))
            if constructor is None:
                raise ValueError(
                    "Unsupported array type (must be np.float32/64 or np.int32/64)"
                )
            self._data = constructor(arr)
            self.ttype = _NP_TO_SB[_sized_type(arr.dtype)]
            self.vtype = _NP_TO_SB[_sized_type(arr.dtype)]

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

    def integrate(self) -> Plf:
        r"""The integral :math:`F(t) = \int_0^t f(u)\,du`, as a PLF.

        The PLF has the PCF's breakpoints. After the last one,
        :math:`t_n`, it grows with the PCF's last value :math:`v_n`. When
        that differs from the slope of the segment before :math:`t_n`, the
        PLF gets one more breakpoint, at :math:`t_n + \max(1, t_n)`, so
        that its last segment has slope :math:`v_n`.

        .. note::

           Any time after :math:`t_n` would give the same function. The step
           is :math:`\max(1, t_n)` rather than 1 because at large times, such
           as from :math:`2^{24}` in ``float32``, :math:`t_n + 1` rounds to
           :math:`t_n`.

        Returns
        -------
        Plf
            The integral, with the PCF's ``vtype`` as its dtype.

        Raises
        ------
        TypeError
            If the PCF's ``vtype`` is ``int32`` or ``int64``.
        ValueError
            If the PCF's times are not strictly increasing, which can happen
            for a PCF from
            :func:`~stablebear.serialize.from_serial_content`.

        Examples
        --------
        >>> import stablebear as sb
        >>> f = sb.Pcf([[0.0, 2.0], [1.0, -1.0], [3.0, 0.5]])
        >>> f.integrate().to_numpy()
        array([[0. , 0. ],
               [1. , 2. ],
               [3. , 0. ],
               [6. , 1.5]])
        """
        if self.vtype not in (float32, float64):
            raise TypeError("Only PCFs with vtype float32 or float64 can be integrated.")
        return Plf._from_cpp(self._data.integrate(), self.vtype)

    # The framework times and values of the breakpoints of a PCF that
    # carries gradients (see _tracked). None where they are constants, as
    # for any other PCF.
    _ttimes: Any = None
    _tvalues: Any = None

    def _serial(self):
        """(content, enumeration) as from_serial_content takes, for this one PCF."""
        pts = self.to_numpy()
        return pts, np.array([0, len(pts)])

    def _binop(self, rhs, op, combine):
        if isinstance(rhs, Pcf) and not _has_matching_types(self, rhs):
            raise TypeError("Mismatched PCF types")

        def primal_op(f, g):
            return Pcf(op(f._data, g._data if isinstance(g, Pcf) else g))

        return _autodiff.pcf_op(primal_op, combine, self, rhs)

    def _rbinop(self, lhs, op, combine):
        return _autodiff.pcf_op(lambda l, f: Pcf(op(f._data, l)), combine, lhs, self)

    def __add__(self, rhs):
        return self._binop(rhs, type(self._data).__add__, operator.add)

    def __sub__(self, rhs):
        return self._binop(rhs, type(self._data).__sub__, operator.sub)

    def __mul__(self, rhs):
        return self._binop(rhs, type(self._data).__mul__, operator.mul)

    def __radd__(self, lhs):
        return self._rbinop(lhs, type(self._data).__radd__, operator.add)

    def __rsub__(self, lhs):
        return self._rbinop(lhs, type(self._data).__rsub__, operator.sub)

    def __rmul__(self, lhs):
        return self._rbinop(lhs, type(self._data).__rmul__, operator.mul)

    def __truediv__(self, rhs):
        return self._binop(rhs, type(self._data).__truediv__, operator.truediv)

    def __rtruediv__(self, lhs):
        return self._rbinop(lhs, type(self._data).__rtruediv__, operator.truediv)

    def __neg__(self):
        return _autodiff.pcf_op(lambda f: Pcf(-f._data), operator.neg, self)

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


def iterate_rectangles(f: Pcf, g: Pcf, a=0.0, b=float('inf')):
    """Iterate over the rectangles formed by two PCFs.

    Parameters
    ----------
    f, g : Pcf
        The two piecewise constant functions.
    a : float, optional
        Left integration bound (default 0).
    b : float, optional
        Right integration bound (default infinity).

    Returns
    -------
    list of Rectangle
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
    cpp_rects = backend.iterate_rectangles(f._data, g._data, a, b)
    return [Rectangle(r) for r in cpp_rects]


def _float_types(np_dtype: Any) -> dict[str, Any]:
    sb_type = _NP_TO_SB[np_dtype]
    return {"ttype": sb_type, "vtype": sb_type}


Pcf._tracking = Tracking(  # pyright: ignore[reportAttributeAccessIssue]
    groups=(("_ttimes", 1), ("_tvalues", 1)),
    from_serial=_autodiff._pcfs_from_content,
    attributes=_float_types,
)

# A Pcf whose breakpoint times, values, or both depend on PyTorch, JAX, or
# TensorFlow arrays, built from [times, values] and the enumeration of
# _serial; see _tracked.
_TrackedPcf = tracked_class(Pcf)
