from __future__ import annotations

import datetime
import math
import numbers
import operator
from fractions import Fraction

import numpy as np

from .. import _sb_cpp as cpp
from ..base_tensor import BoolTensor, FloatTensor, IntTensor

# NumPy datetime64 resolutions with a fixed length, and the matching Times
# units. NumPy's "m" is minutes, which Times spells "min".
_NUMPY_TO_UNIT = {
    "as": "as", "fs": "fs", "ps": "ps", "ns": "ns", "us": "us", "ms": "ms",
    "s": "s", "m": "min", "h": "h", "D": "d",
}
_UNIT_TO_NUMPY = {unit: code for code, unit in _NUMPY_TO_UNIT.items()}
# The length of each Times unit in attoseconds, the finest of them.
_ATTOSECONDS = {
    "as": 1, "fs": 10**3, "ps": 10**6, "ns": 10**9, "us": 10**12, "ms": 10**15,
    "s": 10**18, "min": 60 * 10**18, "h": 3600 * 10**18, "d": 86400 * 10**18,
}


class _Counts:
    """A count of a unit, or a tensor of counts in one unit: the storage
    that :class:`DateTime` and :class:`TimeDelta` share."""

    # NumPy defers to the operators below instead of making object arrays.
    __array_ufunc__ = None

    @classmethod
    def _of(cls, count, unit):
        value = cls.__new__(cls)
        value._count, value._unit = count, unit
        return value

    @property
    def unit(self) -> str:
        """The resolution, as a ``Times`` unit such as ``"h"`` or ``"us"``."""
        return self._unit

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the counts, ``()`` for a single value."""
        return tuple(self._count.shape) if isinstance(self._count, (IntTensor, FloatTensor)) else ()

    @property
    def ndim(self) -> int:
        """The number of dimensions of the counts, 0 for a single value."""
        return len(self.shape)

    def _counts(self):
        """The count, or an array of the counts."""
        return np.asarray(self._count) if self.ndim else self._count

    def __len__(self) -> int:
        if not self.ndim:
            raise TypeError(f"A single {type(self).__name__} has no len()")
        return self.shape[0]

    def __getitem__(self, index):
        if not self.ndim:
            raise TypeError(f"A single {type(self).__name__} cannot be indexed")
        return self._of(self._count[index], self._unit)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def array_equal(self, other) -> bool:
        """Whether ``other`` is of the same type, with the same shape,
        counts, and unit."""
        return (isinstance(other, type(self)) and self._unit == other._unit
                and np.array_equal(np.asarray(self._count), np.asarray(other._count)))

    def __eq__(self, other) -> bool | BoolTensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        return _as_boolean(np.equal(self._counts(), other._counts()) & (self._unit == other._unit))

    def __ne__(self, other) -> bool | BoolTensor:
        if not isinstance(other, type(self)):
            return NotImplemented
        return _as_boolean(np.not_equal(self._counts(), other._counts()) | (self._unit != other._unit))

    def __hash__(self):
        if self.ndim:
            raise TypeError(f"A {type(self).__name__} of an array is unhashable")
        return hash((self._count, self._unit))


class DateTime(_Counts):
    """A calendar date and time: a count of a time unit after
    1970-01-01T00:00, as a ``np.datetime64`` stores it. The epoch of
    :class:`Times`, the instant its time 0 stands for, is one.

    Dates follow the Gregorian calendar and every day has 86400
    seconds: leap seconds are not counted.

    A ``DateTime`` can also hold an array of instants in one unit, as in
    ``DateTime(["2023-01-01T00:30", "2023-01-01T01:30"])``. Its
    :attr:`count` is then an ``IntTensor``. It has :attr:`shape`, ``len``,
    iteration, and indexing, which gives a ``DateTime`` of one instant or
    of an array of them, and :meth:`to_datetime64` gives a ``datetime64``
    array. Comparisons, ``==`` and ``!=`` included, give a ``BoolTensor``,
    as they do on tensors; :meth:`array_equal` compares two as a whole. It
    has no hash, and an epoch is a ``DateTime`` of one instant.

    Parameters
    ----------
    value : str, np.datetime64, datetime.datetime, datetime.date, DateTime, or array_like
        The instant, such as ``"2023-01-01T12:30"``. Its resolution is kept:
        ``"2023-01-01"`` counts days, a ``datetime`` counts microseconds, and
        a ``pd.Timestamp`` counts its own unit, such as nanoseconds. A
        timezone-aware value is converted to UTC and its timezone dropped.
        An array of instants is a ``datetime64`` array, a pandas
        ``DatetimeIndex`` or ``Series`` of dates, or a sequence of values
        a single ``DateTime`` accepts, kept at the finest of their
        resolutions.

    Raises
    ------
    TypeError
        If ``value`` is of another type.
    ValueError
        If ``value`` is or holds ``NaT`` or has a resolution of weeks,
        months, or years, which have no fixed length.

    Notes
    -----
    Two values are equal when they have the same count of the same unit; the
    same instant at another resolution is a different value. Compare
    :meth:`to_datetime64` values to compare instants. ``<``, ``<=``, ``>``,
    and ``>=`` compare instants.

    Adding or subtracting a :class:`TimeDelta`, or a duration it accepts,
    gives a ``DateTime`` at the finer of the two resolutions; a duration with
    a float count is first rounded as :meth:`TimeDelta.to_timedelta64`
    rounds it. Subtracting a ``DateTime`` gives the :class:`TimeDelta`
    between them.
    """

    def __init__(self, value):
        if isinstance(value, DateTime):
            self._count, self._unit = value._count, value._unit
            return
        if isinstance(value, (str, datetime.date, np.datetime64)):
            if isinstance(value, datetime.datetime) and value.tzinfo is not None:
                value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
            # A pd.Timestamp converts at its own resolution, such as nanoseconds.
            to_datetime64 = getattr(value, "to_datetime64", None)
            value = np.datetime64(value) if to_datetime64 is None else to_datetime64()
        else:
            given, value = value, _as_array(value)
            if value.ndim == 0 or value.dtype.kind not in "MOUS":
                raise TypeError(f"Expected a date, datetime, datetime64, or string, or an array of them; "
                                f"got {type(given).__name__}")
            if value.dtype.kind != "M":
                # Each converted as a single value is, at the finest of their
                # resolutions.
                value = np.array([DateTime(element).to_datetime64() for element in value.flat]).reshape(value.shape)
        if np.isnat(value).any():
            raise ValueError("A DateTime cannot be NaT")
        code, step = np.datetime_data(value.dtype)
        if code not in _NUMPY_TO_UNIT:
            raise ValueError(f"A DateTime needs a resolution of fixed length, not {code!r}")
        counts = value.astype(np.int64) * step
        self._count = IntTensor(counts) if counts.ndim else int(counts)
        self._unit = _NUMPY_TO_UNIT[code]

    @classmethod
    def _from_cpp(cls, data):
        return cls._of(data.count, data.unit)

    @property
    def _data(self):
        """The native value, for an epoch."""
        if self.ndim:
            raise TypeError("An epoch is a single DateTime, not an array of them")
        return cpp.DateTime(self._count, self._unit)

    @property
    def count(self) -> int | IntTensor:
        """The number of :attr:`unit` after 1970-01-01T00:00, or a tensor of
        them."""
        return self._count

    def to_datetime64(self) -> np.datetime64 | np.ndarray:
        """The value as a ``np.datetime64`` at its resolution, or a
        ``datetime64`` array."""
        return np.asarray(self._counts()).astype(f"datetime64[{_UNIT_TO_NUMPY[self._unit]}]")[()]

    def _compare(self, other, compare):
        if not isinstance(other, DateTime):
            return NotImplemented
        return _as_boolean(compare(self.to_datetime64(), other.to_datetime64()))

    def __lt__(self, other) -> bool | BoolTensor:
        return self._compare(other, operator.lt)

    def __le__(self, other) -> bool | BoolTensor:
        return self._compare(other, operator.le)

    def __gt__(self, other) -> bool | BoolTensor:
        return self._compare(other, operator.gt)

    def __ge__(self, other) -> bool | BoolTensor:
        return self._compare(other, operator.ge)

    def __add__(self, other):
        other = _single_duration(other)
        if other is None:
            return NotImplemented
        return DateTime(self.to_datetime64() + other.to_timedelta64())

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, DateTime):
            return TimeDelta(self.to_datetime64() - other.to_datetime64())
        other = _single_duration(other)
        if other is None:
            return NotImplemented
        return DateTime(self.to_datetime64() - other.to_timedelta64())

    def __reduce__(self):
        return DateTime, (self.to_datetime64(),)

    def __repr__(self):
        if self.ndim:
            return f"DateTime({np.array2string(self.to_datetime64(), separator=', ')})"
        return f"DateTime({str(self.to_datetime64())!r})"


class TimeDelta(_Counts):
    """A duration: a count of a time unit, such as 1.5 hours.

    The count is an integer, as a ``np.timedelta64`` stores it, or a float,
    as a time series stores its times. Every day has 86400 seconds; weeks,
    months, and years, which have no fixed length, are not units.

    A ``TimeDelta`` can also hold an array of counts in one unit, as in
    ``TimeDelta(sb.FloatTensor([[30, 90], [240, 60]]), "min")``. Its
    :attr:`count` is then an ``IntTensor`` of integers or a ``FloatTensor``
    of floats. It has :attr:`shape`, ``len``, iteration, and indexing,
    which gives a ``TimeDelta`` of one count or of an array of them, and
    :meth:`to_timedelta64` gives a ``timedelta64`` array. Arithmetic works
    elementwise, and comparisons, ``==`` and ``!=`` included, give a
    ``BoolTensor``, as they do on tensors; :meth:`array_equal` compares two
    as a whole. It has no hash, and a :class:`DateTime` takes only a
    ``TimeDelta`` of one count.

    Parameters
    ----------
    value : np.timedelta64, datetime.timedelta, pd.Timedelta, TimeDelta, number, or array_like
        The duration, such as ``np.timedelta64(3, "h")``, or a
        ``timedelta64`` array. Its resolution is kept: a ``timedelta``
        counts microseconds, and a ``pd.Timedelta`` counts its own unit,
        such as nanoseconds. With ``unit``, a number of ``unit``, as in
        ``TimeDelta(1.5, "h")``, where a float that is a whole number
        becomes an integer; or an ``IntTensor``, ``FloatTensor``, array, or
        nested list of them.
    unit : str, optional
        The unit of numbers ``value``: one of ``"as"``, ``"fs"``, ``"ps"``,
        ``"ns"``, ``"us"``, ``"ms"``, ``"s"``, ``"min"``, ``"h"``, and
        ``"d"``.

    Raises
    ------
    TypeError
        If ``value`` is of another type, or is a number without ``unit``.
    ValueError
        If ``value`` is or holds ``NaT``, is not finite, or has a resolution
        of weeks, months, or years, or ``unit`` is unknown.

    Notes
    -----
    Two values are equal when they have the same count of the same unit; the
    same length in another unit is a different value. Compare
    :meth:`to_timedelta64` values to compare lengths. ``<``, ``<=``, ``>``,
    and ``>=`` compare lengths.

    Adding or subtracting another duration, which may be any value
    ``TimeDelta`` accepts, gives a ``TimeDelta`` in the finer of the two
    units; each count is converted by an exact factor. Negating it or
    multiplying it by a number keeps its unit.
    """

    def __init__(self, value, unit: str | None = None):
        if unit is not None:
            if unit not in _UNIT_TO_NUMPY:
                raise ValueError(f"Unknown unit {unit!r}; expected one of {', '.join(_UNIT_TO_NUMPY)}")
            if not isinstance(value, numbers.Real):
                self._count = _array_counts(value, unit)
            elif isinstance(value, numbers.Integral):
                self._count = int(value)
            else:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("A TimeDelta needs a finite count")
                self._count = int(value) if value.is_integer() else value
            self._unit = unit
            return
        if isinstance(value, TimeDelta):
            self._count, self._unit = value._count, value._unit
            return
        if hasattr(value, "to_timedelta64"):
            # A pd.Timedelta, at its own resolution.
            value = value.to_timedelta64()
        elif isinstance(value, datetime.timedelta):
            value = np.timedelta64(value)
        elif not (isinstance(value, np.timedelta64)
                  or (isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.timedelta64))):
            raise TypeError(f"Expected a timedelta, timedelta64, or TimeDelta; got {type(value).__name__}")
        if np.isnat(value).any():
            raise ValueError("A TimeDelta cannot be NaT")
        code, step = np.datetime_data(value.dtype)
        if code not in _NUMPY_TO_UNIT:
            raise ValueError(f"A TimeDelta needs a unit of fixed length, not {code!r}")
        counts = value.astype(np.int64) * step
        self._count = IntTensor(counts) if counts.ndim else int(counts)
        self._unit = _NUMPY_TO_UNIT[code]

    @property
    def count(self) -> int | float | IntTensor | FloatTensor:
        """The number of :attr:`unit` in the duration, or a tensor of them."""
        return self._count

    @classmethod
    def _of(cls, count, unit):
        return TimeDelta(count, unit)

    def to_timedelta64(self, resolution: str | None = None) -> np.timedelta64:
        """The value as a ``np.timedelta64``, rounded to the nearest step of
        ``resolution``.

        ``resolution`` is one of the units of :class:`TimeDelta`. It
        defaults to :attr:`unit` for an integer count, and otherwise to the
        finer of :attr:`unit` and microseconds.

        Raises
        ------
        ValueError
            If ``resolution`` is unknown.
        """
        if resolution is None:
            fine = isinstance(self._count, (int, IntTensor)) or _ATTOSECONDS[self._unit] < _ATTOSECONDS["us"]
            resolution = self._unit if fine else "us"
        if resolution not in _ATTOSECONDS:
            raise ValueError(f"Unknown resolution {resolution!r}; expected one of {', '.join(_ATTOSECONDS)}")
        if self.ndim:
            # Each count rounded exactly, as a single count is.
            steps = [TimeDelta(count, self._unit).to_timedelta64(resolution) for count in np.asarray(self._count).flat]
            return np.array(steps, dtype=f"timedelta64[{_UNIT_TO_NUMPY[resolution]}]").reshape(self.shape)
        steps = round(Fraction(self._count) * _ATTOSECONDS[self._unit] / _ATTOSECONDS[resolution])
        return np.timedelta64(steps, _UNIT_TO_NUMPY[resolution])

    def _length(self):
        """The duration in attoseconds, exactly."""
        return Fraction(self._count) * _ATTOSECONDS[self._unit]

    def _compare(self, other, compare):
        other = _as_time_delta(other)
        if other is None:
            return NotImplemented
        if not (self.ndim or other.ndim):
            return compare(self._length(), other._length())
        unit = min(self._unit, other._unit, key=_ATTOSECONDS.__getitem__)
        return BoolTensor(compare(self._in(unit), other._in(unit)))

    def __lt__(self, other) -> bool | BoolTensor:
        return self._compare(other, operator.lt)

    def __le__(self, other) -> bool | BoolTensor:
        return self._compare(other, operator.le)

    def __gt__(self, other) -> bool | BoolTensor:
        return self._compare(other, operator.gt)

    def __ge__(self, other) -> bool | BoolTensor:
        return self._compare(other, operator.ge)

    def _in(self, unit):
        """The count in ``unit``, a unit at least as fine, by an exact factor."""
        return self._counts() * (_ATTOSECONDS[self._unit] // _ATTOSECONDS[unit])

    def __add__(self, other):
        other = _as_time_delta(other)
        if other is None:
            return NotImplemented
        unit = min(self._unit, other._unit, key=_ATTOSECONDS.__getitem__)
        return TimeDelta(self._in(unit) + other._in(unit), unit)

    __radd__ = __add__

    def __sub__(self, other):
        other = _as_time_delta(other)
        return NotImplemented if other is None else self + -other

    def __rsub__(self, other):
        other = _as_time_delta(other)
        return NotImplemented if other is None else other + -self

    def __neg__(self):
        return TimeDelta(-self._counts(), self._unit)

    def __mul__(self, factor):
        if not isinstance(factor, numbers.Real):
            return NotImplemented
        return TimeDelta(self._counts() * factor, self._unit)

    __rmul__ = __mul__

    def __reduce__(self):
        return TimeDelta, (self._count, self._unit)

    def __repr__(self):
        count = np.array2string(np.asarray(self._count), separator=", ") if self.ndim else self._count
        return f"TimeDelta({count}, {self._unit!r})"


def _array_counts(value, unit):
    """``value``, an array-like of numbers of ``unit``, as an ``IntTensor`` of
    integers or a ``FloatTensor`` of floats."""
    counts = np.asarray(value)
    if counts.ndim == 0:
        raise TypeError(f"Expected a number or an array of numbers of {unit!r}; got {type(value).__name__}")
    if np.issubdtype(counts.dtype, np.integer):
        return IntTensor(counts.astype(np.int64))
    if np.issubdtype(counts.dtype, np.floating):
        if not np.isfinite(counts).all():
            raise ValueError("A TimeDelta needs finite counts")
        return FloatTensor(counts.astype(np.float64))
    raise TypeError(f"Expected numbers of {unit!r}; got {counts.dtype} counts")


def _as_boolean(equal):
    """``equal``, a NumPy result, as a ``bool`` or a ``BoolTensor``."""
    return BoolTensor(equal) if np.ndim(equal) else bool(equal)


def _single_duration(value):
    """``value`` as a :class:`TimeDelta` of one count if it is a duration,
    else ``None``."""
    duration = _as_time_delta(value)
    if duration is not None and duration.ndim:
        raise TypeError("A DateTime takes a single duration, not an array of them")
    return duration


def _as_time_delta(value):
    """``value`` as a :class:`TimeDelta` if it is a duration, else ``None``."""
    if isinstance(value, TimeDelta):
        return value
    if isinstance(value, (np.timedelta64, datetime.timedelta)) or hasattr(value, "to_timedelta64"):
        return TimeDelta(value)
    return None


def _as_array(values):
    """``values`` as an array. Dates with a timezone, as pandas holds them,
    become naive UTC ``datetime64`` at their resolution."""
    dtype = getattr(values, "dtype", None)
    if getattr(dtype, "tz", None) is not None:
        return values.to_numpy(dtype=f"datetime64[{dtype.unit}]")
    return np.asarray(values)


def _duration(value, message):
    """``value`` as a positive duration: a :class:`TimeDelta` for a duration,
    such as a ``timedelta64`` or ``pd.Timedelta``, kept at its resolution, or
    else a float. Raises ``ValueError`` with ``message`` if it is not
    positive or is ``NaT``, and for months and years, which have no fixed
    length."""
    if isinstance(value, np.timedelta64) and np.isnat(value):
        raise ValueError(message)
    duration = _as_time_delta(value)
    if duration is not None:
        if duration.count <= 0:
            raise ValueError(message)
        return duration
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(message)
    return value


def _in_unit(duration, unit):
    """A duration from :func:`_duration` as a number of ``unit``, a Times unit;
    a :class:`TimeDelta` is converted by an exact factor and rounded once."""
    if isinstance(duration, TimeDelta):
        return float(Fraction(duration.count) * _ATTOSECONDS[duration.unit] / _ATTOSECONDS[unit])
    return duration
