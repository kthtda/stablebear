from __future__ import annotations

import datetime
import math
import numbers
import operator
from fractions import Fraction
from typing import Any, Generic, TypeVar, overload

import numpy as np

from .. import _sb_cpp as cpp
from ..base_tensor import BoolTensor, FloatTensor, IntTensor

# NumPy datetime64 resolutions with a fixed length, and the matching time
# units. NumPy's "m" is minutes, which stablebear spells "min".
_NUMPY_TO_UNIT = {
    "as": "as", "fs": "fs", "ps": "ps", "ns": "ns", "us": "us", "ms": "ms",
    "s": "s", "m": "min", "h": "h", "D": "d",
}
_UNIT_TO_NUMPY = {unit: code for code, unit in _NUMPY_TO_UNIT.items()}
# The length of each time unit in attoseconds, the finest of them.
_ATTOSECONDS = {
    "as": 1, "fs": 10**3, "ps": 10**6, "ns": 10**9, "us": 10**12, "ms": 10**15,
    "s": 10**18, "min": 60 * 10**18, "h": 3600 * 10**18, "d": 86400 * 10**18,
}


# The type of the count of a date or duration: a number, or a tensor of them.
_CountT = TypeVar("_CountT", bound="int | float | IntTensor | FloatTensor")
_TensorCountT = TypeVar("_TensorCountT", bound="IntTensor | FloatTensor")


class _Counts(Generic[_CountT]):
    """A count of a unit, or a tensor of counts in one unit: the storage
    that dates and durations share."""

    # NumPy defers to the operators below instead of making object arrays.
    __array_ufunc__ = None

    # The count (int, float, IntTensor, or FloatTensor) of ``_unit``,
    # published with its type by ``count``.
    _count: Any
    _unit: str

    @property
    def _family(self) -> type:
        """The base class of the values this one compares with."""
        raise NotImplementedError

    @classmethod
    def _of(cls, count, unit) -> Any:
        """A value of ``count`` of ``unit``."""
        raise NotImplementedError

    @property
    def unit(self) -> str:
        """The resolution, as a time unit such as ``"h"`` or ``"us"``."""
        return self._unit

    @property
    def _array(self):
        """Whether this holds a tensor of counts."""
        return isinstance(self._count, (IntTensor, FloatTensor))

    def _counts(self):
        """The count, or an array of the counts."""
        return np.asarray(self._count) if self._array else self._count

    @overload
    def __eq__(self, other: DateTimeTensor | TimeDeltaTensor) -> BoolTensor: ...

    @overload
    def __eq__(self: _CountsTensor[Any], other: object) -> BoolTensor: ...

    @overload
    def __eq__(self, other: object) -> bool: ...

    def __eq__(self, other):  # pyright: ignore[reportIncompatibleMethodOverride]
        if not isinstance(other, self._family):
            return NotImplemented
        return _as_boolean(np.equal(self._counts(), other._counts()) & (self._unit == other._unit))

    @overload
    def __ne__(self, other: DateTimeTensor | TimeDeltaTensor) -> BoolTensor: ...

    @overload
    def __ne__(self: _CountsTensor[Any], other: object) -> BoolTensor: ...

    @overload
    def __ne__(self, other: object) -> bool: ...

    def __ne__(self, other):  # pyright: ignore[reportIncompatibleMethodOverride]
        if not isinstance(other, self._family):
            return NotImplemented
        return _as_boolean(np.not_equal(self._counts(), other._counts()) | (self._unit != other._unit))

    def __hash__(self):
        return hash((self._count, self._unit))


class _CountsTensor(_Counts[_TensorCountT]):
    """What :class:`DateTimeTensor` and :class:`TimeDeltaTensor` share."""

    __hash__ = None  # type: ignore[assignment]

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the tensor."""
        return tuple(self._count.shape)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the tensor."""
        return len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index):
        return self._of(self._count[index], self._unit)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def array_equal(self, other) -> bool:
        """Whether ``other`` is of the same type, with the same shape,
        counts, and unit."""
        return (isinstance(other, type(self)) and self._unit == other._unit
                and np.array_equal(np.asarray(self._count), np.asarray(other._count)))


class _Instants(_Counts[_CountT]):
    """The arithmetic that :class:`DateTime` and :class:`DateTimeTensor`
    share."""

    @property
    def _family(self):
        return _Instants

    @overload
    @classmethod
    def _of(cls, count: IntTensor, unit: str) -> DateTimeTensor: ...

    @overload
    @classmethod
    def _of(cls, count: int, unit: str) -> DateTime: ...

    @classmethod
    def _of(cls, count, unit):
        kind = DateTimeTensor if isinstance(count, IntTensor) else DateTime
        value = kind.__new__(kind)
        value._count, value._unit = count, unit
        return value

    @property
    def count(self) -> _CountT:
        """The number of ``unit`` after ``1970-01-01T00:00``, or a tensor of
        them."""
        return self._count

    @overload
    def to_datetime64(self: DateTime) -> np.datetime64: ...

    @overload
    def to_datetime64(self: DateTimeTensor) -> np.ndarray: ...

    @overload
    def to_datetime64(self) -> np.datetime64 | np.ndarray: ...

    def to_datetime64(self) -> np.datetime64 | np.ndarray:
        """The value as a ``np.datetime64`` at its resolution, or a
        ``datetime64`` array."""
        return np.asarray(self._counts()).astype(f"datetime64[{_UNIT_TO_NUMPY[self._unit]}]")[()]

    def _compare(self, other, compare):
        if not isinstance(other, _Instants):
            return NotImplemented
        return _as_boolean(compare(self.to_datetime64(), other.to_datetime64()))

    @overload
    def __lt__(self: DateTime, other: DateTime) -> bool: ...

    @overload
    def __lt__(self, other: DateTime | DateTimeTensor) -> BoolTensor: ...

    def __lt__(self, other):
        return self._compare(other, operator.lt)

    @overload
    def __le__(self: DateTime, other: DateTime) -> bool: ...

    @overload
    def __le__(self, other: DateTime | DateTimeTensor) -> BoolTensor: ...

    def __le__(self, other):
        return self._compare(other, operator.le)

    @overload
    def __gt__(self: DateTime, other: DateTime) -> bool: ...

    @overload
    def __gt__(self, other: DateTime | DateTimeTensor) -> BoolTensor: ...

    def __gt__(self, other):
        return self._compare(other, operator.gt)

    @overload
    def __ge__(self: DateTime, other: DateTime) -> bool: ...

    @overload
    def __ge__(self, other: DateTime | DateTimeTensor) -> BoolTensor: ...

    def __ge__(self, other):
        return self._compare(other, operator.ge)

    @overload
    def __add__(self: DateTime, other: _ScalarDuration) -> DateTime: ...

    @overload
    def __add__(self, other: Duration) -> DateTimeTensor: ...

    def __add__(self, other):
        other = _as_time_delta(other)
        if other is None:
            return NotImplemented
        return _date_time(self.to_datetime64() + other.to_timedelta64())

    __radd__ = __add__

    @overload
    def __sub__(self: DateTime, other: DateTime) -> TimeDelta: ...

    @overload
    def __sub__(self, other: DateTime | DateTimeTensor) -> TimeDeltaTensor: ...

    @overload
    def __sub__(self: DateTime, other: _ScalarDuration) -> DateTime: ...

    @overload
    def __sub__(self, other: Duration) -> DateTimeTensor: ...

    def __sub__(self, other):
        if isinstance(other, _Instants):
            return _time_delta(self.to_datetime64() - other.to_datetime64())
        other = _as_time_delta(other)
        if other is None:
            return NotImplemented
        return _date_time(self.to_datetime64() - other.to_timedelta64())

    def __reduce__(self):
        return type(self), (self.to_datetime64(),)


class DateTime(_Instants[int]):
    """A calendar date and time: a count of a time unit after
    ``1970-01-01T00:00``, as a ``np.datetime64`` stores it. The epoch of
    :class:`TimeSeries`, the instant its time 0 stands for, is one.

    Dates follow the Gregorian calendar and every day has 86400
    seconds: leap seconds are not counted. A ``DateTime`` holds one
    instant; :class:`DateTimeTensor` holds an array of them.

    Parameters
    ----------
    value : str, np.datetime64, datetime.datetime, datetime.date, or DateTime
        The instant, such as ``"2023-01-01T12:30"``. Its resolution is kept:
        ``"2023-01-01"`` counts days, a ``datetime`` counts microseconds, and
        a ``pd.Timestamp`` counts its own unit, such as nanoseconds. A
        timezone-aware value is converted to UTC and its timezone dropped.

    Raises
    ------
    TypeError
        If ``value`` is of another type.
    ValueError
        If ``value`` is ``NaT`` or has a resolution of weeks, months, or
        years, which have no fixed length.

    Notes
    -----
    Two values are equal when they have the same count of the same unit; the
    same instant at another resolution is a different value. Compare
    ``to_datetime64()`` values to compare instants. ``<``, ``<=``, ``>``,
    and ``>=`` compare instants.

    Adding or subtracting a :class:`TimeDelta`, or a duration it accepts,
    gives a ``DateTime`` at the finer of the two resolutions; a duration with
    a float count is first rounded as ``TimeDelta.to_timedelta64()``
    rounds it. Subtracting a ``DateTime`` gives the :class:`TimeDelta`
    between them. With a :class:`TimeDeltaTensor` or a ``DateTimeTensor``,
    the result is a ``DateTimeTensor`` or a ``TimeDeltaTensor``.
    """

    def __init__(self, value):
        if isinstance(value, DateTime):
            self._count, self._unit = value._count, value._unit
            return
        if not isinstance(value, (str, datetime.date, np.datetime64)):
            raise TypeError(f"Expected a date, datetime, datetime64, or string; got {type(value).__name__}")
        if isinstance(value, datetime.datetime) and value.tzinfo is not None:
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        # A pd.Timestamp converts at its own resolution, such as nanoseconds.
        to_datetime64 = getattr(value, "to_datetime64", None)
        value = np.datetime64(value) if to_datetime64 is None else to_datetime64()
        self._count, self._unit = _numpy_counts(value)

    @classmethod
    def _from_cpp(cls, data):
        return cls._of(data.count, data.unit)

    @property
    def _data(self):
        """The native value, for an epoch."""
        return cpp.DateTime(self._count, self._unit)

    def __repr__(self):
        return f"DateTime({str(self.to_datetime64())!r})"


class DateTimeTensor(_CountsTensor[IntTensor], _Instants[IntTensor]):
    """A tensor of calendar dates and times in one unit.

    Its ``count`` is an ``IntTensor`` of counts after
    ``1970-01-01T00:00``. It has ``shape``, ``len``, iteration, and
    indexing, which gives a :class:`DateTime` for one element and a
    ``DateTimeTensor`` otherwise, and ``to_datetime64()`` gives a
    ``datetime64`` array. It has no hash, and it cannot be an epoch.

    Parameters
    ----------
    value : array_like
        A ``datetime64`` array, a pandas ``DatetimeIndex`` or ``Series`` of
        dates, or a sequence of values :class:`DateTime` accepts, such as
        ``["2023-01-01T00:30", "2023-01-01T01:30"]``, kept at the finest of
        their resolutions. A timezone is converted to UTC and dropped.

    Raises
    ------
    TypeError
        If ``value`` is of another type or is not an array.
    ValueError
        If ``value`` holds ``NaT`` or has a resolution of weeks, months, or
        years.

    Notes
    -----
    Arithmetic and comparisons work elementwise, with a single date or
    duration broadcast against the tensor, as :class:`DateTime` does them.
    ``<``, ``<=``, ``>``, ``>=``, ``==``, and ``!=`` give a ``BoolTensor``,
    as they do on tensors; ``==`` compares count and unit.
    ``array_equal()`` tells whether two are equal as a whole.
    """

    def __init__(self, value):
        if isinstance(value, DateTimeTensor):
            self._count, self._unit = value._count, value._unit
            return
        given, value = value, _as_array(value)
        if value.ndim == 0 or value.dtype.kind not in "MOUS":
            raise TypeError(f"Expected an array of dates and times; got {type(given).__name__}")
        if value.dtype.kind != "M":
            # Each converted as a single value is, at the finest of their
            # resolutions.
            value = np.array([DateTime(element).to_datetime64() for element in value.flat]).reshape(value.shape)
        self._count, self._unit = _numpy_counts(value)

    def __repr__(self):
        return f"DateTimeTensor({np.array2string(self.to_datetime64(), separator=', ')})"


def _date_time(value):
    """A :class:`DateTime` of ``value``, a ``datetime64``, or a
    :class:`DateTimeTensor` of an array of them."""
    return (DateTimeTensor if np.ndim(value) else DateTime)(value)


class _Duration(_Counts[_CountT]):
    """The arithmetic that :class:`TimeDelta` and :class:`TimeDeltaTensor`
    share."""

    @property
    def _family(self):
        return _Duration

    @property
    def count(self) -> _CountT:
        """The number of ``unit`` in the duration, or a tensor of them."""
        return self._count

    @classmethod
    def _of(cls, count, unit):
        return _time_delta(count, unit)

    @overload
    def to_timedelta64(self: TimeDelta, resolution: str | None = None) -> np.timedelta64: ...

    @overload
    def to_timedelta64(self: TimeDeltaTensor, resolution: str | None = None) -> np.ndarray: ...

    @overload
    def to_timedelta64(self, resolution: str | None = None) -> np.timedelta64 | np.ndarray: ...

    def to_timedelta64(self, resolution: str | None = None) -> np.timedelta64 | np.ndarray:
        """The value as a ``np.timedelta64``, or a ``timedelta64`` array,
        rounded to the nearest step of ``resolution``.

        ``resolution`` is one of the units of :class:`TimeDelta`. It
        defaults to ``unit`` for integer counts, and otherwise to the
        finer of ``unit`` and microseconds.

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
        if self._array:
            # Each count rounded exactly, as a single count is.
            steps = [TimeDelta(count, self._unit).to_timedelta64(resolution) for count in np.asarray(self._count).flat]
            return np.array(steps, dtype=f"timedelta64[{_UNIT_TO_NUMPY[resolution]}]").reshape(np.shape(self._counts()))
        steps = round(self._length() / _ATTOSECONDS[resolution])
        return np.timedelta64(steps, _UNIT_TO_NUMPY[resolution])

    def _length(self):
        """The duration in attoseconds, exactly."""
        return Fraction(self._count) * _ATTOSECONDS[self._unit]

    def _compare(self, other, compare):
        other = _as_time_delta(other)
        if other is None:
            return NotImplemented
        if not (self._array or other._array):
            return compare(self._length(), other._length())
        unit = min(self._unit, other._unit, key=_ATTOSECONDS.__getitem__)
        return BoolTensor(compare(self._in(unit), other._in(unit)))

    @overload
    def __lt__(self: TimeDelta, other: _ScalarDuration) -> bool: ...

    @overload
    def __lt__(self, other: Duration) -> BoolTensor: ...

    def __lt__(self, other):
        return self._compare(other, operator.lt)

    @overload
    def __le__(self: TimeDelta, other: _ScalarDuration) -> bool: ...

    @overload
    def __le__(self, other: Duration) -> BoolTensor: ...

    def __le__(self, other):
        return self._compare(other, operator.le)

    @overload
    def __gt__(self: TimeDelta, other: _ScalarDuration) -> bool: ...

    @overload
    def __gt__(self, other: Duration) -> BoolTensor: ...

    def __gt__(self, other):
        return self._compare(other, operator.gt)

    @overload
    def __ge__(self: TimeDelta, other: _ScalarDuration) -> bool: ...

    @overload
    def __ge__(self, other: Duration) -> BoolTensor: ...

    def __ge__(self, other):
        return self._compare(other, operator.ge)

    def _in(self, unit):
        """The count in ``unit``, a unit at least as fine, by an exact factor."""
        return self._counts() * (_ATTOSECONDS[self._unit] // _ATTOSECONDS[unit])

    @overload
    def __add__(self: TimeDelta, other: _ScalarDuration) -> TimeDelta: ...

    @overload
    def __add__(self, other: Duration) -> TimeDeltaTensor: ...

    def _plus(self, other):
        """The sum with ``other``, a duration, in the finer of the two units."""
        unit = min(self._unit, other._unit, key=_ATTOSECONDS.__getitem__)
        return _time_delta(self._in(unit) + other._in(unit), unit)

    def _negated(self):
        return _time_delta(-self._counts(), self._unit)

    def __add__(self, other):
        other = _as_time_delta(other)
        return NotImplemented if other is None else self._plus(other)

    __radd__ = __add__

    @overload
    def __sub__(self: TimeDelta, other: _ScalarDuration) -> TimeDelta: ...

    @overload
    def __sub__(self, other: Duration) -> TimeDeltaTensor: ...

    def __sub__(self, other):
        other = _as_time_delta(other)
        return NotImplemented if other is None else self._plus(other._negated())

    @overload
    def __rsub__(self: TimeDelta, other: _ScalarDuration) -> TimeDelta: ...

    @overload
    def __rsub__(self, other: Duration) -> TimeDeltaTensor: ...

    def __rsub__(self, other):
        other = _as_time_delta(other)
        return NotImplemented if other is None else other._plus(self._negated())

    @overload
    def __neg__(self: TimeDelta) -> TimeDelta: ...

    @overload
    def __neg__(self: TimeDeltaTensor) -> TimeDeltaTensor: ...

    def __neg__(self):
        return self._negated()

    @overload
    def __mul__(self: TimeDelta, factor: float) -> TimeDelta: ...

    @overload
    def __mul__(self: TimeDeltaTensor, factor: float) -> TimeDeltaTensor: ...

    def __mul__(self, factor):
        if not isinstance(factor, numbers.Real):
            return NotImplemented
        return _time_delta(self._counts() * factor, self._unit)

    __rmul__ = __mul__

    def __reduce__(self):
        return type(self), (self._count, self._unit)


class TimeDelta(_Duration[int | float]):
    """A duration: a count of a time unit, such as 1.5 hours.

    The count is an integer, as a ``np.timedelta64`` stores it, or a float,
    as a time series stores its times. Every day has 86400 seconds; weeks,
    months, and years, which have no fixed length, are not units. A
    ``TimeDelta`` holds one duration; :class:`TimeDeltaTensor` holds an
    array of them.

    Parameters
    ----------
    value : np.timedelta64, datetime.timedelta, pd.Timedelta, TimeDelta, or number
        The duration, such as ``np.timedelta64(3, "h")``. Its resolution is
        kept: a ``timedelta`` counts microseconds, and a ``pd.Timedelta``
        counts its own unit, such as nanoseconds. With ``unit``, a number of
        ``unit``, as in ``TimeDelta(1.5, "h")``, where a float that is a
        whole number becomes an integer.
    unit : str, optional
        The unit of a number ``value``: one of ``"as"``, ``"fs"``, ``"ps"``,
        ``"ns"``, ``"us"``, ``"ms"``, ``"s"``, ``"min"``, ``"h"``, and
        ``"d"``, or an alias such as ``"hours"`` or ``"\u00b5s"``.
        ``unit`` reports the short name.

    Raises
    ------
    TypeError
        If ``value`` is of another type, or is a number without ``unit``.
    ValueError
        If ``value`` is ``NaT``, is not finite, or has a resolution of
        weeks, months, or years, or ``unit`` is unknown.

    Notes
    -----
    Two values are equal when they have the same count of the same unit; the
    same length in another unit is a different value. Compare
    ``to_timedelta64()`` values to compare lengths. ``<``, ``<=``, ``>``,
    and ``>=`` compare lengths.

    Adding or subtracting another duration, which may be any value
    ``TimeDelta`` accepts, gives a ``TimeDelta`` in the finer of the two
    units; each count is converted by an exact factor. Negating it or
    multiplying it by a number keeps its unit. With a ``TimeDeltaTensor``,
    the result is a ``TimeDeltaTensor``.
    """

    def __init__(self, value, unit: str | None = None):
        if isinstance(value, TimeDelta):
            self._count, self._unit = value._count, value._unit
            return
        if unit is not None:
            unit = _unit_name(unit)
            if isinstance(value, numbers.Integral):
                self._count = int(value)
            elif isinstance(value, numbers.Real):
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("A TimeDelta needs a finite count")
                self._count = int(value) if value.is_integer() else value
            else:
                raise TypeError(f"Expected a number of {unit!r}; got {type(value).__name__}")
            self._unit = unit
            return
        if hasattr(value, "to_timedelta64"):
            # A pd.Timedelta, at its own resolution.
            value = value.to_timedelta64()
        elif isinstance(value, datetime.timedelta):
            value = np.timedelta64(value)
        elif not isinstance(value, np.timedelta64):
            raise TypeError(f"Expected a timedelta, timedelta64, or TimeDelta; got {type(value).__name__}")
        self._count, self._unit = _numpy_counts(value)

    def __repr__(self):
        return f"TimeDelta({self._count}, {self._unit!r})"


class TimeDeltaTensor(_CountsTensor[IntTensor | FloatTensor], _Duration[IntTensor | FloatTensor]):
    """A tensor of durations in one unit, such as the times at which to
    evaluate a time series.

    Its ``count`` is an ``IntTensor`` of integer counts or a
    ``FloatTensor`` of float counts. It has ``shape``, ``len``,
    iteration, and indexing, which gives a :class:`TimeDelta` for one
    element and a ``TimeDeltaTensor`` otherwise, and ``to_timedelta64()``
    gives a ``timedelta64`` array. It has no hash.

    Parameters
    ----------
    value : IntTensor, FloatTensor, array_like, timedelta64 array, pd.TimedeltaIndex, or sequence of durations
        With ``unit``, numbers of ``unit``, as in
        ``TimeDeltaTensor(sb.FloatTensor([[30, 90], [240, 60]]), "min")``,
        kept as a ``FloatTensor``, or an ``IntTensor`` for integers so that
        their counts stay exact. Without ``unit``, a ``timedelta64`` array or
        a ``TimedeltaIndex``, kept at its resolution, or a sequence of
        values :class:`TimeDelta` accepts, in the finest of their units.
    unit : str, optional
        The unit of numbers ``value``, one of the units of
        :class:`TimeDelta`.

    Raises
    ------
    TypeError
        If ``value`` is of another type or is not an array, or holds numbers
        without ``unit``.
    ValueError
        If ``value`` holds ``NaT`` or counts that are not finite, or has a
        resolution of weeks, months, or years, or ``unit`` is unknown.

    Notes
    -----
    Arithmetic and comparisons work elementwise, with a single duration
    broadcast against the tensor, as :class:`TimeDelta` does them. ``<``,
    ``<=``, ``>``, ``>=``, ``==``, and ``!=`` give a ``BoolTensor``, as they
    do on tensors; ``==`` compares count and unit. ``array_equal()`` tells
    whether two are equal as a whole.
    """

    def __init__(self, value, unit: str | None = None):
        if isinstance(value, TimeDeltaTensor):
            self._count, self._unit = value._count, value._unit
            return
        if unit is not None:
            unit = _unit_name(unit)
            self._count, self._unit = _array_counts(value, unit), unit
            return
        array = _as_array(value)
        if array.ndim == 0:
            raise TypeError(f"Expected an array of durations; got {type(value).__name__}")
        if array.dtype.kind == "m":
            self._count, self._unit = _numpy_counts(array)
            return
        given = [_as_time_delta(element) for element in array.flat] if array.dtype == object else [None]
        durations = [duration for duration in given if duration is not None]
        if len(durations) < len(given):
            raise TypeError("Expected durations, or numbers with a unit")
        unit = min((duration.unit for duration in durations), key=_ATTOSECONDS.__getitem__)
        counts = np.array([duration._in(unit) for duration in durations]).reshape(array.shape)
        self._count, self._unit = _array_counts(counts, unit), unit

    def __repr__(self):
        return f"TimeDeltaTensor({np.asarray(self._count).tolist()}, {self._unit!r})"


def _time_delta(value, unit=None):
    """A :class:`TimeDelta` of ``value``, or a :class:`TimeDeltaTensor` of an
    array of them."""
    array = isinstance(value, (IntTensor, FloatTensor)) or np.ndim(value) > 0
    return (TimeDeltaTensor if array else TimeDelta)(value, unit)


def _unit_name(unit):
    """The short name of ``unit``, a time unit or an alias such as
    ``"hours"``; raises ``ValueError`` for an unknown unit."""
    return cpp.time_unit_name(unit)


def _numpy_counts(value):
    """The integer count, or an ``IntTensor`` of counts, and the unit of
    ``value``, a ``datetime64`` or ``timedelta64``, or an array of them."""
    if np.isnat(value).any():
        raise ValueError("Dates, times, and durations cannot be NaT")
    code, step = np.datetime_data(value.dtype)
    if code not in _NUMPY_TO_UNIT:
        raise ValueError(f"Dates, times, and durations need a resolution of fixed length, not {code!r}")
    counts = value.astype(np.int64) * step
    return IntTensor(counts) if counts.ndim else int(counts), _NUMPY_TO_UNIT[code]


def _array_counts(value, unit):
    """``value``, an array-like of numbers of ``unit``, as an ``IntTensor`` of
    integers or a ``FloatTensor`` of floats."""
    counts = np.asarray(value)
    if counts.ndim == 0:
        raise TypeError(f"Expected an array of numbers of {unit!r}; got {type(value).__name__}")
    if np.issubdtype(counts.dtype, np.integer):
        return IntTensor(counts.astype(np.int64))
    if np.issubdtype(counts.dtype, np.floating):
        if not np.isfinite(counts).all():
            raise ValueError("A TimeDeltaTensor needs finite counts")
        return FloatTensor(counts.astype(np.float64))
    raise TypeError(f"Expected numbers of {unit!r}; got {counts.dtype} counts")


def _as_boolean(equal):
    """``equal``, a NumPy result, as a ``bool`` or a ``BoolTensor``."""
    return BoolTensor(equal) if np.ndim(equal) else bool(equal)


def _from_pair(value):
    """A tuple ``(count, unit)`` as ``TimeDelta(count, unit)``, or as a
    :class:`TimeDeltaTensor` for an array of counts; any other value, such
    as a tuple of two dates, as itself."""
    if not (isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str)):
        return value
    count, unit = value
    if isinstance(count, str) or (np.ndim(count) > 0 and np.asarray(count).dtype.kind in "MOUS"):
        return value
    return _time_delta(count, unit)


def _as_time_delta(value) -> TimeDelta | TimeDeltaTensor | None:
    """``value`` as a :class:`TimeDelta` if it is a duration, such as a tuple
    ``(count, unit)``, or itself if it is a :class:`TimeDeltaTensor`, else
    ``None``."""
    value = _from_pair(value)
    if isinstance(value, TimeDelta | TimeDeltaTensor):
        return value
    if isinstance(value, (np.timedelta64, datetime.timedelta)) or hasattr(value, "to_timedelta64"):
        return TimeDelta(value)
    return None


def _as_array(values):
    """``values`` as an array. Dates with a timezone, as pandas holds them,
    become naive UTC ``datetime64`` at their resolution."""
    dtype = getattr(values, "dtype", None)
    if dtype is not None and getattr(dtype, "tz", None) is not None:
        return values.to_numpy(dtype=f"datetime64[{dtype.unit}]")
    return np.asarray(values)


def _duration(value, message) -> TimeDelta | float:
    """``value`` as a positive duration: a :class:`TimeDelta` for a duration,
    such as a ``timedelta64`` or ``pd.Timedelta``, kept at its resolution, or
    else a float. Raises ``ValueError`` with ``message`` if it is not
    positive or is ``NaT``, and for months and years, which have no fixed
    length."""
    if isinstance(value, np.timedelta64) and np.isnat(value):
        raise ValueError(message)
    duration = _as_time_delta(value)
    if isinstance(duration, TimeDelta):
        if duration.count <= 0:
            raise ValueError(message)
        return duration
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(message)
    return value


@overload
def _in_unit(duration: TimeDeltaTensor, unit: str | None) -> np.ndarray: ...


@overload
def _in_unit(duration: TimeDelta | float, unit: str | None) -> float: ...


@overload
def _in_unit(duration: _Duration | float, unit: str | None) -> float | np.ndarray: ...


def _in_unit(duration, unit):
    """A duration from :func:`_duration` as a number of ``unit``, a time unit;
    a :class:`TimeDelta` is converted by an exact factor and rounded once,
    and a :class:`TimeDeltaTensor` likewise to a float64 array. A number
    is taken only by ``unit`` ``None``, of a series without a unit, and is
    kept; ``unit`` ``None`` takes only numbers."""
    if not isinstance(duration, _Duration):
        if unit is not None:
            raise TypeError(f"This series has a unit, so this takes a duration, such as TimeDelta(1, {unit!r}), "
                            "not a plain number")
        return duration
    if unit is None:
        raise TypeError("A series without a unit takes numbers, not durations")
    if isinstance(duration, TimeDeltaTensor):
        # One unit is a whole multiple of the other, so one multiplication
        # or division by an integer rounds each count once.
        counts = np.asarray(duration.count, dtype=np.float64)
        source, target = _ATTOSECONDS[duration.unit], _ATTOSECONDS[unit]
        return counts * (source // target) if source >= target else counts / (target // source)
    return float(duration._length() / _ATTOSECONDS[unit])


# A single duration, in any of the forms the operators of TimeDelta and
# DateTime take, and a duration that may also be a tensor of them.
_ScalarDuration = TimeDelta | tuple[float, str] | np.timedelta64 | datetime.timedelta
Duration = _ScalarDuration | TimeDeltaTensor
