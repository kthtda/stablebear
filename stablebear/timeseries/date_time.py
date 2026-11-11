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
        return self._equal(other, operator.eq)

    @overload
    def __ne__(self, other: DateTimeTensor | TimeDeltaTensor) -> BoolTensor: ...

    @overload
    def __ne__(self: _CountsTensor[Any], other: object) -> BoolTensor: ...

    @overload
    def __ne__(self, other: object) -> bool: ...

    def __ne__(self, other):  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._equal(other, operator.ne)

    def _comparable(self, other) -> Any:
        """``other`` as a value this one compares with, or ``None``."""
        raise NotImplementedError

    def _compare(self, other, compare) -> Any:
        """``compare``, an operator such as ``operator.lt``, of this and
        ``other`` as instants or lengths."""
        raise NotImplementedError

    def _equal(self, other, compare):
        """``compare``, ``==`` or ``!=``, of instants or lengths. A value that
        is not one, such as ``NaT``, is unequal instead of raising."""
        try:
            comparable = self._comparable(other)
        except (TypeError, ValueError):
            comparable = None
        return NotImplemented if comparable is None else self._compare(comparable, compare)

    def __hash__(self):
        # Equal instants or lengths hash the same: the exact length in
        # attoseconds hashes as an int when whole.
        return hash(Fraction(self._count) * _ATTOSECONDS[self._unit])


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
        counts, and unit; 1 hour and 60 minutes differ, unlike for ``==``."""
        return (isinstance(other, type(self)) and self._unit == other._unit
                and np.array_equal(np.asarray(self._count), np.asarray(other._count)))


class _Instants(_Counts[_CountT]):
    """The arithmetic that :class:`DateTime` and :class:`DateTimeTensor`
    share."""

    def _comparable(self, other):
        return other if isinstance(other, _Instants) else None

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
        other = self._comparable(other)
        if other is None:
            return NotImplemented
        return _as_boolean(_exact_compare(self._counts(), self._unit, other._counts(), other._unit, compare))

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
        return _date_time(_numpy_sum(self.to_datetime64(), other.to_timedelta64()))

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
            return _time_delta(_numpy_sum(self.to_datetime64(), other.to_datetime64(), subtract=True))
        other = _as_time_delta(other)
        if other is None:
            return NotImplemented
        return _date_time(_numpy_sum(self.to_datetime64(), other.to_timedelta64(), subtract=True))

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
    ``==``, ``<``, and the other comparisons compare instants, exactly at
    any two resolutions: ``DateTime("2023-01-01")`` equals
    ``DateTime(datetime.datetime(2023, 1, 1))``, and the two hash the same.
    Compare ``unit`` as well to tell them apart.

    Adding or subtracting a :class:`TimeDelta`, or a duration it accepts,
    gives a ``DateTime`` at the finer of the two resolutions; a duration with
    a float count is first rounded as ``TimeDelta.to_timedelta64()``
    rounds it. Subtracting a ``DateTime`` gives the :class:`TimeDelta`
    between them. With a :class:`TimeDeltaTensor` or a ``DateTimeTensor``,
    the result is a ``DateTimeTensor`` or a ``TimeDeltaTensor``. A count
    that does not fit in int64 at the finer resolution, such as that of a
    date before 1678 in nanoseconds, raises ``OverflowError``.
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
    as they do on tensors; ``==`` compares instants, as for a single date.
    ``array_equal()`` tells whether two are equal as a whole, with the same
    counts of the same unit.
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

    def _comparable(self, other):
        return _as_time_delta(other)

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
            counts = self._counts()
            factor = Fraction(_ATTOSECONDS[self._unit], _ATTOSECONDS[resolution])
            if self._integral:
                steps, exact = _integer_steps(counts.astype(np.int64), factor)
            else:
                steps, exact = _float_steps(counts.astype(np.float64), factor)
            # The few counts that the rounding above cannot take exactly, such
            # as ones that do not fit in int64, round as a single count does,
            # which raises for those that do not fit.
            flat_counts, flat_steps = counts.reshape(-1), steps.reshape(-1)
            for i in np.flatnonzero(~exact):
                flat_steps[i] = TimeDelta(flat_counts[i].item(), self._unit).to_timedelta64(resolution).astype(np.int64)
            return steps.astype(f"timedelta64[{_UNIT_TO_NUMPY[resolution]}]")
        steps = round(self._length() / _ATTOSECONDS[resolution])
        return np.timedelta64(steps, _UNIT_TO_NUMPY[resolution])

    def _length(self):
        """The duration in attoseconds, exactly."""
        return Fraction(self._count) * _ATTOSECONDS[self._unit]

    def _compare(self, other, compare):
        other = self._comparable(other)
        if other is None:
            return NotImplemented
        if not (self._array or other._array):
            return compare(self._length(), other._length())
        if self._integral and other._integral:
            return BoolTensor(_exact_compare(self._counts(), self._unit, other._counts(), other._unit, compare))
        unit = _finer(self._unit, other._unit)
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

    @property
    def _integral(self):
        """Whether the count is an integer or an ``IntTensor``."""
        return isinstance(self._count, (int, IntTensor))

    def _in(self, unit):
        """The count in ``unit``, a unit at least as fine, by an exact factor;
        integer counts of a tensor raise ``OverflowError`` if one leaves
        int64."""
        if isinstance(self._count, IntTensor):
            return _counts_in(self._counts(), self._unit, unit)
        return self._counts() * (_ATTOSECONDS[self._unit] // _ATTOSECONDS[unit])

    @overload
    def __add__(self: TimeDelta, other: _ScalarDuration) -> TimeDelta: ...

    @overload
    def __add__(self, other: Duration) -> TimeDeltaTensor: ...

    def _plus(self, other):
        """The sum with ``other``, a duration, in the finer of the two units."""
        unit = _finer(self._unit, other._unit)
        if (self._array or other._array) and self._integral and other._integral:
            # In int64, which a sum can leave; two integers add exactly.
            return _time_delta(_checked_sum(_counts_in(self._counts(), self._unit, unit),
                                            _counts_in(other._counts(), other._unit, unit)), unit)
        return _time_delta(self._in(unit) + other._in(unit), unit)

    def _negated(self):
        if isinstance(self._count, IntTensor):
            return _time_delta(_checked_negation(self._counts()), self._unit)
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
        if isinstance(self._count, IntTensor) and isinstance(factor, numbers.Integral) and factor:
            most = _INT64.max // abs(int(factor))
            counts = self._counts()
            if ((counts > most) | (counts < -most)).any():
                raise OverflowError("The result is too large to hold in int64")
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
    value : np.timedelta64, datetime.timedelta, pd.Timedelta, TimeDelta, int, or float
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
        If ``value`` is of another type, is a number without ``unit``, or is
        a duration with ``unit``.
    ValueError
        If ``value`` is ``NaT``, is not finite, or has a resolution of
        weeks, months, or years, or ``unit`` is unknown.

    Notes
    -----
    ``==``, ``<``, and the other comparisons compare lengths, exactly, with
    another ``TimeDelta`` or a duration it accepts: ``TimeDelta(1, "h")``
    equals ``TimeDelta(60, "min")`` and ``np.timedelta64(1, "h")``, and
    equal durations hash the same. Compare ``unit`` as well to tell 1 hour
    from 60 minutes.

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
            # NumPy counts a timedelta64 among the integers.
            if isinstance(value, np.timedelta64):
                raise TypeError("A timedelta64 has its own unit; pass it without unit")
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"Expected a number of {unit!r}; got {type(value).__name__}")
            if isinstance(value, numbers.Integral):
                self._count = int(value)
            else:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("A TimeDelta needs a finite count")
                self._count = int(value) if value.is_integer() else value
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
    value : IntTensor, FloatTensor, array_like, numpy.ndarray, pd.TimedeltaIndex, or sequence
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
        If ``value`` is of another type or is not an array, holds numbers
        without ``unit``, or holds durations with ``unit``.
    ValueError
        If ``value`` holds ``NaT`` or counts that are not finite, or has a
        resolution of weeks, months, or years, or ``unit`` is unknown.

    Notes
    -----
    Arithmetic and comparisons work elementwise, with a single duration
    broadcast against the tensor, as :class:`TimeDelta` does them. ``<``,
    ``<=``, ``>``, ``>=``, ``==``, and ``!=`` give a ``BoolTensor``, as they
    do on tensors; ``==`` compares lengths, as for a single duration.
    ``array_equal()`` tells whether two are equal as a whole, with the same
    counts of the same unit. Integer counts that do not fit in
    int64 in the finer unit of an operation raise ``OverflowError``.
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


_INT64 = np.iinfo(np.int64)


def _finer(unit, other):
    """The shorter of two time units."""
    return min(unit, other, key=_ATTOSECONDS.__getitem__)


def _coarser(unit, other):
    """The longer of two time units."""
    return max(unit, other, key=_ATTOSECONDS.__getitem__)


def _counts_in(counts, unit, finer):
    """``counts``, integer counts of ``unit`` or an array of them, as int64
    counts of ``finer``, a unit at most as long; raises ``OverflowError`` if
    one does not fit."""
    counts = np.asarray(counts, dtype=np.int64)
    factor = _ATTOSECONDS[unit] // _ATTOSECONDS[finer]
    if factor == 1:
        return counts
    most = _INT64.max // factor
    if ((counts > most) | (counts < -most)).any():
        raise OverflowError(f"A count of {unit} is too large to hold in {finer}, "
                            "the finer resolution of the operation")
    # A factor beyond int64 leaves only counts of 0.
    return counts * factor if most else np.zeros_like(counts)


def _divide_half_even(numerators, denominators):
    """``numerators / denominators``, int64 arrays with positive
    denominators, rounded to the nearest integer, and at ties to the even
    one."""
    quotients, remainders = np.divmod(numerators, denominators)
    # The remainder is in [0, denominator), so the quotient rounds up when
    # the remainder is more than what is left of the denominator.
    left = denominators - remainders
    return quotients + ((remainders > left) | ((remainders == left) & (quotients % 2 == 1)))


def _integer_steps(counts, factor):
    """``counts``, an int64 array, times ``factor``, rounded to the nearest
    integer and at ties to the even one, and a mask of the counts whose
    step is exact. ``factor`` is the ratio of two time units, and so a whole
    number or the inverse of one."""
    if factor.denominator == 1:
        most = _INT64.max // factor.numerator
        exact = (counts >= -most) & (counts <= most)
        # A factor beyond int64 leaves only counts of 0.
        return (np.where(exact, counts, 0) * factor.numerator if most else np.zeros_like(counts)), exact
    divisor = factor.denominator
    if divisor > _INT64.max:
        # Each count of less than half the divisor rounds to 0.
        most = min((divisor - 1) // 2, int(_INT64.max))
        return np.zeros_like(counts), (counts >= -most) & (counts <= most)
    return _divide_half_even(counts, np.int64(divisor)), np.ones(counts.shape, dtype=bool)


def _split(a):
    """``a`` as the sum of its high 26 bits and the rest (Veltkamp)."""
    scaled = 134217729.0 * a  # 2**27 + 1
    high = scaled - (scaled - a)
    return high, a - high


def _two_product(a, b):
    """The product ``a * b`` rounded, and its rounding error, so that the two
    sum to the product exactly (Dekker)."""
    product = a * b
    a_high, a_low = _split(a)
    b_high, b_low = _split(np.float64(b))
    error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low
    return product, error


def _float_steps(counts, factor):
    """``counts``, a float64 array, times ``factor``, rounded to the nearest
    integer and at ties to the even one, and a mask of the counts whose
    step is exact. ``factor`` is the ratio of two time units, and so a whole
    number or the inverse of one. ``np.rint(counts * factor)`` is not exact,
    since the product rounds before ``rint`` does."""
    finite = np.isfinite(counts)
    counts = np.where(finite, counts, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        if factor.denominator == 1:
            steps, exact = _multiplied_float_steps(counts, factor.numerator)
        else:
            steps, exact = _divided_float_steps(counts, factor.denominator)
    return steps, exact & finite


def _multiplied_float_steps(counts, multiplier):
    """The steps of :func:`_float_steps` for a whole factor."""
    if float(multiplier) != multiplier:
        return np.zeros(counts.shape, dtype=np.int64), np.zeros(counts.shape, dtype=bool)
    product, error = _two_product(counts, multiplier)
    # The exact product is product + error, and the error is at most half a
    # step of the rounded product. Below 2**52, where the rounded product
    # has steps of at most a half, the exact product rounds as the rounded
    # one does, except when the rounded one is at a tie, where the error
    # decides.
    floor = np.floor(product)
    tie = (product - floor == 0.5) & (error != 0)
    steps = np.where(tie, floor + (error > 0), np.rint(product))
    # From 2**52, a rounded product is whole, and exact without an error.
    exact = (np.abs(product) < 2.0**52) | ((error == 0) & (np.abs(product) < 2.0**63))
    return np.where(exact, steps, 0).astype(np.int64), exact


def _divided_float_steps(counts, divisor):
    """The steps of :func:`_float_steps` for the inverse of a whole factor."""
    mantissas, exponents = np.frexp(counts)
    # Each count is numerators / 2**shifts, with |numerators| < 2**53.
    numerators = (mantissas * 2.0**53).astype(np.int64)
    shifts = 53 - exponents
    # A count with no shift is whole, and divides as an integer.
    whole = shifts <= 0
    fits = np.abs(counts) < 2.0**63
    steps, exact = _integer_steps(np.where(whole & fits, counts, 0).astype(np.int64), Fraction(1, divisor))
    exact &= fits | ~whole
    # Another divides its numerator by divisor * 2**shifts while that fits in
    # int64. A larger one is at least 2**62, and the quotient rounds to 0.
    room = 62 - divisor.bit_length()
    near = ~whole & (shifts <= room)
    if near.any():
        denominators = np.left_shift(np.int64(divisor), np.where(near, shifts, 0))
        steps = np.where(near, _divide_half_even(numerators, denominators), steps)
    return np.where(~whole & ~near, 0, steps), exact


def _checked_sum(a, b):
    """``a + b``, int64 counts or arrays of them; raises ``OverflowError`` if a
    sum leaves int64. Its least value, which NumPy reads as ``NaT``, counts
    as leaving it."""
    with np.errstate(over="ignore"):
        total = np.add(a, b)
    # A sum overflows when it has the sign of neither operand.
    if (((a ^ total) & (b ^ total)) < 0).any() or (total == _INT64.min).any():
        raise OverflowError("The result is too large to hold in int64")
    return total


def _checked_negation(counts):
    """``-counts``, int64 counts or an array of them; raises ``OverflowError``
    for the least int64, which has no negation."""
    if (counts == _INT64.min).any():
        raise OverflowError("The result is too large to hold in int64")
    return -counts


def _numpy_sum(a, b, subtract=False):
    """``a + b``, or ``a - b`` with ``subtract``, of ``datetime64`` or
    ``timedelta64`` values or arrays, at the finer of their resolutions;
    raises ``OverflowError`` if an operand or the result does not fit in
    int64 there, where NumPy would wrap."""
    a, b = np.asarray(a), np.asarray(b)
    unit_a = _NUMPY_TO_UNIT[np.datetime_data(a.dtype)[0]]
    unit_b = _NUMPY_TO_UNIT[np.datetime_data(b.dtype)[0]]
    unit = _finer(unit_a, unit_b)
    counts_b = _counts_in(b.astype(np.int64), unit_b, unit)
    total = _checked_sum(_counts_in(a.astype(np.int64), unit_a, unit),
                         _checked_negation(counts_b) if subtract else counts_b)
    # A date plus or minus a duration is a date; anything else a duration.
    kind = "datetime64" if a.dtype.kind == "M" and b.dtype.kind == "m" else "timedelta64"
    return total.astype(f"{kind}[{_UNIT_TO_NUMPY[unit]}]")[()]


def _exact_compare(counts, unit, other_counts, other_unit, compare):
    """``compare(counts, other_counts)``, an operator such as ``operator.lt``,
    for integer counts of two units, or arrays of them, as the lengths they
    stand for, without converting either to the finer unit, where it can
    overflow: the finer is split into whole units of the coarser and a
    remainder."""
    counts, other_counts = np.asarray(counts, dtype=np.int64), np.asarray(other_counts, dtype=np.int64)
    if unit == other_unit:
        return compare(counts, other_counts)
    fine_first = _ATTOSECONDS[unit] < _ATTOSECONDS[other_unit]
    fine, coarse = (counts, other_counts) if fine_first else (other_counts, counts)
    # The number of the finer unit in one of the coarser.
    ratio = _ATTOSECONDS[_coarser(unit, other_unit)] // _ATTOSECONDS[_finer(unit, other_unit)]
    if ratio <= _INT64.max:
        whole, rest = np.divmod(fine, ratio)
    else:
        # Every count is below the ratio: a negative one is -1 whole units
        # and a positive remainder.
        whole, rest = np.where(fine < 0, -1, 0), np.where(fine < 0, 1, fine)
    # The sign of fine - coarse, from the whole units and then the
    # remainder, which is never negative.
    sign = (whole > coarse).astype(np.int64) - (whole < coarse) + ((whole == coarse) & (rest > 0))
    return compare(sign if fine_first else -sign, 0)


def _array_counts(value, unit):
    """``value``, an array-like of numbers of ``unit``, as an ``IntTensor`` of
    integers or a ``FloatTensor`` of floats."""
    counts = np.asarray(value)
    if counts.ndim == 0:
        raise TypeError(f"Expected an array of numbers of {unit!r}; got {type(value).__name__}")
    # NumPy counts timedelta64 among the integers.
    if counts.dtype.kind == "m":
        raise TypeError("Durations have their own unit; pass them without unit")
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
