from __future__ import annotations

import datetime
import math
import numbers
from fractions import Fraction

import numpy as np

from .. import _sb_cpp as cpp

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


class DateTime:
    """A calendar date and time: a count of a time unit after
    1970-01-01T00:00, as a ``np.datetime64`` stores it. The epoch of
    :class:`Times`, the instant its time 0 stands for, is one.

    Dates follow the Gregorian calendar and every day has 86400
    seconds: leap seconds are not counted.

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
    :meth:`to_datetime64` values to compare instants.
    """

    def __init__(self, value):
        if isinstance(value, DateTime):
            self._data = value._data
            return
        if not isinstance(value, (str, datetime.date, np.datetime64)):
            raise TypeError(f"Expected a date, datetime, datetime64, or string; got {type(value).__name__}")
        if isinstance(value, datetime.datetime) and value.tzinfo is not None:
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        # A pd.Timestamp converts at its own resolution, such as nanoseconds.
        to_datetime64 = getattr(value, "to_datetime64", None)
        value = np.datetime64(value) if to_datetime64 is None else to_datetime64()
        if np.isnat(value):
            raise ValueError("A DateTime cannot be NaT")
        code, step = np.datetime_data(value.dtype)
        if code not in _NUMPY_TO_UNIT:
            raise ValueError(f"A DateTime needs a resolution of fixed length, not {code!r}")
        self._data = cpp.DateTime(int(value.astype(np.int64)) * step, _NUMPY_TO_UNIT[code])

    @classmethod
    def _from_cpp(cls, data):
        date_time = cls.__new__(cls)
        date_time._data = data
        return date_time

    @property
    def count(self) -> int:
        """The number of :attr:`unit` after 1970-01-01T00:00."""
        return self._data.count

    @property
    def unit(self) -> str:
        """The resolution, as a ``Times`` unit such as ``"d"`` or ``"us"``."""
        return self._data.unit

    def to_datetime64(self) -> np.datetime64:
        """The value as a ``np.datetime64`` at its resolution."""
        return np.datetime64(self.count, _UNIT_TO_NUMPY[self.unit])

    def __eq__(self, other):
        if not isinstance(other, DateTime):
            return NotImplemented
        return self._data == other._data

    def __reduce__(self):
        return DateTime, (self.to_datetime64(),)

    def __hash__(self):
        return hash((self.count, self.unit))

    def __repr__(self):
        return f"DateTime({str(self.to_datetime64())!r})"


class TimeDelta:
    """A duration: a count of a time unit, such as 1.5 hours.

    The count is an integer, as a ``np.timedelta64`` stores it, or a float,
    as a time series stores its times. Every day has 86400 seconds; weeks,
    months, and years, which have no fixed length, are not units.

    Parameters
    ----------
    value : np.timedelta64, datetime.timedelta, pd.Timedelta, TimeDelta, or number
        The duration, such as ``np.timedelta64(3, "h")``. Its resolution is
        kept: a ``timedelta`` counts microseconds, and a ``pd.Timedelta``
        counts its own unit, such as nanoseconds. With ``unit``, a number of
        ``unit``, as in ``TimeDelta(1.5, "h")``; a float that is a whole
        number becomes an integer.
    unit : str, optional
        The unit of a number ``value``: one of ``"as"``, ``"fs"``, ``"ps"``,
        ``"ns"``, ``"us"``, ``"ms"``, ``"s"``, ``"min"``, ``"h"``, and
        ``"d"``.

    Raises
    ------
    TypeError
        If ``value`` is of another type, or is a number without ``unit``.
    ValueError
        If ``value`` is ``NaT``, not finite, or has a resolution of weeks,
        months, or years, or ``unit`` is unknown.

    Notes
    -----
    Two values are equal when they have the same count of the same unit; the
    same length in another unit is a different value. Compare
    :meth:`to_timedelta64` values to compare lengths.
    """

    def __init__(self, value, unit: str | None = None):
        if unit is not None:
            if unit not in _UNIT_TO_NUMPY:
                raise ValueError(f"Unknown unit {unit!r}; expected one of {', '.join(_UNIT_TO_NUMPY)}")
            if not isinstance(value, numbers.Real):
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
        if isinstance(value, TimeDelta):
            self._count, self._unit = value._count, value._unit
            return
        if hasattr(value, "to_timedelta64"):
            # A pd.Timedelta, at its own resolution.
            value = value.to_timedelta64()
        elif isinstance(value, datetime.timedelta):
            value = np.timedelta64(value)
        elif not isinstance(value, np.timedelta64):
            raise TypeError(f"Expected a timedelta, timedelta64, or TimeDelta; got {type(value).__name__}")
        if np.isnat(value):
            raise ValueError("A TimeDelta cannot be NaT")
        code, step = np.datetime_data(value.dtype)
        if code not in _NUMPY_TO_UNIT:
            raise ValueError(f"A TimeDelta needs a unit of fixed length, not {code!r}")
        self._count = int(value.astype(np.int64)) * step
        self._unit = _NUMPY_TO_UNIT[code]

    @property
    def count(self) -> int | float:
        """The number of :attr:`unit` in the duration."""
        return self._count

    @property
    def unit(self) -> str:
        """The resolution, as a ``Times`` unit such as ``"h"`` or ``"us"``."""
        return self._unit

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
            fine = isinstance(self._count, int) or _ATTOSECONDS[self._unit] < _ATTOSECONDS["us"]
            resolution = self._unit if fine else "us"
        if resolution not in _ATTOSECONDS:
            raise ValueError(f"Unknown resolution {resolution!r}; expected one of {', '.join(_ATTOSECONDS)}")
        steps = round(Fraction(self._count) * _ATTOSECONDS[self._unit] / _ATTOSECONDS[resolution])
        return np.timedelta64(steps, _UNIT_TO_NUMPY[resolution])

    def __eq__(self, other):
        if not isinstance(other, TimeDelta):
            return NotImplemented
        return (self._count, self._unit) == (other._count, other._unit)

    def __reduce__(self):
        return TimeDelta, (self._count, self._unit)

    def __hash__(self):
        return hash((self._count, self._unit))

    def __repr__(self):
        return f"TimeDelta({self._count}, {self._unit!r})"


def _as_array(values):
    """``values`` as an array. Dates with a timezone, as pandas holds them,
    become naive UTC ``datetime64`` at their resolution."""
    dtype = getattr(values, "dtype", None)
    if getattr(dtype, "tz", None) is not None:
        return values.to_numpy(dtype=f"datetime64[{dtype.unit}]")
    return np.asarray(values)


def _duration(value, message):
    """``value`` as a positive duration: a ``np.timedelta64`` for a
    ``timedelta``, ``pd.Timedelta``, or ``timedelta64``, kept as given, or
    else a float. Raises ``ValueError`` with ``message`` if it is not
    positive, and for months and years, which have no fixed length."""
    if hasattr(value, "to_timedelta64"):
        # A pd.Timedelta, at its own resolution, such as nanoseconds.
        value = value.to_timedelta64()
    elif isinstance(value, datetime.timedelta):
        value = np.timedelta64(value)
    if isinstance(value, np.timedelta64):
        if np.isnat(value):
            raise ValueError(message)
        code, _ = np.datetime_data(value.dtype)
        if code not in _NUMPY_TO_UNIT:
            raise ValueError(f"A duration needs a unit of fixed length, not {code!r}")
        if value <= np.timedelta64(0, code):
            raise ValueError(message)
        return value
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(message)
    return value


def _in_unit(duration, unit):
    """A duration from :func:`_duration` as a number of ``unit``, a Times unit;
    a ``timedelta64`` is divided by ``unit`` once."""
    if isinstance(duration, np.timedelta64):
        return float(duration / np.timedelta64(1, _UNIT_TO_NUMPY[unit]))
    return duration
