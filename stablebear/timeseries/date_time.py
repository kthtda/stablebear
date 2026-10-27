from __future__ import annotations

import datetime
import math

import numpy as np

from .. import _sb_cpp as cpp

# NumPy datetime64 resolutions with a fixed length, and the matching Times
# units. NumPy's "m" is minutes, which Times spells "min".
_NUMPY_TO_UNIT = {
    "as": "as", "fs": "fs", "ps": "ps", "ns": "ns", "us": "us", "ms": "ms",
    "s": "s", "m": "min", "h": "h", "D": "d",
}
_UNIT_TO_NUMPY = {unit: code for code, unit in _NUMPY_TO_UNIT.items()}


class DateTime:
    """A calendar date and time: a count of a time unit after
    1970-01-01T00:00, as a ``np.datetime64`` stores it. The epoch of
    :class:`Times`, the instant its time 0 stands for, is one.

    As in NumPy, dates follow the Gregorian calendar and every day has 86400
    seconds: leap seconds are not counted.

    Parameters
    ----------
    value : str, np.datetime64, datetime.datetime, datetime.date, or DateTime
        The instant, such as ``"2023-01-01T12:30"``. Its resolution is kept:
        ``"2023-01-01"`` counts days and a ``datetime`` counts microseconds.

    Raises
    ------
    TypeError
        If ``value`` is of another type.
    ValueError
        If ``value`` is ``NaT``, a timezone-aware ``datetime``, or has a
        resolution of weeks, months, or years, which have no fixed length.

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
            raise ValueError("A DateTime cannot have a timezone; convert it to naive UTC first")
        value = np.datetime64(value)
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


def _duration(value, message):
    """``value`` as a positive duration: a ``np.timedelta64`` for a
    ``timedelta`` or ``timedelta64``, kept as given, or else a float. Raises
    ``ValueError`` with ``message`` if it is not positive, and for months and
    years, which have no fixed length."""
    if isinstance(value, datetime.timedelta):
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
