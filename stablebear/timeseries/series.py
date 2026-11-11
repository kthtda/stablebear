from __future__ import annotations

import datetime
import operator
import sys
from typing import Any, Generic, NamedTuple, NoReturn, Protocol, TypeVar

import numpy as np

from .. import _sb_cpp as cpp
from .._binary_io import _BinaryIoMixin
from .._tensor_base import IndexedElementTensor, Tensor, _infer_shape_and_flatten, _tensor_from_nested
from ..base_tensor import FloatTensor, IntTensor
from ..distance_matrix import DistanceMatrix
from ..functional.pcf import Pcf
from ..nested_tensor import NestedTensor, _leaf_wrappers
from ..point_cloud import PointCloud
from ..symmetric_matrix import SymmetricMatrix
from .date_time import _ATTOSECONDS, _NUMPY_TO_UNIT, _UNIT_TO_NUMPY, DateTime, DateTimeTensor, TimeDelta, TimeDeltaTensor, _as_array, _as_time_delta, _from_pair, _in_unit, _unit_name
from ..typing import (
    _NATIVE_SUFFIX,
    barcode32,
    barcode64,
    boolean,
    distmat32,
    distmat64,
    float32,
    float64,
    int32,
    int64,
    pcf32,
    pcf32i,
    pcf64,
    pcf64i,
    pcloud32,
    pcloud64,
    ts,
    symmat32,
    symmat64,
    uint32,
    uint64,
)

T = TypeVar("T")


def _checked_dates(dates):
    """``dates``, a ``datetime64`` array, with no ``NaT`` and a resolution of
    fixed length."""
    if np.isnat(dates).any():
        raise ValueError("Dates and times cannot be NaT")
    code, _ = np.datetime_data(dates.dtype)
    if code not in _NUMPY_TO_UNIT:
        raise ValueError(f"Dates and times need a resolution of fixed length, not {code!r}")
    return dates


def _is_date(t):
    """Whether ``t`` is a date and time that :class:`DateTime` accepts."""
    return isinstance(t, (str, datetime.date, np.datetime64, DateTime))


def _dates_in_sequence(t):
    """``t``, a list or tuple, as a :class:`DateTimeTensor` if it holds
    values that :class:`DateTime` accepts, else ``None``. Raises
    ``TypeError`` if it mixes them with other values, or pairs dates with a
    unit."""
    if isinstance(t, tuple) and len(t) == 2 and np.ndim(t[0]) > 0 and isinstance(t[1], str):
        raise TypeError("Dates and times have no unit; pass them alone")
    if not any(_is_date(element) for element in t):
        return None
    if not all(_is_date(element) for element in t):
        raise TypeError("Times must be all durations or all dates and times, not a mix")
    return DateTimeTensor(t)


def _dates_of(t):
    """``t`` as a checked ``datetime64`` array if it is dates and times:
    a :class:`DateTime` or a value it accepts, a :class:`DateTimeTensor`, or
    an array, sequence, or pandas object of them; else ``None``."""
    if _is_date(t):
        return np.asarray(DateTime(t).to_datetime64())
    if isinstance(t, DateTimeTensor):
        return t.to_datetime64()
    if isinstance(t, (Tensor, int, float, np.integer, np.floating)) or _as_time_delta(t) is not None:
        return None
    if isinstance(t, (list, tuple)) and (dates := _dates_in_sequence(t)) is not None:
        return dates.to_datetime64()
    values = _as_array(t)
    if values.dtype.kind not in "MOUS" or _finest_duration_unit(values) is not None:
        return None
    try:
        dates = np.asarray(values, dtype="datetime64")
    except (TypeError, ValueError):
        raise TypeError("Times must be all durations or all dates and times, not a mix") from None
    return _checked_dates(dates)


def _date_counts(dates):
    """``dates``, a checked ``datetime64`` array, as an ``IntTensor`` of
    counts and their unit."""
    code, _ = np.datetime_data(dates.dtype)
    # Counts of the unit itself, such as minutes for datetime64[15m].
    return IntTensor(dates.astype(f"datetime64[{code}]").astype(np.int64)), _NUMPY_TO_UNIT[code]


def _finest_duration_unit(values):
    """The resolution of ``values``, an array, if it holds durations: that of
    a ``timedelta64`` array, or the finest unit of other durations; else
    ``None``."""
    if np.issubdtype(values.dtype, np.timedelta64):
        # A resolution without a fixed length is rejected by _durations_in.
        return _NUMPY_TO_UNIT.get(np.datetime_data(values.dtype)[0], "s")
    if values.dtype == object and values.size > 0:
        durations = [duration for value in values.flat if (duration := _as_time_delta(value)) is not None]
        if len(durations) == values.size:
            return min((duration.unit for duration in durations), key=_ATTOSECONDS.__getitem__)
    return None


def _durations_in(values, unit):
    """``values``, an array, as float64 numbers of ``unit``, a time unit, if
    it holds durations, else ``None``. Each is converted by an exact factor
    and rounded once."""
    if np.issubdtype(values.dtype, np.timedelta64):
        if np.isnat(values).any():
            raise ValueError("Durations cannot be NaT")
        code, _ = np.datetime_data(values.dtype)
        if code not in _NUMPY_TO_UNIT:
            raise ValueError(f"Durations need a resolution of fixed length, not {code!r}")
        # Both in integers of the finer resolution, divided once.
        return values / np.timedelta64(1, _UNIT_TO_NUMPY[unit])
    if values.dtype == object and values.size > 0:
        durations = [duration for value in values.flat if (duration := _as_time_delta(value)) is not None]
        if len(durations) == values.size:
            return np.array([_in_unit(duration, unit) for duration in durations]).reshape(values.shape)
    return None


# Counts below this in the finest unit of a date, its epoch, and the series
# have a difference that fits in int64.
_FITS_IN_FINE = 2**62 - 1
_INT64 = np.iinfo(np.int64)


def _units_per(coarse, fine):
    """The number of ``fine`` in one ``coarse``, time units, or ``None`` if
    it exceeds int64."""
    ratio = _ATTOSECONDS[coarse] // _ATTOSECONDS[fine]
    return ratio if ratio <= _INT64.max else None


def _fits_in_fine(counts, factor):
    """Whether ``counts``, of a unit with ``factor`` of a finer one each, are
    below ``_FITS_IN_FINE`` in the finer unit."""
    if factor is None:
        return np.zeros(np.shape(counts), dtype=bool)
    most = _FITS_IN_FINE // factor
    return (counts <= most) & (counts >= -most)


def _converted(values, source, target):
    """``values``, float64 numbers of the time unit ``source``, in
    ``target``, multiplied or divided by one integer factor."""
    if source == target:
        return values
    source_length, target_length = _ATTOSECONDS[source], _ATTOSECONDS[target]
    if source_length > target_length:
        return values * float(source_length // target_length)
    return values / float(target_length // source_length)


def _offsets_in(dates, epoch, unit):
    """``dates - epoch`` as float64 numbers of ``unit``, a time unit, for
    ``dates`` a checked ``datetime64`` array and ``epoch`` a
    :class:`DateTime`, as ``time_since`` in C++ computes each, so that a
    series and a tensor of it agree. When both counts fit, the difference is
    taken in int64 in the finest of the three units and converted to
    ``unit``. Otherwise :func:`_split_offsets` gives it, so that no
    resolution or span overflows."""
    code, step = np.datetime_data(dates.dtype)
    if step != 1:
        # Counts of the unit itself, such as minutes for datetime64[15m].
        dates = dates.astype(f"datetime64[{code}]")
    date_unit = _NUMPY_TO_UNIT[code]
    counts = dates.astype(np.int64)
    fine = min(date_unit, epoch.unit, unit, key=_ATTOSECONDS.__getitem__)
    date_factor, epoch_factor = _units_per(date_unit, fine), _units_per(epoch.unit, fine)
    fast = _fits_in_fine(counts, date_factor) & _fits_in_fine(epoch.count, epoch_factor)
    offsets = np.empty(counts.shape)
    if fast.any():
        assert date_factor is not None and epoch_factor is not None
        differences = counts[fast] * date_factor - epoch.count * epoch_factor
        offsets[fast] = _converted(differences.astype(np.float64), fine, unit)
    if not fast.all():
        offsets[~fast] = _split_offsets(counts[~fast], date_unit, epoch, unit)
    return offsets


def _whole_days(counts, unit):
    """``counts`` of the time unit ``unit`` as whole days and a remainder in
    ``unit``, truncated as C++ divides."""
    ratio = _units_per("d", unit)
    if ratio is None:
        # Beyond int64, so more than any count, which is all remainder.
        return np.zeros_like(counts), counts
    rest = np.fmod(counts, ratio)
    return (counts - rest) // ratio, rest


def _split_offsets(counts, date_unit, epoch, unit):
    """``dates - epoch`` as :func:`_offsets_in` gives it, for ``counts`` of
    dates in ``date_unit`` too far from ``epoch`` to subtract in int64: each
    count is split into whole days and a remainder in its own unit, and the
    difference of the days and that of the remainders, taken in integers
    when a day of the finer unit fits, are converted to ``unit`` and
    added."""
    date_days, date_rest = _whole_days(counts, date_unit)
    epoch_days, epoch_rest = _whole_days(np.int64(epoch.count), epoch.unit)
    days = date_days - epoch_days
    # A difference overflows when its operands differ in sign and it does
    # not have the sign of the first.
    if (((date_days ^ epoch_days) & (date_days ^ days)) < 0).any():
        raise OverflowError("A date is too many days from the epoch to count")
    day_difference = _converted(days.astype(np.float64), "d", unit)
    # The remainders, below a day each, subtract in int64 in the finer of
    # their units when a day of it is below 2**62, as down to picoseconds.
    finer = min(date_unit, epoch.unit, key=_ATTOSECONDS.__getitem__)
    finer_per_day = _units_per("d", finer)
    if finer_per_day is not None and finer_per_day <= _FITS_IN_FINE:
        # Both factors are at most finer_per_day, so they fit too.
        date_factor = _ATTOSECONDS[date_unit] // _ATTOSECONDS[finer]
        epoch_factor = _ATTOSECONDS[epoch.unit] // _ATTOSECONDS[finer]
        rest_difference = date_rest * date_factor - epoch_rest * epoch_factor
        return day_difference + _converted(rest_difference.astype(np.float64), finer, unit)
    return (day_difference + _converted(date_rest.astype(np.float64), date_unit, unit)
            - _converted(np.float64(epoch_rest), epoch.unit, unit))


def _times_from_datetimes(stamps, unit, epoch):
    """The times of ``datetime64`` stamps counted from ``epoch``, or from the
    first stamp, in ``unit``, as :func:`_native_times` gives them."""
    stamps = _checked_dates(stamps)
    if epoch is None:
        if stamps.size == 0:
            raise ValueError("Pass an epoch for an empty array of dates and times")
        epoch = stamps.reshape(-1)[0]
    epoch = DateTime(epoch)
    unit = _unit_name(unit)
    return _offsets_in(stamps, epoch, unit), unit, epoch._data


def _times_in(counts, unit):
    """``counts``, a float or an array of them, as durations in ``unit``,
    or as numbers for a series without a unit, whose ``unit`` is ``None``."""
    if np.ndim(counts) == 0:
        return float(counts) if unit is None else TimeDelta(float(counts), unit)
    counts = FloatTensor(np.asarray(counts, dtype=np.float64))
    return counts if unit is None else TimeDeltaTensor(counts, unit)


def _end_of_domain(t):
    """``t``, an end of a domain, as a :class:`TimeDelta` if it is a
    duration, else as a float."""
    duration = _as_time_delta(t)
    if isinstance(duration, TimeDelta):
        return duration
    if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, float, np.integer, np.floating)):
        raise TypeError(f"The ends of a domain are durations or numbers, not {type(t).__name__}")
    return float(t)


def _numbers(t):
    """``t`` as a float or a ``FloatTensor`` if it holds plain numbers, else
    ``None``."""
    if isinstance(t, (FloatTensor, IntTensor)):
        return FloatTensor(np.asarray(t, dtype=np.float64))
    # A DateTimeTensor is checked before np.asarray, which would convert it
    # one element at a time, and read an empty one as numbers.
    if isinstance(t, (bool, np.bool_, Tensor, str, DateTimeTensor)) or _as_time_delta(t) is not None:
        return None
    if isinstance(t, (int, float, np.integer, np.floating)):
        return float(t)
    array = _as_array(t)
    if array.dtype.kind not in "iuf":
        return None
    return FloatTensor(np.asarray(array, dtype=np.float64))


def _with_unit(t, unit):
    """Numbers ``t`` read as counts of ``unit``, as durations."""
    numbers = _numbers(t)
    if numbers is None:
        raise TypeError(f"Durations and dates have no unit; pass unit only with numbers, as in series(90, {unit!r})")
    return _times_in(numbers, unit)


def _native_times(times, unit=None, epoch=None):
    """The times of ``times``, as the constructor of :class:`TimeSeries`
    accepts them, in ``unit`` and counted from ``epoch``, as a tuple of the
    timestamps, a float64 array, the unit, and the native epoch or ``None``.
    A plain array is in ``unit``, by default none."""
    times = _from_pair(times)
    if isinstance(times, (list, tuple)) and (dates := _dates_in_sequence(times)) is not None:
        times = dates
    if isinstance(times, TimeDeltaTensor):
        # Counts in one unit, by default theirs, converted by one exact
        # factor; the short name of unit, which may be an alias.
        unit = unit or times.unit
        times = _in_unit(times, _unit_name(unit))
    elif isinstance(times, DateTimeTensor):
        times = times.to_datetime64()
    times = _as_array(times)
    if np.issubdtype(times.dtype, np.datetime64) or np.issubdtype(times.dtype, np.timedelta64):
        code = np.datetime_data(times.dtype)[0]
        # The resolution of the dates or durations; one without a fixed
        # length is rejected when they are converted.
        unit = unit or _NUMPY_TO_UNIT.get(code, "s")
        if np.issubdtype(times.dtype, np.datetime64):
            return _times_from_datetimes(times, unit, epoch)
        # Durations as dates after 1970-01-01, so that they are converted
        # to unit in integers, as dates are.
        origin = np.datetime64(0, code)  # pyright: ignore[reportCallIssue, reportArgumentType]
        values, unit, _ = _times_from_datetimes(origin + times, unit, origin)
        return values, unit, None if epoch is None else DateTime(epoch)._data
    finest = _finest_duration_unit(times)
    if finest is not None:
        # TimeDelta and other durations, by default in the finest of
        # their units, each converted by an exact factor; the short name
        # of unit, which may be an alias.
        unit = unit or finest
        times = _durations_in(times, _unit_name(unit))
    return np.asarray(times, dtype=np.float64), unit, None if epoch is None else DateTime(epoch)._data


class _PandasLike(Protocol):
    """A pandas ``Series`` or ``DataFrame``, described by the members they
    share so that type hints need no pandas, which is optional."""

    @property
    def index(self) -> Any: ...

    def to_numpy(self) -> np.ndarray: ...


def _as_series(obj: TimeSeries | _PandasLike) -> TimeSeries:
    """``obj``, a :class:`TimeSeries`, or a pandas ``Series`` or
    ``DataFrame`` converted with :meth:`TimeSeries.from_pandas`."""
    if isinstance(obj, TimeSeries):
        return obj
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(obj, (pd.Series, pd.DataFrame)):
        return TimeSeries.from_pandas(obj)
    raise TypeError(f"Expected a TimeSeries or a pandas Series or DataFrame, not {type(obj).__name__}; "
                    "build a TimeSeries from arrays with TimeSeries(values, times)")


# Element dtypes of observations that convert to pandas.
_SCALAR_DTYPES = (boolean, float32, float64, int32, int64, uint32, uint64)

# Element dtype -> native series class. The tensor class wrapping the values
# of each dtype comes from _leaf_wrappers, which loads persistence lazily, so
# this module does not import stablebear.persistence and persistence can
# import TimeSeries.
_SERIES_NATIVE = {
    boolean: cpp.TimeSeriesBool,
    float32: cpp.TimeSeriesFloat32,
    float64: cpp.TimeSeriesFloat64,
    int32: cpp.TimeSeriesInt32,
    int64: cpp.TimeSeriesInt64,
    uint32: cpp.TimeSeriesUint32,
    uint64: cpp.TimeSeriesUint64,
    pcf32: cpp.TimeSeriesPcf32,
    pcf64: cpp.TimeSeriesPcf64,
    pcf32i: cpp.TimeSeriesPcf32i,
    pcf64i: cpp.TimeSeriesPcf64i,
    pcloud32: cpp.TimeSeriesPointCloud32,
    pcloud64: cpp.TimeSeriesPointCloud64,
    distmat32: cpp.TimeSeriesDistanceMatrix32,
    distmat64: cpp.TimeSeriesDistanceMatrix64,
    symmat32: cpp.TimeSeriesSymmetricMatrix32,
    symmat64: cpp.TimeSeriesSymmetricMatrix64,
    barcode32: cpp.TimeSeriesBarcode32,
    barcode64: cpp.TimeSeriesBarcode64,
}
_SERIES_CPP_TO_DTYPE = {cls: dtype for dtype, cls in _SERIES_NATIVE.items()}
_SERIES_TENSOR_TO_DTYPE = {
    getattr(cpp, f"TimeSeries{suffix}Tensor"): ts(element)
    for element, suffix in _NATIVE_SUFFIX.items()
}


def _is_stablebear_object(value):
    from ..persistence.barcode import Barcode

    return isinstance(
        value, (Barcode, DistanceMatrix, Pcf, PointCloud, SymmetricMatrix, Tensor)
    )


def _on_one_epoch(parts, unit):
    """``parts``, one-channel series, counted from the earliest of their
    epochs if they all have one. A part that moves is first converted to
    ``unit``, if given, so that it is shifted at the resolution of the
    result: shifting by centuries in a fine unit, such as nanoseconds,
    would round its times."""
    epochs = [part.epoch for part in parts]
    # One instant at two resolutions is two epochs: the channels of a series
    # share one.
    stored = [None if epoch is None else (epoch.count, epoch.unit) for epoch in epochs]
    if None in stored or len(set(stored)) == 1:
        return parts
    earliest = min(epochs)
    return [part if epoch == (earliest.count, earliest.unit)
            else _shifted_to(part if unit is None else part.to(unit), earliest)
            for part, epoch in zip(parts, stored)]


def _shifted_to(part, epoch):
    """``part``, a one-channel series, counted from ``epoch``, an earlier
    :class:`DateTime`."""
    offset = float(_offsets_in(np.asarray(part.epoch.to_datetime64()), epoch, part.unit))
    return TimeSeries._from_cpp(part._data.shifted(offset, epoch._data))


def _values_tensor(values):
    """Convert values to a tensor whose axis 0 is time."""
    from ..tensor_create import tensor

    if isinstance(values, (list, tuple)) and any(
        _is_stablebear_object(value) for value in values
    ):
        # np.asarray would silently turn equal-size objects into numbers.
        raise TypeError(
            "Pass a tensor of observations, such as a PointCloudTensor, "
            "rather than a list of stablebear objects"
        )
    if not isinstance(values, Tensor):
        # The same dtype inference as sb.tensor(): float32, int32, uint32,
        # uint64, and bool are kept; other numbers become float64 or int64.
        array = np.asarray(values)
        if array.dtype.kind == "m":
            # sb.tensor() would keep the counts of the durations without their unit.
            raise TypeError("Durations cannot be observations; convert them to numbers of a unit first, "
                            'such as values / np.timedelta64(1, "s")')
        try:
            values = tensor(array)
        except TypeError as error:
            raise TypeError(
                f"Unsupported time series values of dtype {array.dtype}"
            ) from error
    if values.dtype not in _NATIVE_SUFFIX or not isinstance(
        values, _leaf_wrappers()[values.dtype]
    ):
        raise TypeError(f"Unsupported time series values: {type(values).__name__}")
    if isinstance(values, IndexedElementTensor) and values._has_indexed_storage():
        values = values.to_dense()
    return values


def _native_series(values, times, snap_tolerance):
    """The native series of ``values`` at ``times``, as :func:`_native_times`
    gives them."""
    stamps, unit, epoch = times
    tensor = _values_tensor(values)
    return _SERIES_NATIVE[tensor.dtype](np.ascontiguousarray(stamps, dtype=np.float64), unit, epoch, tensor._data,
                                        snap_tolerance)


class _DomainFields(NamedTuple):
    # A DateTime if the series has an epoch, else a TimeDelta, or a float
    # for a series without a unit.
    start: Any
    end: Any
    start_closed: bool
    end_closed: bool


class Domain(_DomainFields):
    """The times at which a series can be evaluated, from ``start`` to ``end``.

    ``start_closed`` and ``end_closed`` say whether ``start`` and ``end``
    themselves are included. The ends are durations, dates, or numbers, of
    one kind.

    Attributes
    ----------
    start : TimeDelta, DateTime, or float
        The first time in the domain, or its lower bound if not included.
    end : TimeDelta, DateTime, or float
        The last time in the domain, or its upper bound if not included.
    start_closed : bool
        Whether ``start`` is in the domain.
    end_closed : bool
        Whether ``end`` is in the domain.

    Raises
    ------
    ValueError
        If the domain holds no time: ``start`` comes after ``end``, or
        equals it without both ends included.
    """

    __slots__ = ()

    def __new__(cls, start: Any, end: Any, start_closed: bool, end_closed: bool):
        if end < start or (not start < end and not (start_closed and end_closed)):
            raise ValueError(f"A domain from {start} to {end} holds no time")
        return super().__new__(cls, start, end, bool(start_closed), bool(end_closed))


class TimeSeriesDomainError(ValueError):
    """A time outside the domain of a series, raised by calling a series or
    a tensor of them and by :func:`~stablebear.timeseries.to_point_clouds`.

    Attributes
    ----------
    time : DateTime, TimeDelta, or float
        The first time outside the domain: a :class:`DateTime` for a date,
        else a duration from time 0 of the series in its unit, or a number
        for a series without a unit.
    domain : Domain
        The domain that ``time`` is outside, with ends of the same kind as
        ``time``: that of the series, or of a channel where dates or a
        tensor of series place a time in each channel.
    index : tuple of int or None
        The index of the series in a tensor of them, or ``None`` for one
        series.
    """

    time: DateTime | TimeDelta | float
    domain: Domain
    index: tuple[int, ...] | None

    def __init__(self, message: str, time: DateTime | TimeDelta | float, domain: Domain,
                 index: tuple[int, ...] | None = None):
        super().__init__(message)
        self.time = time
        self.domain = domain
        self.index = index

    def __reduce__(self):
        return type(self), (str(self), self.time, self.domain, self.index)


def _domain_error(error, unit, index):
    """``error``, a C++ ``TimeSeriesDomainError`` for a time in ``unit``, as an
    :class:`TimeSeriesDomainError` at ``index`` with the time and the
    domain as durations, or as numbers for a series without a unit."""
    def in_unit(count) -> TimeDelta | float:
        return float(count) if unit is None else TimeDelta(float(count), unit)

    domain = Domain(in_unit(error.start), in_unit(error.end), error.start_closed, error.end_closed)
    return TimeSeriesDomainError(str(error), in_unit(error.time), domain, index)


_NOT_ITERABLE = ("A TimeSeries is not iterable; use series[i] for a channel, "
                 "or series.values for the observations")


class TimeSeries(_BinaryIoMixin, Generic[T]):
    """Observations recorded at known times, in one or more channels.

    A series built from ``values`` and ``times`` has one channel;
    :meth:`from_channels` combines several. A series is not iterable and
    has no ``len``, since it could iterate over either: ``series[i]``
    selects a channel, :attr:`values` gives the observations, and
    :attr:`n_observations` counts them. Series can be saved with
    ``sb.save``, loaded with ``sb.load``, and pickled; each channel keeps its
    times, unit, and snap settings.

    A series cannot be changed. The constructor copies ``values``, and
    :attr:`values` and :meth:`__call__` return copies, so changing the
    input or a result leaves the series unchanged. Channel selection,
    :meth:`to`, windows, and delay embeddings share the observations
    instead of copying them.

    Each channel has a domain: the times at which it can be evaluated. For
    a series built from observations, it runs from the first to the last
    observation, both included. A window (:func:`cut_windows`) runs from its
    start to its end, which may exclude its start and extend past its last
    observation, and a delay embedding (:func:`delay_embed`) keeps
    the ends of its channel's domain. A series can be evaluated where all
    of its channels can; :attr:`domain` gives these times.

    Element-wise operations, such as
    :func:`~stablebear.persistence.homology.compute_persistent_homology`, accept a
    series wherever they accept a tensor of the same elements. They apply to
    every channel's observations and return a series with the same channels,
    times, domains, unit, and snap settings. An operation that consumes
    trailing axes or adds axes to each element changes each channel's
    observation shape the same way it changes a tensor's shape.

    Parameters
    ----------
    values : array_like or Tensor
        Observations with time along axis 0. Each observation has shape
        ``values.shape[1:]``. A tensor of any element type, such as a
        ``PointCloudTensor`` or ``BarcodeTensor``, keeps its dtype. Arrays
        get the dtype ``sb.tensor`` would give them: float32, int32,
        uint32, uint64, and bool are kept, other floats become float64,
        and other integers become int64.
    times : TimeDeltaTensor, DateTimeTensor, or array_like
        One finite timestamp per observation, strictly increasing. Durations
        are a :class:`TimeDeltaTensor`, a tuple ``(counts, unit)`` such as
        ``([0.0, 1.5], "h")``, a ``timedelta64`` array, a sequence of
        :class:`TimeDelta` or other durations, or a pandas
        ``TimedeltaIndex``. Dates and times are a :class:`DateTimeTensor`, a
        ``datetime64`` array, a list of values that :class:`DateTime`
        accepts, such as ``datetime`` objects or date strings, or a pandas
        ``DatetimeIndex`` or ``Series`` of dates, whose timezone, if any, is
        converted to UTC and dropped; they take no unit and are counted from
        ``epoch``, which defaults to the first of them. The
        :attr:`unit` is that of the durations or dates: the unit of a
        ``TimeDeltaTensor`` or ``DateTimeTensor``, the resolution of an array or index, such as
        minutes for a ``timedelta64[m]`` array, or the finest unit of a
        sequence of durations. A plain array of numbers gives a series
        without a unit, whose :attr:`unit` is ``None``.
    epoch : DateTime, str, np.datetime64, datetime.datetime, or datetime.date, optional
        The date and time that time 0 stands for, such as ``"2023-01-01"``.
        Dates keep their instants and are counted from it, in integers and
        converted to the unit once, so whole multiples of the unit are
        exact. Durations have their time 0 at it, as with
        :meth:`with_epoch`. Defaults to the first of the dates, or to none
        for durations. A series without a unit has no epoch.
    snap_tolerance : float, optional
        How close below an observation a query may land and still get that
        observation, relative to the magnitude of the timestamps, as in the
        tolerant comparison of APL. This lets a time computed in floating
        point, such as ``0.3 - 0.1``, reach the observation it means. Must
        be in ``[0, 1)``; 0 disables snapping. Defaults to ``1e-14``, about
        45 times machine epsilon, the default comparison tolerance of APL.

    Notes
    -----
    Let tol be ``snap_tolerance``, and let the timestamps be
    t[0] < t[1] < ... < t[n-1], with observations x[0], x[1], ..., x[n-1].
    At a time t >= t[0], the series is x[i] for the largest i such that
    i = 0 or ::

        t >= t[i] - min(tol * max(|t[i-1]|, |t[i]|), (t[i] - t[i-1]) / 4).

    On large timestamps close together, ``tol * max(|t[i-1]|, |t[i]|)`` can
    exceed the gap, so that a query at t[i-1] itself would get x[i]; the
    quarter of the gap keeps a query nearer the previous observation from
    snapping. With tol = 0, the series is the last observation at or
    before t.
    """

    def __init__(self, values, times, *, epoch=None, snap_tolerance: float = 1e-14):
        self._data = _native_series(values, _native_times(times, epoch=epoch), snap_tolerance)

    @classmethod
    def _from_cpp(cls, data):
        series = cls.__new__(cls)
        series._data = data
        return series

    def _binary_io_data(self):
        return self._data

    @classmethod
    def from_channels(cls, channels, unit: str | None = None) -> TimeSeries:
        """Combine one-channel series into one series with several channels.

        Each channel keeps its own times and ``snap_tolerance``, converted
        to ``unit`` as :meth:`to` converts them. Channels with different
        epochs are counted from the earliest
        of them: each channel's times and domain end are shifted by the
        difference of its epoch, converted to its unit by an exact factor,
        so it keeps its dates up to that floating-point addition.

        Parameters
        ----------
        channels : list of TimeSeries
            One-channel series of one element type; channel ``i`` of the
            result is ``channels[i]``.
        unit : str, optional
            The unit of the series, into which every channel is converted.
            Defaults to the unit the channels share; required when their
            units differ.

        Raises
        ------
        ValueError
            If there are no channels, a channel has several channels,
            ``unit`` is omitted while the channels' units differ, or some
            channels have an :attr:`epoch` and others do not.
        TypeError
            If a channel is not a ``TimeSeries`` or the channels have
            different element types.
        """
        parts = list(channels)
        if not all(isinstance(part, TimeSeries) for part in parts):
            raise TypeError("Each channel must be a TimeSeries")
        dtypes = {part.dtype for part in parts}
        if len(dtypes) > 1:
            raise TypeError(
                "All channels must have the same element type, got "
                + ", ".join(sorted(str(dtype) for dtype in dtypes))
            )
        if not parts:
            raise ValueError("A time series needs at least one channel")
        native = _SERIES_NATIVE[parts[0].dtype]
        return cls._from_cpp(native.from_channels(
            [part._data for part in _on_one_epoch(parts, unit)], unit
        ))

    @classmethod
    def from_pandas(cls, obj, unit: str | None = None, *, snap_tolerance: float = 1e-14) -> TimeSeries:
        """Build a series from a pandas ``Series`` or ``DataFrame``, whose index
        gives the times.

        A ``Series`` gives one channel, and a ``DataFrame`` one channel per
        column, in column order, all at the times of the index. Each column
        is converted as a ``Series`` is, and the columns then take one
        common dtype, such as float64 for int and float columns. Column
        names are not kept.

        A ``DatetimeIndex`` is counted from its first date and time, which
        becomes the :attr:`epoch`; one with a timezone is converted to UTC
        first. A ``TimedeltaIndex`` gives times with no epoch; set one with
        :meth:`with_epoch`. A numeric index counts ``unit``, or gives a
        series without a unit if ``unit`` is omitted.

        Parameters
        ----------
        obj : pandas.Series or pandas.DataFrame
            Observations of numbers or booleans. Missing values, ``NaN`` and
            ``pd.NA``, are kept as ``NaN``; call ``obj.dropna()`` first to
            drop them.
        unit : str, optional
            The unit of the times, as for :class:`TimeDelta`. Defaults to the
            resolution of a ``DatetimeIndex`` or ``TimedeltaIndex``, mostly
            microseconds in pandas 3 and nanoseconds in pandas 2, and to none
            for a numeric index.
        snap_tolerance : float, optional
            As for the constructor.

        Raises
        ------
        TypeError
            If ``obj`` is not a pandas ``Series`` or ``DataFrame``, its index
            holds neither dates, durations, nor numbers, or its values are
            not numbers or booleans.
        ValueError
            If ``obj`` is a ``DataFrame`` with no columns, or its index holds
            ``NaT`` or ``NaN``, is not sorted, or has duplicates.
        """
        pd = sys.modules.get("pandas")
        if pd is None or not isinstance(obj, (pd.Series, pd.DataFrame)):
            raise TypeError(f"Expected a pandas Series or DataFrame; got {type(obj).__name__}")
        index = obj.index
        temporal = isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex))
        if not temporal and (isinstance(index, pd.MultiIndex) or index.dtype.kind not in "iuf"):
            raise TypeError(f"Expected an index of dates, durations, or numbers; got {type(index).__name__}")
        if index.hasnans:
            raise ValueError("The index holds missing times; drop those rows with obj[obj.index.notna()]")
        if index.has_duplicates:
            raise ValueError("The index holds duplicated times; find them with obj.index.duplicated()")
        if not index.is_monotonic_increasing:
            raise ValueError("The index is not sorted; sort it with obj.sort_index()")
        times = _native_times(index if temporal else np.asarray(index, dtype=np.float64), unit)
        if isinstance(obj, pd.Series):
            return cls._from_cpp(_native_series(np.asarray(obj), times, snap_tolerance))
        if obj.shape[1] == 0:
            raise ValueError("A DataFrame needs at least one column to give a channel")
        columns = [np.asarray(obj.iloc[:, i]) for i in range(obj.shape[1])]
        dtype = np.result_type(*columns)
        return cls.from_channels([cls._from_cpp(_native_series(column.astype(dtype), times, snap_tolerance))
                                  for column in columns])

    def to_pandas(self, resolution: str | None = None) -> Any:  # A pandas Series or DataFrame; pandas is optional.
        """The series as a pandas ``Series`` of its observations, indexed by
        its times, or for several channels a ``DataFrame`` with columns ``0``
        to ``n_channels - 1``, indexed by the union of the channels' times.
        A column holds its channel's value in effect at each of these times,
        as :meth:`__call__` gives it, and ``NaN`` outside the channel's
        domain.

        The index is a ``DatetimeIndex`` of ``epoch + origin + times`` for a
        series with an :attr:`epoch`, a ``TimedeltaIndex`` of :attr:`times`
        for one without, and an ``Index`` of the numbers for a series
        without a unit. Dates and durations are rounded to ``resolution``.

        Parameters
        ----------
        resolution : {"s", "ms", "us", "ns"}, optional
            The resolution of the dates or durations, one that pandas
            supports. Defaults to the finer of :attr:`unit` and
            microseconds. A unit finer than nanoseconds, the finest
            resolution pandas supports, needs ``resolution="ns"`` to round
            to nanoseconds. A series without a unit takes none.

        Raises
        ------
        TypeError
            If the observations are not scalar numbers or booleans.
        ValueError
            For an unknown ``resolution``, a unit finer than nanoseconds
            without ``resolution``, or a ``resolution`` for a series without
            a unit.
        ImportError
            If pandas is not installed.
        """
        from .._pandas import import_pandas
        pd = import_pandas()
        shapes = (self.shape,) if self.n_channels == 1 else self.shape
        if self.dtype not in _SCALAR_DTYPES or any(shape != () for shape in shapes):
            raise TypeError(f"Only series of scalar numbers or booleans convert to pandas, not of {self.dtype}")
        resolution = self._pandas_resolution(resolution)
        # Each channel's times where they lie: placed by its origin for a
        # series with an epoch, else from its own time 0.
        offsets = [self._data.origin(i) if self.epoch is not None else 0.0 for i in range(self.n_channels)]
        if self.n_channels == 1:
            return pd.Series(np.asarray(self.values),
                             index=self._pandas_index(pd, np.asarray(self._data.times(0)) + offsets[0], resolution))
        # The index is the union of the channels' times rounded to labels,
        # so that one instant held as two floats, as shifting a channel by
        # its epoch can leave it, gives one label.
        times = [np.asarray(self._data.times(i)) for i in range(self.n_channels)]
        labels = [self._pandas_index(pd, t + offset, resolution) for t, offset in zip(times, offsets)]
        placed = pd.Series(np.concatenate([t + offset for t, offset in zip(times, offsets)]),
                           index=labels[0].append(labels[1:]))
        placed = placed.groupby(level=0).first()
        index = placed.index
        columns = {}
        for i, offset in enumerate(offsets):
            # The value in effect at every label of the union in the
            # channel's domain, and NaN outside it. The domain is checked on
            # labels too, and a label is evaluated at the channel's own time
            # if it has one there, else at the first time placed there,
            # moved into the domain if it missed a closed end by rounding.
            channel = self[i]
            own = pd.Series(times[i], index=labels[i]).groupby(level=0).last().reindex(index)
            start, end, start_closed, end_closed = channel._data.domain
            after_start, before_end = np.full(len(index), True), np.full(len(index), True)
            if np.isfinite(start):
                start_label = self._pandas_index(pd, np.array([start + offset]), resolution)[0]
                after_start = (index > start_label) | (start_closed & (index == start_label))
            if np.isfinite(end):
                end_label = self._pandas_index(pd, np.array([end + offset]), resolution)[0]
                before_end = (index < end_label) | (end_closed & (index == end_label))
            inside = own.notna().to_numpy() | (after_start & before_end)
            t = np.clip(own.fillna(placed - offset).to_numpy()[inside], start, end)
            values = np.asarray(channel._evaluate_numbers(FloatTensor(t))) if inside.any() else []
            columns[i] = pd.Series(values, index=index[inside], dtype=np.asarray(channel.values).dtype).reindex(index)
        return pd.DataFrame(columns, index=index)

    def _pandas_resolution(self, resolution):
        """``resolution`` for :meth:`to_pandas`, checked, or its default."""
        if self.unit is None:
            if resolution is not None:
                raise ValueError("A series without a unit has no resolution; pass none")
            return None
        if resolution is None:
            if _ATTOSECONDS[self.unit] < _ATTOSECONDS["ns"]:
                raise ValueError(f"pandas holds times to nanoseconds at finest, not {self.unit!r}; "
                                 "pass resolution='ns' to round to nanoseconds")
            return self.unit if _ATTOSECONDS[self.unit] < _ATTOSECONDS["us"] else "us"
        if resolution not in ("s", "ms", "us", "ns"):
            raise ValueError(f"resolution must be 's', 'ms', 'us', or 'ns', not {resolution!r}")
        return resolution

    def _pandas_index(self, pd, counts, resolution):
        """A pandas index of ``counts``, numbers of :attr:`unit`: dates from
        the epoch, durations, or for a series without a unit, the numbers."""
        if self.unit is None:
            return pd.Index(counts)
        durations = TimeDeltaTensor(FloatTensor(np.asarray(counts, dtype=np.float64)), self.unit)
        if self.epoch is None:
            return pd.TimedeltaIndex(durations.to_timedelta64(resolution))
        dates = self.epoch.to_datetime64() + durations.to_timedelta64(resolution)
        return pd.DatetimeIndex(dates.astype(f"datetime64[{_UNIT_TO_NUMPY[resolution]}]"))

    def _single_channel(self):
        if self._data.n_channels != 1:
            raise ValueError(
                "This series has several channels; select one with series[i]"
            )
        return self._data.channel(0)

    @property
    def n_channels(self) -> int:
        """The number of channels."""
        return self._data.n_channels

    @property
    def unit(self) -> str | None:
        """The time unit in which the times of every channel are stored, or
        ``None`` for a series without a unit, whose times are plain
        numbers."""
        return self._data.unit

    @property
    def dtype(self):
        """The element dtype shared by all channels."""
        return _SERIES_CPP_TO_DTYPE[type(self._data)]

    @property
    def times(self) -> Any:  # Depends on whether the series has a unit; see the docstring.
        """The timestamps of the recorded observations, a
        :class:`TimeDeltaTensor` of float counts in :attr:`unit` from time
        0, with or without an :attr:`epoch`. Their dates are
        ``epoch + origin + times``, where :attr:`origin` places a window in
        the series it was cut from. For a series without a unit, a
        ``FloatTensor`` of the numbers.

        Raises
        ------
        ValueError
            For several channels, whose times differ in length; select a
            channel with ``series[i]`` first.
        """
        if self._data.n_channels > 1:
            raise ValueError("The channels of this series have their own times; select one with series[i].times")
        return _times_in(np.array(self._data.times(0)), self.unit)

    @property
    def datetimes(self) -> DateTimeTensor:
        """The dates and times of the recorded observations, a
        :class:`DateTimeTensor` of ``epoch + origin + times``, at the finer
        of :attr:`unit` and microseconds. For a window, these are the
        instants where its observations lie in the series it was cut from.

        Raises
        ------
        ValueError
            If the series has no :attr:`epoch`, or for several channels,
            whose times differ in length; select a channel with
            ``series[i]`` first.
        """
        if self._data.n_channels > 1:
            raise ValueError("The channels of this series have their own times; select one with series[i].datetimes")
        if self.epoch is None:
            raise ValueError("This series has no epoch to give its times as dates; set one with with_epoch")
        return self.epoch + (self.origin + self.times)

    @property
    def values(self) -> Any:  # Depends on the number of channels; see the docstring.
        """A copy of the observations, with time along axis 0.

        A tensor of shape ``(n_observations, *shape)`` for a series with one
        channel. For several channels, a ``NestedTensor`` with one such
        leaf per channel, in channel order.
        """
        wrapper = _leaf_wrappers()[self.dtype]
        if self.n_channels == 1:
            return wrapper(self._data.channel(0).values)
        return NestedTensor([
            wrapper(self._data.channel(i).values) for i in range(self.n_channels)
        ])

    @property
    def shape(self) -> Any:  # Depends on the number of channels; see the docstring.
        """The shape of one observation; ``()`` for scalars.

        For several channels, a tuple with one shape per channel, in
        channel order.
        """
        shapes = tuple(self._data.channel(i).shape for i in range(self.n_channels))
        return shapes[0] if len(shapes) == 1 else shapes

    @property
    def origin(self) -> Any:  # Depends on the number of channels; see the docstring.
        """The time in the series a channel was cut from that its time 0
        stands for, as a duration in :attr:`unit`.

        0 for a series built from observations. Evaluation does not use it;
        it places a window in the series it was cut from, at its
        :attr:`source_domain`. A :class:`TimeDelta` for a series with one channel; for
        several, a :class:`TimeDeltaTensor` with one entry per channel. For
        a series without a unit, a float or a ``FloatTensor``.
        """
        origins = [self._data.origin(i) for i in range(self.n_channels)]
        return _times_in(origins[0] if len(origins) == 1 else origins, self.unit)

    @property
    def epoch(self) -> Any:  # A DateTime, or None; see the docstring.
        """The date and time that time 0 stands for, or ``None``.

        One value even for several channels, which share their epoch.
        """
        data = self._data.epoch
        return None if data is None else DateTime._from_cpp(data)

    def with_epoch(self, epoch) -> TimeSeries:
        """Return the same series with every channel counted from ``epoch``.

        ``epoch`` is a :class:`DateTime` or a value it accepts, or ``None``
        to remove the epoch. No time is shifted, and the observations are
        shared, not copied. The epoch is the date of time 0 of the series a
        window was cut from, so a window keeps its :attr:`origin`.

        Raises
        ------
        ValueError
            If ``epoch`` is not ``None`` and the series has no unit.
        """
        if epoch is not None:
            epoch = DateTime(epoch)._data
        return TimeSeries._from_cpp(self._data.with_epoch(epoch))

    @property
    def domain(self) -> Domain:
        """The times at which every channel can be evaluated.

        Without an :attr:`epoch`, the ends are :class:`TimeDelta` durations
        from time 0 in :attr:`unit`, or floats for a series without a unit.
        For several channels, the domain is the
        intersection of theirs, which :meth:`__call__` checks queries
        against.

        With an epoch, the ends are :class:`DateTime` instants: the dates
        and times at which the series can be evaluated, so a window's domain
        is where it lies in the series it was cut from, the same as its
        :attr:`source_domain`. For several channels, it is the intersection
        of where each lies.

        Raises
        ------
        ValueError
            If a channel has no observations.
        """
        if self.epoch is None:
            start, end, start_closed, end_closed = self._data.domain
            return Domain(_times_in(start, self.unit), _times_in(end, self.unit), start_closed, end_closed)
        return self.source_domain

    @property
    def source_domain(self) -> Domain:
        """Where the series lies in the series it was cut from.

        The :attr:`domain` of each channel moved by its :attr:`origin`, so
        for a window, the times it covers in the series it was cut from,
        with the ends it includes. For a series built from observations,
        whose origin is 0, it equals its domain. The ends are as for
        :attr:`domain`: durations from time 0 in :attr:`unit`, floats for a
        series without a unit, or :class:`DateTime` instants for a series
        with an :attr:`epoch`. For several channels, it is the intersection
        of where each lies.

        Raises
        ------
        ValueError
            If a channel has no observations.
        """
        start, end, start_closed, end_closed = self._source_domain_counts()
        epoch = self.epoch
        if epoch is None:
            return Domain(_times_in(start, self.unit), _times_in(end, self.unit), start_closed, end_closed)
        return Domain(epoch + TimeDelta(start, self.unit), epoch + TimeDelta(end, self.unit), start_closed, end_closed)

    def _source_domain_counts(self):
        """The intersection of each channel's domain moved by its origin, in
        :attr:`unit`, as C++ intersects channels: an open end at the same
        time as a closed one excludes that time."""
        start, end, start_closed, end_closed = -np.inf, np.inf, True, True
        for i in range(self.n_channels):
            first, last, first_closed, last_closed = self._data.select([i]).domain
            origin = self._data.origin(i)
            first, last = first + origin, last + origin
            if first > start or (first == start and not first_closed):
                start, start_closed = first, first_closed
            if last < end or (last == end and not last_closed):
                end, end_closed = last, last_closed
        return start, end, start_closed, end_closed

    def restrict(self, domain: Domain) -> TimeSeries:
        """The series on ``domain``, a part of its own domain.

        The ends of ``domain`` are times of the series: durations from
        time 0, or numbers for a series without a unit. Build one with
        :class:`Domain`, as in ``Domain(TimeDelta(1, "s"), TimeDelta(3, "s"),
        start_closed=False, end_closed=True)``. The times keep their place:
        the unit, epoch, and origin stay the same. Each channel holds the
        observation in effect at the new start from the start on, then its
        observations inside the domain, as a window from :func:`cut_windows`
        does. Observations are shared, not copied.

        For a series with an :attr:`epoch`, restrict it without one, as in
        ``series.with_epoch(None).restrict(domain).with_epoch(series.epoch)``.

        Parameters
        ----------
        domain : Domain
            The domain of the result, within that of the series.

        Returns
        -------
        TimeSeries
            The series on ``domain``.

        Raises
        ------
        ValueError
            If ``domain`` reaches outside the series' domain, including an
            end the series leaves out but ``domain`` includes, or a channel
            has no observations.
        TypeError
            For a series with an epoch, and for ends of the wrong kind:
            numbers on a series with a unit, or durations on one without.
        """
        if self.epoch is not None:
            raise TypeError("Restrict a series with an epoch without it: "
                            "series.with_epoch(None).restrict(domain).with_epoch(series.epoch)")
        start, end, start_closed, end_closed = domain
        a = _in_unit(_end_of_domain(start), self.unit)
        b = _in_unit(_end_of_domain(end), self.unit)
        lo, hi, lo_closed, hi_closed = self._data.domain
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError("The ends of a domain must be finite")
        if (a < lo or (a == lo and start_closed and not lo_closed)
                or b > hi or (b == hi and end_closed and not hi_closed)):
            raise ValueError(f"The domain {domain} reaches outside the series' domain {self.domain}")
        return TimeSeries._from_cpp(cpp.restrict(self._data, a, b, start_closed, end_closed))

    @property
    def n_observations(self) -> Any:  # Depends on the number of channels; see the docstring.
        """The number of recorded observations.

        An int for a series with one channel;
        for several, an int64 ``IntTensor`` with one entry per channel.
        """
        counts = [len(self._data.channel(i)) for i in range(self.n_channels)]
        if len(counts) == 1:
            return counts[0]
        return IntTensor(np.array(counts, dtype=np.int64))

    def _per_channel(self, name):
        values = [getattr(self._data.channel(i), name) for i in range(self.n_channels)]
        if len(values) == 1:
            return values[0]
        return FloatTensor(np.array(values, dtype=np.float64))

    @property
    def snap_tolerance(self) -> Any:  # Depends on the number of channels; see the docstring.
        """The snap tolerance used when evaluating; see the constructor.

        A float for a series with one channel; for several, a float64
        ``FloatTensor`` with one entry per channel.
        """
        return self._per_channel("snap_tolerance")

    def to(self, unit: str) -> TimeSeries:
        """Return the same series with its times in ``unit``.

        Every timestamp, domain end, and :attr:`origin` is converted by one
        exact factor and rounded once, as for :class:`TimeDelta`; the
        observations are shared, not copied. Accepts the units of
        :class:`TimeDelta`. A series without a unit gets ``unit`` instead:
        its numbers become counts of it, so times ``[0.0, 1.5]`` become 0 s
        and 1.5 s with ``to("s")``.

        Raises
        ------
        ValueError
            If ``unit`` is unknown.
        """
        return TimeSeries._from_cpp(self._data.to(unit))

    def array_equal(self, other) -> bool:
        """Whether ``other`` is a series of the same dtype and unit whose
        channels have the same timestamps, values, snap tolerance, and
        start and end.

        :attr:`origin` is not compared: it places a series but does not
        change how it evaluates, so two windows with the same contents cut
        from different places are equal.
        """
        return (isinstance(other, TimeSeries) and other.dtype is self.dtype
                and self._data.array_equal(other._data))

    def __call__(self, t, unit: str | None = None) -> Any:  # Depends on the number of channels; see the docstring.
        """Each channel's observation with the largest timestamp at or before ``t``.

        A ``t`` just below a timestamp snaps forward to it, by each
        channel's own ``snap_tolerance``; see the constructor. ``t`` must lie
        in the domain of every channel; see :class:`TimeSeries`.

        ``t`` is a single time or many, of shape ``t_shape``. Many times are
        evaluated together: they are sorted once and merged with each
        channel's timestamps. Times count from the series' own time 0, which
        for a window is its start.

        A series with a :attr:`unit` takes durations. A duration is a
        :class:`TimeDelta`, a tuple ``(count, unit)`` such as
        ``(90, "min")``, or a value :class:`TimeDelta` accepts, such as
        ``np.timedelta64(90, "m")``. Many durations are a
        :class:`TimeDeltaTensor`, a tuple ``(counts, unit)``, or an array,
        sequence, or pandas ``TimedeltaIndex`` of durations. Durations are
        converted to the series' unit by an exact factor. With ``unit``,
        ``t`` is a number or numbers of it: ``series(90, "min")`` is
        ``series(TimeDelta(90, "min"))``. For a series with an
        :attr:`epoch`, ``t`` may also be a date and time, a :class:`DateTime`
        or anything it accepts, such as ``"2023-01-01T12:30"``, or a
        :class:`DateTimeTensor`, a list, tuple, or array, such as a
        ``datetime64[15m]`` array, or a pandas ``DatetimeIndex`` or
        ``Series`` of them. A date names an instant, not an offset:
        it is placed at ``t - epoch - origin`` in each channel, so a window
        gives the value it held at that instant, as the series it was cut
        from does.

        A series without a unit takes plain numbers: a number, or an
        array-like or ``FloatTensor`` of them, and no ``unit``.

        For one channel and a single time, returns a scalar or element for
        observations of shape ``()``, otherwise a tensor of that shape; for
        times of shape ``t_shape``, returns a tensor of shape
        ``(*t_shape, *shape)``. For several channels, returns a
        ``NestedTensor`` with one such result per channel, in channel order,
        as a tensor (0-d for a single time and shape ``()``).

        Raises
        ------
        TimeSeriesDomainError
            If a time is outside the domain of any channel.
        ValueError
            If a channel has no observations, or if ``unit`` is unknown.
        TypeError
            For numbers without ``unit`` on a series with a unit, or for
            durations, dates, or ``unit`` on a series without one.
        """
        if unit is not None:
            if self.unit is None:
                raise TypeError("A series without a unit takes plain numbers; pass no unit")
            t = _with_unit(t, unit)
        numbers = _numbers(t)
        if numbers is not None:
            if self.unit is not None:
                raise TypeError(f"This series has a unit, so its times are durations; pass the unit, "
                                f"as in series(t, {self.unit!r}), or a TimeDelta")
            return self._evaluate_numbers(numbers)
        if self.unit is None:
            raise TypeError("A series without a unit takes plain numbers, not durations or dates")
        dates = _dates_of(t)
        if dates is not None:
            return self._evaluate_date(dates)
        duration = _as_time_delta(t)
        if duration is not None:
            return self._evaluate_numbers(_in_unit(duration, self.unit))
        if not isinstance(t, Tensor):
            # Lists, arrays, and pandas objects, such as a TimedeltaIndex.
            durations = _durations_in(_as_array(t), self.unit)
            if durations is not None:
                return self._evaluate_numbers(FloatTensor(np.asarray(durations, dtype=np.float64)))
        raise TypeError(f"Expected durations or dates and times, not {type(t).__name__}")

    def _evaluate_numbers(self, t):
        """Evaluate at ``t``, a float or an array or ``FloatTensor`` of times
        in :attr:`unit`."""
        if np.ndim(t) > 0:
            t = FloatTensor(np.asarray(t, dtype=np.float64))
        try:
            native = self._data.evaluate(t._data if isinstance(t, FloatTensor) else float(t))
        except cpp.TimeSeriesDomainError as error:
            raise _domain_error(error, self.unit, None) from None
        values = NestedTensor._from_cpp(native)
        if isinstance(t, FloatTensor):
            return values if self.n_channels > 1 else values[0]
        if self.n_channels > 1:
            return values
        leaf = values[0]
        return leaf[()] if leaf.ndim == 0 else leaf

    def _time_of_date(self, date, channel):
        """The time of ``date`` in channel ``channel``, in :attr:`unit`:
        ``date - epoch - origin``, with the difference taken in integers, and
        placed at a closed end of the channel's domain that it misses by at
        most the snap tolerance times the larger magnitude of
        ``date - epoch`` and the origin, as ``time_of_date`` in C++ places
        it."""
        epoch, unit = self.epoch, self.unit
        assert epoch is not None and unit is not None
        offsets = _offsets_in(date, epoch, unit)
        origin = self._data.origin(channel)
        t = offsets - origin
        start, end, start_closed, end_closed = self._data.select([channel]).domain
        reach = self._data.channel(channel).snap_tolerance * np.maximum(np.abs(offsets), abs(origin))
        t = np.where(start_closed & (t < start) & (start - t <= reach), start, t)
        return np.where(end_closed & (t > end) & (t - end <= reach), end, t)

    def _check_date_in_domain(self, date, t, channel):
        """Raise naming ``date`` and the domain of channel ``channel`` as dates
        if a time ``t`` is outside it."""
        start, end, start_closed, end_closed = self._data.select([channel]).domain
        t = np.asarray(t)
        inside = (((t > start) | (start_closed & (t == start)))
                  & ((t < end) | (end_closed & (t == end))))
        if inside.all():
            return
        first = np.asarray(date).reshape(-1)[np.argmin(inside.reshape(-1))]
        origin = self._data.origin(channel)
        ends = (self.epoch.to_datetime64()
                + TimeDeltaTensor(FloatTensor([start + origin, end + origin]), self.unit).to_timedelta64())
        # Formatted at one precision, the coarsest that shows all three.
        dates = np.array([first, *ends])
        finest = np.datetime_data(dates.dtype)[0]
        precision = next(code for code in ("m", "s", "ms", "us", "ns", "ps", "fs", "as")
                         if code == finest or (dates == dates.astype(f"datetime64[{code}]")).all())
        first_text, start_text, end_text = np.datetime_as_string(dates, unit=precision)  # pyright: ignore[reportCallIssue, reportArgumentType]
        domain_text = ("[" if start_closed else "(") + f"{start_text}, {end_text}" + ("]" if end_closed else ")")
        domain = Domain(DateTime(ends[0]), DateTime(ends[1]), start_closed, end_closed)
        raise TimeSeriesDomainError(f"{first_text} is outside the domain {domain_text}", DateTime(first), domain)

    def _evaluate_date(self, date):
        """Evaluate at ``date``, one date or an array of them, placing it in
        each channel by its origin."""
        if self.epoch is None:
            raise ValueError("This series has no epoch to place dates and times; set one with with_epoch")

        def query(channel):
            t = self._time_of_date(date, channel)
            self._check_date_in_domain(date, t, channel)
            return float(t) if np.ndim(t) == 0 else FloatTensor(np.asarray(t, dtype=np.float64))

        if self.n_channels == 1:
            return self._evaluate_numbers(query(0))
        # Channels of a window can have different origins, so each is
        # queried at its own time.
        leaves = []
        for i in range(self.n_channels):
            t = query(i)
            native = self._data.select([i]).evaluate(t._data if isinstance(t, FloatTensor) else t)
            leaves.append(NestedTensor._from_cpp(native)[0])
        return NestedTensor(leaves)

    def __iter__(self) -> NoReturn:
        """Raises ``TypeError``: a series is not iterable, since it could
        iterate over its channels or its observations. Use ``series[i]`` for
        a channel, :attr:`values` for the observations, and
        :attr:`n_observations` to count them."""
        raise TypeError(_NOT_ITERABLE)

    def __array__(self, dtype=None, copy=None) -> NoReturn:
        """Raises ``TypeError``, as :meth:`__iter__` does, instead of letting
        NumPy build an array of the series."""
        raise TypeError(_NOT_ITERABLE)

    def __getitem__(self, index) -> TimeSeries:
        """Select channels by index: ``series[i]`` or ``series[[i, j, ...]]``.

        Returns a series of the selected channels, in the order given, with
        the same unit. The channels share their observations with this
        series. Negative indices count from the end.

        Raises
        ------
        IndexError
            If an index is out of range.
        ValueError
            If the list of indices is empty.
        """
        if isinstance(index, (list, tuple)):
            indices = [operator.index(i) for i in index]
        else:
            indices = [operator.index(index)]
        return TimeSeries._from_cpp(self._data.select(indices))

    def __repr__(self):
        return (
            f"TimeSeries(n_channels={self.n_channels}, dtype={self.dtype}, "
            f"unit={self.unit!r})"
        )


def _queries(t, unit=None):
    """Times ``t``, durations or numbers of ``unit``, as numbers and their
    unit for C++, which converts them into the unit of each series; plain
    numbers without ``unit`` get the unit ``None``. Durations are counted in
    their own unit."""
    if unit is not None:
        t = _with_unit(t, unit)
    duration = _as_time_delta(t)
    if duration is not None:
        return _in_unit(duration, duration.unit), duration.unit
    if not isinstance(t, (Tensor, int, float, np.integer, np.floating)):
        t = _as_array(t)
        finest = _finest_duration_unit(t)
        if finest is not None:
            return _durations_in(t, finest), finest
    return t, None


class TimeSeriesTensor(Tensor):
    """Tensor whose elements are :class:`TimeSeries`, such as the windows of
    a series. Its dtype is ``sb.ts(value_type)`` for series of ``value_type``
    observations."""

    def __init__(self, data):
        super().__init__()
        if isinstance(data, TimeSeriesTensor):
            data = data._data
        elif type(data) not in _SERIES_TENSOR_TO_DTYPE:
            raise TypeError(f"Cannot create TimeSeriesTensor from {type(data)}")
        self._data = data
        self.dtype = _SERIES_TENSOR_TO_DTYPE[type(data)]

    @staticmethod
    def _from_series(data, dtype=None):
        """The tensor of the series in the nested lists ``data``, of dtype
        ``dtype`` if given."""
        _, flat = _infer_shape_and_flatten(data)
        if not all(isinstance(series, TimeSeries) for series in flat):
            raise TypeError("A tensor of series takes only TimeSeries elements")
        value_types = list(dict.fromkeys(series.dtype for series in flat))
        if len(value_types) > 1:
            raise TypeError(
                "The series of a tensor must have one element type, not "
                + ", ".join(repr(value_type) for value_type in value_types))
        if dtype is not None and value_types and ts(value_types[0]) is not dtype:
            raise TypeError(f"Series of {value_types[0]!r} do not have dtype {dtype!r}")
        if value_types:
            value_type = value_types[0]
        elif dtype is not None:
            value_type = dtype.value_type
        else:
            raise TypeError("An empty tensor of series needs a dtype")
        suffix = _NATIVE_SUFFIX[value_type]
        tensor_class = getattr(cpp, f"TimeSeries{suffix}Tensor")
        return TimeSeriesTensor(_tensor_from_nested(
            data, {getattr(cpp, f"TimeSeries{suffix}"): tensor_class}, tensor_class))

    def _to_py_tensor(self, data):
        return TimeSeriesTensor(data)

    def _decay_value(self, val):
        return val._data

    def _represent_element(self, element):
        return TimeSeries._from_cpp(element)

    def _get_valid_setitem_dtypes(self):
        return [TimeSeries, TimeSeriesTensor]

    def __call__(self, t, unit: str | None = None) -> Any:  # Depends on the number of channels; see the docstring.
        """Every series evaluated at the times ``t``, as
        :meth:`~stablebear.timeseries.TimeSeries.__call__` does: durations, or
        numbers of ``unit``, for series with a unit, and plain numbers without
        ``unit`` for series without one. Times count from each series' time 0.
        When every series has an
        :attr:`~stablebear.timeseries.TimeSeries.epoch`, ``t`` may also be
        dates and times, as :meth:`~stablebear.timeseries.TimeSeries.__call__`
        takes them; each series places a date at ``date - epoch - origin`` in
        each channel, so series with different epochs are evaluated at the
        same instants.

        ``t`` is a single time or an array-like or ``FloatTensor`` of times
        of shape ``t_shape``. For series with one channel, returns a tensor
        of shape ``(*self.shape, *t_shape, *shape)``, as evaluating a
        ``PcfTensor`` gives ``(*self.shape, *t_shape)``. For several
        channels, returns a ``NestedTensor`` with one such tensor per
        channel, in channel order, each with that channel's ``shape``. For
        windows of a delay embedding of a scalar series evaluated on a grid
        of ``n`` times, the last two axes are clouds of ``n`` points, which
        :func:`~stablebear.persistence.homology.compute_persistent_homology` reads
        as point clouds. Observations of shape ``S``, embedded with
        dimension ``m``, give the axes ``(n, m, *S)`` instead; reshape them
        to ``(n, m * prod(S))`` to get clouds of ``n`` points.

        Raises
        ------
        TimeSeriesDomainError
            If a time is outside the domain of a series, with the
            ``index`` of the first such series.
        ValueError
            If the tensor is empty or its series differ in channels or
            shapes, if ``unit`` is unknown, or for dates and times when a
            series has no epoch.
        TypeError
            For numbers without ``unit`` on series with a unit, or for
            durations or ``unit`` on series without one.
        """
        dates = None if unit is not None else _dates_of(t)
        if dates is not None:
            counts, date_unit = _date_counts(dates)
            try:
                native = cpp.evaluate_series_dates(self._data, counts._data, date_unit)
            except cpp.TimeSeriesDomainError as error:
                raise self._naming_dates(error, dates) from None
            return self._leaves(native)
        return self._evaluate(*_queries(t, unit))

    def _naming_dates(self, error, dates):
        """``error``, a C++ ``TimeSeriesDomainError`` for ``dates``, a
        ``datetime64`` array, as an :class:`TimeSeriesDomainError` with the
        date and the domain of the series and channel it gives as dates, as
        a call of that series gives them."""
        index = tuple(error.index)
        series = self[index]
        try:
            series._check_date_in_domain(dates, series._time_of_date(dates, error.channel), error.channel)
        except TimeSeriesDomainError as named:
            return TimeSeriesDomainError(f"TimeSeries at position ({', '.join(map(str, index))}): {named}",
                                      named.time, named.domain, index)
        # The series takes the dates in Python but not in C++; keep the
        # times of C++.
        return self._domain_error(error)

    def _domain_error(self, error):
        """``error``, a C++ ``TimeSeriesDomainError`` for a time of a series of
        the tensor, as an :class:`TimeSeriesDomainError` in the unit of that
        series."""
        index = tuple(error.index)
        return _domain_error(error, self[index].unit, index)

    def _evaluate(self, t, unit):
        """Evaluate at numbers ``t`` of ``unit``, or of no unit if ``unit``
        is ``None``."""
        times = FloatTensor(np.asarray(t, dtype=np.float64))
        try:
            native = cpp.evaluate_series(self._data, times._data, unit)
        except cpp.TimeSeriesDomainError as error:
            raise self._domain_error(error) from None
        return self._leaves(native)

    @staticmethod
    def _leaves(native):
        """The values of a C++ evaluation: the tensor of the one channel, or
        a ``NestedTensor`` of one per channel."""
        values = NestedTensor._from_cpp(native)
        if values.shape[0] > 1:
            return values
        leaf = values[0]
        return leaf[()] if leaf.ndim == 0 else leaf

