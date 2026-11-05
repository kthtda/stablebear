from __future__ import annotations

import datetime
import operator
import sys
from typing import Any, Generic, NamedTuple, Protocol, TypeVar

import numpy as np

from .. import _sb_cpp as cpp
from .._binary_io import _BinaryIoMixin
from .._tensor_base import IndexedElementTensor, Tensor
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
    series,
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


def _times_from_datetimes(stamps, unit, epoch):
    """The times of ``datetime64`` stamps counted from ``epoch``, or from the
    first stamp, in ``unit``, as :func:`_native_times` gives them."""
    if epoch is None:
        if stamps.size == 0:
            raise ValueError("Pass an epoch for an empty array of dates and times")
        epoch = stamps.reshape(-1)[0]
    epoch = DateTime(epoch)
    # Integer differences at the finer of the two resolutions, then converted
    # to unit by one exact factor.
    offsets = stamps - epoch.to_datetime64()
    if np.isnat(offsets).any():
        raise ValueError("Dates and times cannot be NaT")
    code, step = np.datetime_data(offsets.dtype)
    if code not in _NUMPY_TO_UNIT:
        raise ValueError(f"Dates and times need a resolution of fixed length, not {code!r}")
    counts = offsets.astype(np.int64).astype(np.float64) * step
    unit = _unit_name(unit)
    return _in_unit(TimeDeltaTensor(FloatTensor(counts), _NUMPY_TO_UNIT[code]), unit), unit, epoch._data


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
    if isinstance(t, (bool, np.bool_, Tensor, str)) or _as_time_delta(t) is not None:
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
    getattr(cpp, f"TimeSeries{suffix}Tensor"): series(element)
    for element, suffix in _NATIVE_SUFFIX.items()
}


def _is_stablebear_object(value):
    from ..persistence.barcode import Barcode

    return isinstance(
        value, (Barcode, DistanceMatrix, Pcf, PointCloud, SymmetricMatrix, Tensor)
    )


def _on_one_epoch(parts):
    """``parts``, one-channel series, counted from the earliest of their
    epochs if they all have one."""
    epochs = [part.epoch for part in parts]
    if None in epochs or len(set(epochs)) == 1:
        return parts
    earliest = min(epochs)
    return [part if part.epoch == earliest
            else TimeSeries._from_cpp(part._data.shifted(_in_unit(part.epoch - earliest, part.unit), earliest._data))
            for part in parts]


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


class TimeSeries(_BinaryIoMixin, Generic[T]):
    """Observations recorded at known times, in one or more channels.

    A series built from ``values`` and ``times`` has one channel;
    :meth:`from_channels` combines several. Series can be saved with
    ``sb.save``, loaded with ``sb.load``, and pickled; each channel keeps its
    times, unit, and snap settings.

    A series cannot be changed. The constructor copies ``values``, and
    :attr:`values` and :meth:`evaluate` return copies, so changing the
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
        ``datetime64`` array, or a pandas ``DatetimeIndex`` or ``Series`` of
        dates, whose timezone, if any, is converted to UTC and dropped; they
        are counted from ``epoch``, which defaults to the first of them. The
        :attr:`unit` is that of the durations or dates: the unit of a
        ``TimeDeltaTensor``, the resolution of an array or index, such as
        minutes for a ``timedelta64[m]`` array and nanoseconds for most
        pandas indexes, or the finest unit of a sequence of durations. A
        plain array of numbers gives a series without a unit, whose
        :attr:`unit` is ``None``.
    epoch : DateTime or a value DateTime accepts, optional
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
            [part._data for part in _on_one_epoch(parts)], unit
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
            resolution of a ``DatetimeIndex`` or ``TimedeltaIndex``, usually
            nanoseconds, and to none for a numeric index.
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
        as :meth:`evaluate` gives it, and ``NaN`` outside the channel's
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
        placed = np.unique(np.concatenate([np.asarray(self._data.times(i)) + offset
                                           for i, offset in enumerate(offsets)]))
        index = self._pandas_index(pd, placed, resolution)
        columns = {}
        for i, offset in enumerate(offsets):
            # The value in effect at every time of the union in the channel's
            # domain, and NaN outside it.
            channel = self[i]
            t = placed - offset
            start, end, start_closed, end_closed = channel._data.domain
            inside = ((t > start) | (start_closed & (t == start))) & ((t < end) | (end_closed & (t == end)))
            values = np.asarray(channel._evaluate_numbers(FloatTensor(t[inside]))) if inside.any() else []
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

        A tensor of shape ``(len(series), *shape)`` for a series with one
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
        intersection of theirs, which :meth:`evaluate` checks queries
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

        An int for a series with one channel, the same as ``len(series)``;
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

    def evaluate(self, t, unit: str | None = None):
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
        :class:`DateTimeTensor`, an array, a pandas ``DatetimeIndex``, or a
        pandas ``Series`` of them. A date names an instant, not an offset:
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
        ValueError
            If a time is outside the domain of any channel, or a channel has
            no observations, or if ``unit`` is unknown.
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
        if isinstance(t, (str, datetime.date, np.datetime64, DateTime)):
            return self._evaluate_date(DateTime(t).to_datetime64())
        if isinstance(t, DateTimeTensor):
            return self._evaluate_date(t.to_datetime64())
        duration = _as_time_delta(t)
        if duration is not None:
            return self._evaluate_numbers(_in_unit(duration, self.unit))
        if not isinstance(t, Tensor):
            # Lists, arrays, and pandas objects, such as a DatetimeIndex.
            t = _as_array(t)
            durations = _durations_in(t, self.unit)
            if durations is not None:
                return self._evaluate_numbers(FloatTensor(np.asarray(durations, dtype=np.float64)))
        if isinstance(t, np.ndarray) and t.dtype.kind in "MOUS":
            # Dates and times, as datetime64, strings, or datetimes.
            try:
                dates = np.asarray(t, dtype="datetime64")
            except (TypeError, ValueError):
                raise TypeError("Times must be all durations or all dates and times, not a mix") from None
            return self._evaluate_date(_checked_dates(dates))
        raise TypeError(f"Expected durations or dates and times, not {type(t).__name__}")

    def _evaluate_numbers(self, t):
        """Evaluate at ``t``, a float or an array or ``FloatTensor`` of times
        in :attr:`unit`."""
        if np.ndim(t) > 0:
            t = FloatTensor(np.asarray(t, dtype=np.float64))
        if isinstance(t, FloatTensor):
            values = NestedTensor._from_cpp(self._data.evaluate(t._data))
            return values if self.n_channels > 1 else values[0]
        values = NestedTensor._from_cpp(self._data.evaluate(float(t)))
        if self.n_channels > 1:
            return values
        leaf = values[0]
        return leaf[()] if leaf.ndim == 0 else leaf

    def __call__(self, t, unit: str | None = None):
        """``series(t, unit)`` is ``series.evaluate(t, unit)``."""
        return self.evaluate(t, unit)

    def _time_of_date(self, date, channel):
        """The time of ``date`` in channel ``channel``, in :attr:`unit`:
        ``date - epoch - origin``, with the difference taken in integers."""
        epoch, unit = self.epoch, self.unit
        assert epoch is not None and unit is not None
        offset = date - epoch.to_datetime64()
        return offset / np.timedelta64(1, _UNIT_TO_NUMPY[unit]) - self._data.origin(channel)

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
                + TimeDeltaTensor(FloatTensor([start + origin, end + origin]), self.unit).to_timedelta64("us"))
        # Formatted at one precision, the coarsest that shows all three.
        dates = np.array([first, *ends], dtype=ends.dtype)
        precision = next(code for code in ("m", "s", "ms", "us") if (dates == dates.astype(f"datetime64[{code}]")).all())
        first, start, end = np.datetime_as_string(dates, unit=precision)  # pyright: ignore[reportCallIssue, reportArgumentType]
        domain = ("[" if start_closed else "(") + f"{start}, {end}" + ("]" if end_closed else ")")
        raise ValueError(f"{first} is outside the domain {domain}")

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

    def __len__(self) -> int:
        """The number of recorded observations of a one-channel series.

        Raises
        ------
        ValueError
            On several channels; use :attr:`n_observations`.
        """
        return len(self._single_channel())

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


class TimeSeriesTensor(Tensor):
    """Tensor whose elements are :class:`TimeSeries`, such as the windows of
    a series. Its dtype is ``sb.series(inner)`` for series of ``inner``
    observations."""

    def __init__(self, data):
        super().__init__()
        if isinstance(data, TimeSeriesTensor):
            data = data._data
        elif type(data) not in _SERIES_TENSOR_TO_DTYPE:
            raise TypeError(f"Cannot create TimeSeriesTensor from {type(data)}")
        self._data = data
        self.dtype = _SERIES_TENSOR_TO_DTYPE[type(data)]

    def _to_py_tensor(self, data):
        return TimeSeriesTensor(data)

    def _decay_value(self, val):
        return val._data

    def _represent_element(self, element):
        return TimeSeries._from_cpp(element)

    def _get_valid_setitem_dtypes(self):
        return [TimeSeries, TimeSeriesTensor]

    def evaluate(self, t, unit: str | None = None):
        """Every series evaluated at the times ``t``, as
        :meth:`~stablebear.timeseries.TimeSeries.evaluate` does: durations, or
        numbers of ``unit``, for series with a unit, and plain numbers without
        ``unit`` for series without one. Times count from each series' time 0.

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
        ValueError
            If a time is outside the domain of a series, or the tensor is
            empty or its series differ in channels or shapes, or if
            ``unit`` is unknown.
        TypeError
            For numbers without ``unit`` on series with a unit, or for
            durations or ``unit`` on series without one.
        """
        if unit is not None:
            t = _with_unit(t, unit)
        # Durations are queried in their own unit, and C++ converts them into
        # the unit of each series.
        duration = _as_time_delta(t)
        if duration is not None:
            return self._evaluate(_in_unit(duration, duration.unit), duration.unit)
        if not isinstance(t, (Tensor, int, float, np.integer, np.floating)):
            t = _as_array(t)
            finest = _finest_duration_unit(t)
            if finest is not None:
                return self._evaluate(_durations_in(t, finest), finest)
        return self._evaluate(t, None)

    def _evaluate(self, t, unit):
        """Evaluate at numbers ``t`` of ``unit``, or of no unit if ``unit``
        is ``None``."""
        times = FloatTensor(np.asarray(t, dtype=np.float64))
        values = NestedTensor._from_cpp(
            cpp.evaluate_series(self._data, times._data, unit))
        if values.shape[0] > 1:
            return values
        leaf = values[0]
        return leaf[()] if leaf.ndim == 0 else leaf

    def __call__(self, t, unit: str | None = None):
        """``windows(t, unit)`` is ``windows.evaluate(t, unit)``."""
        return self.evaluate(t, unit)
