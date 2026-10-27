from __future__ import annotations

import operator
from typing import Generic, NamedTuple, TypeVar

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
from .date_time import _NUMPY_TO_UNIT, DateTime
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


def _times_from_datetimes(stamps, unit, epoch):
    """The native times of ``datetime64`` stamps counted from ``epoch``, or
    from the first stamp, in ``unit``."""
    if epoch is None:
        if stamps.size == 0:
            raise ValueError("Pass an epoch for an empty array of dates and times")
        epoch = stamps.reshape(-1)[0]
    epoch = DateTime(epoch)
    # Integer differences at the finer of the two resolutions; Times.to then
    # converts them to unit by one exact factor.
    offsets = stamps - epoch.to_datetime64()
    if np.isnat(offsets).any():
        raise ValueError("Dates and times cannot be NaT")
    code, step = np.datetime_data(offsets.dtype)
    if code not in _NUMPY_TO_UNIT:
        raise ValueError(f"Dates and times need a resolution of fixed length, not {code!r}")
    counts = offsets.astype(np.int64).astype(np.float64) * step
    return cpp.Times(counts, _NUMPY_TO_UNIT[code], epoch._data).to(unit)


class Times:
    """Finite, strictly increasing timestamps in one physical time unit.

    ``np.asarray(times)`` gives the timestamps as a new float64 array and
    ``sb.tensor(times)`` as a ``FloatTensor``; changing either leaves the
    timestamps unchanged.

    Parameters
    ----------
    times : array_like
        One-dimensional timestamps, converted to float64, or a ``datetime64``
        array of dates and times. Dates and times are counted from
        ``epoch``, which defaults to the first of them, and converted to
        ``unit``: the differences are taken in integers and converted by one
        exact factor, so whole multiples of ``unit`` are exact.
    unit : str, optional
        One of ``"as"``, ``"fs"``, ``"ps"``, ``"ns"``, ``"us"``, ``"ms"``,
        ``"s"``, ``"min"``, ``"h"``, and ``"d"``, or an alias such as
        ``"sec"``, ``"seconds"``, ``"hr"``, ``"\u00b5s"``, or ``"msec"``.
        Defaults to seconds. :attr:`unit` reports the short name.
    epoch : DateTime or a value DateTime accepts, optional
        The date and time that time 0 stands for, such as
        ``"2023-01-01"``. Defaults to none: the timestamps are durations.

    Raises
    ------
    ValueError
        If the timestamps are not one-dimensional, finite, and strictly
        increasing, or if the unit is unknown.
    """

    def __init__(self, times, unit: str = "s", *, epoch=None):
        times = np.asarray(times)
        if np.issubdtype(times.dtype, np.datetime64):
            self._data = _times_from_datetimes(times, unit, epoch)
            return
        if epoch is not None:
            epoch = DateTime(epoch)._data
        self._data = cpp.Times(np.asarray(times, dtype=np.float64), unit, epoch)

    @classmethod
    def _from_cpp(cls, data):
        times = cls.__new__(cls)
        times._data = data
        return times

    @property
    def unit(self) -> str:
        """The time unit, such as ``"s"`` or ``"ms"``."""
        return self._data.unit

    @property
    def epoch(self) -> DateTime | None:
        """The date and time that time 0 stands for, or ``None``."""
        data = self._data.epoch
        return None if data is None else DateTime._from_cpp(data)

    def with_epoch(self, epoch) -> Times:
        """Return the same timestamps counted from ``epoch``.

        ``epoch`` is a :class:`DateTime` or a value it accepts, or ``None``
        to remove the epoch. No timestamp is shifted.
        """
        if epoch is not None:
            epoch = DateTime(epoch)._data
        return Times._from_cpp(self._data.with_epoch(epoch))

    def to(self, unit: str) -> Times:
        """Return the same timestamps in another unit, from the same epoch.

        The conversion multiplies or divides each timestamp by an exact
        integer factor, so each result is rounded once. Only the factor
        between days and attoseconds exceeds 2^53 and is itself rounded.
        """
        return Times._from_cpp(self._data.to(unit))

    def __array__(self, dtype=None, copy=None):
        """Return a float64 copy of the timestamps, so ``np.asarray(times)`` works."""
        arr = np.array(self._data.values)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other):
        if not isinstance(other, Times):
            return NotImplemented
        return self._data == other._data

    def __repr__(self):
        epoch = "" if self.epoch is None else f", epoch={self.epoch!r}"
        return f"Times({np.asarray(self).tolist()!r}, unit={self.unit!r}{epoch})"


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


class Domain(NamedTuple):
    """The times at which a series can be evaluated, from ``start`` to ``end``.

    ``start_closed`` and ``end_closed`` say whether ``start`` and ``end``
    themselves are included.
    """

    #: The first time in the domain, or its lower bound if not included.
    start: float
    #: The last time in the domain, or its upper bound if not included.
    end: float
    #: Whether ``start`` is in the domain.
    start_closed: bool
    #: Whether ``end`` is in the domain.
    end_closed: bool


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
    observation, both included. A window (:class:`Window`) runs from its
    start to its end, which may exclude its start and extend past its last
    observation, and a delay embedding (:class:`TimeDelayEmbedding`) keeps
    the ends of its channel's domain. A series can be evaluated where all
    of its channels can; :attr:`domain` gives these times.

    Element-wise operations, such as
    :func:`~stablebear.persistence.compute_persistent_homology`, accept a
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
    times : Times or array_like
        One timestamp per observation. A plain array is in seconds.
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

    def __init__(self, values, times, *, snap_tolerance: float = 1e-14):
        if not isinstance(times, Times):
            times = Times(times)
        tensor = _values_tensor(values)
        self._data = _SERIES_NATIVE[tensor.dtype](
            times._data, tensor._data, snap_tolerance
        )

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

        Each channel keeps its own times, unit, and ``snap_tolerance``;
        queries are converted from the series' unit into each channel's
        unit.

        Parameters
        ----------
        channels : list of TimeSeries
            One-channel series of one element type; channel ``i`` of the
            result is ``channels[i]``.
        unit : str, optional
            The unit of queries and reported times. Defaults to the unit
            the channels share; required when their units differ.

        Raises
        ------
        ValueError
            If there are no channels, a channel has several channels,
            ``unit`` is omitted while the channels' units differ, or the
            channels do not share their :attr:`epoch`: all the same, at the
            same resolution, or none. To combine dated channels, count each
            one's times from one shared epoch with ``Times(..., epoch=...)``.
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
            [part._data for part in parts], unit
        ))

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
    def unit(self) -> str:
        """The time unit of queries and reported times."""
        return self._data.unit

    @property
    def dtype(self):
        """The element dtype shared by all channels."""
        return _SERIES_CPP_TO_DTYPE[type(self._data)]

    @property
    def times(self) -> Times | NestedTensor:
        """The timestamps of the recorded observations, in :attr:`unit`.

        A :class:`Times` for a series with one channel. For several
        channels, a float64 ``NestedTensor`` with one leaf per channel, in
        channel order, each holding that channel's timestamps converted
        to :attr:`unit`.
        """
        if self._data.n_channels == 1:
            return Times._from_cpp(self._data.times(0))
        return NestedTensor([
            np.asarray(Times._from_cpp(self._data.times(i)))
            for i in range(self._data.n_channels)
        ])

    @property
    def values(self) -> Tensor | NestedTensor:
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
    def shape(self) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
        """The shape of one observation; ``()`` for scalars.

        For several channels, a tuple with one shape per channel, in
        channel order.
        """
        shapes = tuple(self._data.channel(i).shape for i in range(self.n_channels))
        return shapes[0] if len(shapes) == 1 else shapes

    @property
    def origin(self) -> float | FloatTensor:
        """The time in the series a channel was cut from that its time 0
        stands for, in :attr:`unit`.

        0 for a series built from observations. Evaluation does not use it;
        it places a window in the series it was cut from. A float for a
        series with one channel; for several, a float64 ``FloatTensor`` with
        one entry per channel.
        """
        origins = [self._data.origin(i) for i in range(self.n_channels)]
        if len(origins) == 1:
            return origins[0]
        return FloatTensor(origins, dtype=float64)

    @property
    def epoch(self) -> DateTime | None:
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
        """
        if epoch is not None:
            epoch = DateTime(epoch)._data
        return TimeSeries._from_cpp(self._data.with_epoch(epoch))

    @property
    def domain(self) -> Domain:
        """The times at which every channel can be evaluated, in :attr:`unit`.

        For several channels, the intersection of their domains, which
        :meth:`evaluate` checks queries against.

        Raises
        ------
        ValueError
            If a channel has no observations.
        """
        return Domain(*self._data.domain)

    @property
    def n_observations(self) -> int | IntTensor:
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
    def snap_tolerance(self) -> float | FloatTensor:
        """The snap tolerance used when evaluating; see the constructor.

        A float for a series with one channel; for several, a float64
        ``FloatTensor`` with one entry per channel.
        """
        return self._per_channel("snap_tolerance")

    def to(self, unit: str) -> TimeSeries:
        """Return the same series with queries and reported times in ``unit``.

        The stored observations and timestamps are shared, not converted;
        :attr:`times` are reported in the new unit, and queries are read in
        it. Accepts the units of :class:`Times`.
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
        channel's own ``snap_tolerance``; see the constructor. ``t`` is in ``unit``, which defaults to the series'
        unit and accepts the units of :class:`Times`, and must lie in the
        domain of every channel; see :class:`TimeSeries`.

        ``t`` is a single time or an array-like or ``FloatTensor`` of times
        of shape ``t_shape``. Many times are evaluated together: they are
        sorted once and merged with each channel's timestamps.

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
        """
        if isinstance(t, (int, float, np.integer, np.floating)):
            values = NestedTensor._from_cpp(self._data.evaluate(float(t), unit))
            if self.n_channels > 1:
                return values
            leaf = values[0]
            return leaf[()] if leaf.ndim == 0 else leaf
        times = FloatTensor(np.asarray(t, dtype=np.float64))
        values = NestedTensor._from_cpp(self._data.evaluate(times._data, unit))
        return values if self.n_channels > 1 else values[0]

    def __call__(self, t, unit: str | None = None):
        """``series(t, unit)`` is ``series.evaluate(t, unit)``."""
        return self.evaluate(t, unit)

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
        :meth:`TimeSeries.evaluate` does. ``t`` is in ``unit`` for every
        series, or in each series' own unit if ``unit`` is omitted.

        ``t`` is a single time or an array-like or ``FloatTensor`` of times
        of shape ``t_shape``. For series with one channel, returns a tensor
        of shape ``(*self.shape, *t_shape, *shape)``, as evaluating a
        ``PcfTensor`` gives ``(*self.shape, *t_shape)``. For several
        channels, returns a ``NestedTensor`` with one such tensor per
        channel, in channel order, each with that channel's ``shape``. For
        windows of a delay embedding of a scalar series evaluated on a grid
        of ``n`` times, the last two axes are clouds of ``n`` points, which
        :func:`~stablebear.persistence.compute_persistent_homology` reads
        as point clouds. Observations of shape ``S``, embedded with
        dimension ``m``, give the axes ``(n, m, *S)`` instead; reshape them
        to ``(n, m * prod(S))`` to get clouds of ``n`` points.

        Raises
        ------
        ValueError
            If a time is outside the domain of a series, or the tensor is
            empty or its series differ in channels or shapes, or if
            ``unit`` is unknown.
        """
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
