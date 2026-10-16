from __future__ import annotations

import operator
from typing import Generic, TypeVar

import numpy as np

from .. import _sb_cpp as cpp
from .._binary_io import _BinaryIoMixin
from .._tensor_base import IndexedElementTensor, Tensor
from ..base_tensor import BoolTensor, FloatTensor, IntPcfTensor, IntTensor, PcfTensor
from ..distance_matrix import DistanceMatrix, DistanceMatrixTensor
from ..functional.pcf import Pcf
from ..nested_tensor import NestedTensor
from ..persistence.barcode import Barcode
from ..persistence.ph_tensor import BarcodeTensor
from ..point_cloud import PointCloud, PointCloudTensor
from ..symmetric_matrix import SymmetricMatrix, SymmetricMatrixTensor
from ..typing import (
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
    symmat32,
    symmat64,
    uint32,
    uint64,
)

T = TypeVar("T")


class Times:
    """Finite, strictly increasing timestamps in one physical time unit.

    ``np.asarray(times)`` gives the timestamps as a new float64 array and
    ``sb.tensor(times)`` as a ``FloatTensor``; changing either leaves the
    timestamps unchanged.

    Parameters
    ----------
    times : array_like
        One-dimensional timestamps, converted to float64.
    unit : str, optional
        One of ``"as"``, ``"fs"``, ``"ps"``, ``"ns"``, ``"us"``, ``"ms"``,
        ``"s"``, ``"min"``, ``"h"``, and ``"d"``, or an alias such as
        ``"sec"``, ``"seconds"``, ``"hr"``, ``"\u00b5s"``, or ``"msec"``.
        Defaults to seconds. :attr:`unit` reports the short name.

    Raises
    ------
    ValueError
        If the timestamps are not one-dimensional, finite, and strictly
        increasing, or if the unit is unknown.
    """

    def __init__(self, times, unit: str = "s"):
        self._data = cpp.Times(np.asarray(times, dtype=np.float64), unit)

    @classmethod
    def _from_cpp(cls, data):
        times = cls.__new__(cls)
        times._data = data
        return times

    @property
    def unit(self) -> str:
        """The time unit, such as ``"s"`` or ``"ms"``."""
        return self._data.unit

    def to(self, unit: str) -> Times:
        """Return the same timestamps in another unit.

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
        return f"Times({np.asarray(self).tolist()!r}, unit={self.unit!r})"


# Element dtype -> (native series class, tensor wrapper for its values).
_SERIES_TYPES = {
    boolean: (cpp.TimeSeriesBool, BoolTensor),
    float32: (cpp.TimeSeriesFloat32, FloatTensor),
    float64: (cpp.TimeSeriesFloat64, FloatTensor),
    int32: (cpp.TimeSeriesInt32, IntTensor),
    int64: (cpp.TimeSeriesInt64, IntTensor),
    uint32: (cpp.TimeSeriesUint32, IntTensor),
    uint64: (cpp.TimeSeriesUint64, IntTensor),
    pcf32: (cpp.TimeSeriesPcf32, PcfTensor),
    pcf64: (cpp.TimeSeriesPcf64, PcfTensor),
    pcf32i: (cpp.TimeSeriesPcf32i, IntPcfTensor),
    pcf64i: (cpp.TimeSeriesPcf64i, IntPcfTensor),
    pcloud32: (cpp.TimeSeriesPointCloud32, PointCloudTensor),
    pcloud64: (cpp.TimeSeriesPointCloud64, PointCloudTensor),
    distmat32: (cpp.TimeSeriesDistanceMatrix32, DistanceMatrixTensor),
    distmat64: (cpp.TimeSeriesDistanceMatrix64, DistanceMatrixTensor),
    symmat32: (cpp.TimeSeriesSymmetricMatrix32, SymmetricMatrixTensor),
    symmat64: (cpp.TimeSeriesSymmetricMatrix64, SymmetricMatrixTensor),
    barcode32: (cpp.TimeSeriesBarcode32, BarcodeTensor),
    barcode64: (cpp.TimeSeriesBarcode64, BarcodeTensor),
}
_SERIES_CPP_TO_DTYPE = {cls: dtype for dtype, (cls, _) in _SERIES_TYPES.items()}


_STABLEBEAR_OBJECTS = (
    Barcode, DistanceMatrix, Pcf, PointCloud, SymmetricMatrix, Tensor,
)


def _values_tensor(values):
    """Convert values to a tensor whose axis 0 is time."""
    from ..tensor_create import tensor

    if isinstance(values, (list, tuple)) and any(
        isinstance(value, _STABLEBEAR_OBJECTS) for value in values
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
    if values.dtype not in _SERIES_TYPES or not isinstance(
        values, _SERIES_TYPES[values.dtype][1]
    ):
        raise TypeError(f"Unsupported time series values: {type(values).__name__}")
    if isinstance(values, IndexedElementTensor) and values._has_indexed_storage():
        values = values.to_dense()
    return values


class TimeSeries(_BinaryIoMixin, Generic[T]):
    """Observations recorded at known times, in one or more channels.

    A series built from ``values`` and ``times`` has one channel;
    :meth:`from_channels` combines several. Series can be saved with
    ``sb.save``, loaded with ``sb.load``, and pickled; each channel keeps its
    times, unit, and snap settings.

    Parameters
    ----------
    values : array_like or FloatTensor
        Observations with time along axis 0. Each observation has shape
        ``values.shape[1:]``. Arrays get the dtype ``sb.tensor`` would give
        them: float32, int32, uint32, uint64, and bool are kept, other
        floats become float64, and other integers become int64.
    times : Times or array_like
        One timestamp per observation. A plain array is in seconds.
    snap_tolerance : float, optional
        How close below an observation a query may land and still get that
        observation, as a fraction of the gap to the previous observation.
        This lets a time computed in floating point, such as ``0.3 - 0.1``,
        reach the observation it means. Must be in ``[0, 1)``; 0 disables
        snapping. Defaults to ``1e-6``.
    snap_eps : float, optional
        A cap on the same distance, in multiples of machine epsilon
        (``np.finfo(np.float64).eps``, about ``2.2e-16``) times the
        magnitude of the timestamps. This keeps widely spaced observations
        from capturing queries that are clearly earlier. Must be finite
        and nonnegative; 0 disables snapping. Defaults to ``1024``.

    Notes
    -----
    Let tol be ``snap_tolerance``, k be ``snap_eps``, and
    eps be machine epsilon, and let the timestamps be
    t[0] < t[1] < ... < t[n-1], with observations x[0], x[1], ..., x[n-1].
    At a time t >= t[0], the series is x[i] for the largest i such that
    i = 0 or ::

        t >= t[i] - min(tol * (t[i] - t[i-1]),
                        k * eps * max(|t[i-1]|, |t[i]|)).

    With either parameter 0, this is the last observation at or before t.
    Since tol < 1, the shifted thresholds stay strictly increasing, so no
    observation is skipped.
    """

    def __init__(self, values, times, *, snap_tolerance: float = 1e-6,
                 snap_eps: float = 1024):
        if not isinstance(times, Times):
            times = Times(times)
        tensor = _values_tensor(values)
        self._data = _SERIES_TYPES[tensor.dtype][0](
            times._data, tensor._data, snap_tolerance, snap_eps
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

        Each channel keeps its own times, unit, ``snap_tolerance``, and
        ``snap_eps``; queries are converted from the series' unit into each
        channel's unit.

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
            If there are no channels, a channel has several channels, or
            ``unit`` is omitted while the channels' units differ.
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
        native = _SERIES_TYPES[parts[0].dtype][0]
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
        wrapper = _SERIES_TYPES[self.dtype][1]
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

    @property
    def snap_eps(self) -> float | FloatTensor:
        """The cap on snapping used when evaluating; see the constructor.

        A float for a series with one channel; for several, a float64
        ``FloatTensor`` with one entry per channel.
        """
        return self._per_channel("snap_eps")

    def to(self, unit: str) -> TimeSeries:
        """Return the same series with queries and reported times in ``unit``.

        The stored observations and timestamps are shared, not converted;
        :attr:`times` are reported in the new unit, and queries are read in
        it. Accepts the units of :class:`Times`.
        """
        return TimeSeries._from_cpp(self._data.to(unit))

    def evaluate(self, t):
        """Each channel's observation with the largest timestamp at or before ``t``.

        A ``t`` just below a timestamp snaps forward to it, by each
        channel's own ``snap_tolerance`` and ``snap_eps``; see the
        constructor. ``t`` is in the series' unit and must lie between the
        first and last observations of every channel, both included.

        For one channel, returns a scalar or element for observations of
        shape ``()``, otherwise a tensor of that shape. For several, returns
        a ``NestedTensor`` with one leaf per channel, in channel order; each
        leaf is a tensor of that channel's shape, 0-d for shape ``()``.

        Raises
        ------
        ValueError
            If ``t`` is before the first or after the last observation of
            any channel, or a channel has no observations.
        """
        values = NestedTensor._from_cpp(self._data.evaluate(float(t)))
        if self.n_channels > 1:
            return values
        leaf = values[0]
        return leaf[()] if leaf.ndim == 0 else leaf

    def __call__(self, t):
        """``series(t)`` is ``series.evaluate(t)``."""
        return self.evaluate(t)

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
