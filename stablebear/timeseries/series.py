from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np

from .. import _sb_cpp as cpp
from .._tensor_base import Tensor
from ..base_tensor import BoolTensor, FloatTensor, IntTensor
from ..typing import boolean, float32, float64, int32, int64, uint32, uint64

T = TypeVar("T")


class Times:
    """Finite, strictly increasing timestamps in one physical time unit.

    Parameters
    ----------
    values : array_like
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

    def __init__(self, values, unit: str = "s"):
        self._data = cpp.Times(np.asarray(values, dtype=np.float64), unit)

    @classmethod
    def _from_cpp(cls, data):
        times = cls.__new__(cls)
        times._data = data
        return times

    @property
    def values(self) -> FloatTensor:
        """A float64 copy of the timestamps; changing it leaves these unchanged."""
        return FloatTensor(self._data.values)

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

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other):
        if not isinstance(other, Times):
            return NotImplemented
        return self._data == other._data

    def __repr__(self):
        return f"Times({np.asarray(self.values).tolist()!r}, unit={self.unit!r})"


# Element dtype -> (native series class, tensor wrapper for its values).
_SERIES_TYPES = {
    boolean: (cpp.TimeSeriesBool, BoolTensor),
    float32: (cpp.TimeSeriesFloat32, FloatTensor),
    float64: (cpp.TimeSeriesFloat64, FloatTensor),
    int32: (cpp.TimeSeriesInt32, IntTensor),
    int64: (cpp.TimeSeriesInt64, IntTensor),
    uint32: (cpp.TimeSeriesUint32, IntTensor),
    uint64: (cpp.TimeSeriesUint64, IntTensor),
}
_SERIES_CPP_TO_DTYPE = {cls: dtype for dtype, (cls, _) in _SERIES_TYPES.items()}


def _values_tensor(values):
    """Convert values to a tensor whose axis 0 is time."""
    from ..tensor_create import tensor

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
    if values.dtype not in _SERIES_TYPES:
        raise TypeError(f"Unsupported time series values: {type(values).__name__}")
    return values


class TimeSeries(Generic[T]):
    """Observations recorded at known times, in one or more channels.

    Parameters
    ----------
    values : array_like or FloatTensor
        Observations with time along axis 0. Each observation has shape
        ``values.shape[1:]``. Arrays get the dtype ``sb.tensor`` would give
        them: float32, int32, uint32, uint64, and bool are kept, other
        floats become float64, and other integers become int64.
    times : Times or array_like
        One timestamp per observation. A plain array is in seconds.

    A series built this way has one channel, named ``"0"``.
    """

    def __init__(self, values, times=None):
        if times is None:
            raise ValueError("times are required")
        if not isinstance(times, Times):
            times = Times(times)
        tensor = _values_tensor(values)
        self._data = _SERIES_TYPES[tensor.dtype][0](times._data, tensor._data)

    def _single_channel(self):
        if self._data.n_channels != 1:
            raise ValueError(
                "This series has several channels; select one with series[name]"
            )
        return self._data.channel(0)

    @property
    def channel_names(self) -> tuple[str, ...]:
        """The channel names, in order."""
        return self._data.channel_names

    @property
    def unit(self) -> str:
        """The time unit of queries and reported times."""
        return self._data.unit

    @property
    def dtype(self):
        """The element dtype shared by all channels."""
        return _SERIES_CPP_TO_DTYPE[type(self._data)]

    @property
    def times(self) -> Times:
        """The timestamps of the recorded observations."""
        return Times._from_cpp(self._single_channel().times)

    @property
    def values(self) -> Tensor:
        """A copy of the observations, with time along axis 0."""
        wrapper = _SERIES_TYPES[self.dtype][1]
        return wrapper(self._single_channel().values)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of one observation; ``()`` for scalars."""
        return self._single_channel().shape

    def __len__(self) -> int:
        return len(self._single_channel())

    def __repr__(self):
        return (
            f"TimeSeries(channels={self.channel_names!r}, dtype={self.dtype}, "
            f"unit={self.unit!r})"
        )
