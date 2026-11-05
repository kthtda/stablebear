from __future__ import annotations

from typing import overload

from .. import _sb_cpp as cpp
from ..point_cloud import PointCloud, PointCloudTensor
from ..typing import float32, float64, int32, int64, uint32, uint64
from .series import TimeSeries, TimeSeriesTensor, _PandasLike, _as_series

_NUMBER_DTYPES = (float32, float64, int32, int64, uint32, uint64)


@overload
def to_point_clouds(series: TimeSeries | _PandasLike) -> PointCloud: ...


@overload
def to_point_clouds(series: TimeSeriesTensor) -> PointCloudTensor: ...


def to_point_clouds(series: TimeSeries | TimeSeriesTensor | _PandasLike) -> PointCloud | PointCloudTensor:
    """The point clouds traced by a time series or by each series of a
    tensor of them, such as windows.

    Each observation the series stores at a time in its
    :attr:`~TimeSeries.domain` gives one point, in the order of their
    times: an observation of shape ``()`` a point of dimension 1, and one
    of shape ``(m,)`` a point of dimension ``m``. A window (:func:`cut_windows`) stores the value in
    effect at its start at time 0, so that value is a point if the window
    includes its start, and is left out if the start is open. A
    ``float32`` series gives ``pcloud32`` clouds, and series of other
    numbers ``pcloud64`` clouds. A pandas ``Series`` or ``DataFrame`` is
    converted with :meth:`TimeSeries.from_pandas` first.

    Parameters
    ----------
    series : TimeSeries, TimeSeriesTensor, pandas.Series, or pandas.DataFrame
        Series of one channel of numbers. Combine several channels with
        :func:`~stablebear.tensor_create.stack` or
        :func:`~stablebear.tensor_create.concatenate`, or select one with
        ``series[i]``.

    Returns
    -------
    PointCloud or PointCloudTensor
        The cloud of a series, or for a tensor of series a tensor of the
        same shape holding the cloud of each, whose clouds can differ in
        size.

    Raises
    ------
    ValueError
        If a series has several channels.
    TypeError
        If the observations are not numbers, are booleans, or have more
        than one dimension, or for an object that is not a series or a
        pandas ``Series`` or ``DataFrame``.
    """
    if isinstance(series, TimeSeriesTensor):
        inner = series.dtype.inner
        _check_numbers(inner)
        return PointCloudTensor(cpp.to_point_clouds(series._data))
    series = _as_series(series)
    _check_numbers(series.dtype)
    return PointCloud(cpp.to_point_cloud(series._data))


def _check_numbers(dtype):
    if dtype not in _NUMBER_DTYPES:
        raise TypeError(f"Point clouds come from series of numbers, not of {dtype}")
