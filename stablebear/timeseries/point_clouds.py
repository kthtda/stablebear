from __future__ import annotations

import operator
from typing import overload

from .. import _sb_cpp as cpp
from ..point_cloud import PointCloud, PointCloudTensor
from ..typing import float32, float64, int32, int64, uint32, uint64
from .date_time import TimeDelta, _duration, _in_unit, _ScalarDuration
from .series import TimeSeries, TimeSeriesTensor, _PandasLike, _as_series

_NUMBER_DTYPES = (float32, float64, int32, int64, uint32, uint64)


@overload
def to_point_clouds(series: TimeSeries | _PandasLike, *, n: int | None = None,
                    step: float | _ScalarDuration | None = None) -> PointCloud: ...


@overload
def to_point_clouds(series: TimeSeriesTensor, *, n: int | None = None,
                    step: float | _ScalarDuration | None = None) -> PointCloudTensor: ...


def to_point_clouds(series: TimeSeries | TimeSeriesTensor | _PandasLike, *, n: int | None = None,
                    step: float | _ScalarDuration | None = None) -> PointCloud | PointCloudTensor:
    """The point clouds traced by a time series or by each series of a
    tensor of them, such as windows.

    The observation in effect at each of a set of times in the
    :attr:`~TimeSeries.domain` gives one point, in the order of the times:
    an observation of shape ``()`` a point of dimension 1, and one of shape
    ``(m,)`` a point of dimension ``m``. The times are, by default, those
    at which the series stores an observation. A window
    (:func:`cut_windows`) stores the value in effect at its start at time
    0, so that value is a point if the window includes its start, and is
    left out if the start is open.

    With ``n``, the times are ``n`` evenly spaced times over the domain of
    each series, including each end the domain includes: with both ends
    closed they are ``length / (n - 1)`` apart, with one end open
    ``length / n`` apart from the closed end, and with both open
    ``length / (n + 1)`` apart; one point lies at the closed end, or in
    the middle if both are open. These are the evenly spaced times over
    the domain with both ends closed, with ``n + 1`` or ``n + 2`` times
    when ends are open and the open ends left out. The times thus depend
    on which ends are closed: ``[0, 2]`` gives times 0, 1, 2, and
    ``[0, 2)`` gives 0, 2/3, 4/3. The clouds of half-open windows that
    follow each other then form one evenly spaced grid, without a time
    in two clouds. Windows of one size give clouds of ``n`` points each. With ``step``, the times are ``start + k * step`` in the
    domain of each series, for ``k = 0, 1, ...``, leaving out ``start`` if
    the domain leaves it out.

    A
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
    n : int, optional
        The number of evenly spaced points of each cloud.
    step : TimeDelta, duration, or float, optional
        The time between points: a duration for a series with a unit, or a
        number for one without, as :func:`cut_windows` takes its ``step``.

    Returns
    -------
    PointCloud or PointCloudTensor
        The cloud of a series, or for a tensor of series a tensor of the
        same shape holding the cloud of each, whose clouds can differ in
        size.

    Raises
    ------
    ValueError
        If a series has several channels, both ``n`` and ``step`` are
        given, ``n`` is not positive or is 1 for a domain with both ends
        closed, or ``step`` is not finite and positive.
    TypeError
        If the observations are not numbers, are booleans, or have more
        than one dimension, for a ``step`` that is a number on a series
        with a unit or a duration on one without, or for an object that is
        not a series or a pandas ``Series`` or ``DataFrame``.
    """
    times = _point_times(n, step)
    if isinstance(series, TimeSeriesTensor):
        value_type = series.dtype.value_type
        _check_numbers(value_type)
        return PointCloudTensor(cpp.to_point_clouds(series._data, *times))
    series = _as_series(series)
    _check_numbers(series.dtype)
    return PointCloud(cpp.to_point_cloud(series._data, *times))


def _check_numbers(dtype):
    if dtype not in _NUMBER_DTYPES:
        raise TypeError(f"Point clouds come from series of numbers, not of {dtype}")


def _point_times(n, step):
    """The ``n``, ``step``, and ``step_unit`` arguments of the C++
    ``to_point_clouds``, 0 for a mode not chosen."""
    if n is not None and step is not None:
        raise ValueError("Pass at most one of n and step")
    if n is not None:
        n = operator.index(n)
        if n < 1:
            raise ValueError(f"n must be positive, not {n}")
        return n, 0.0, None
    if step is not None:
        step = _duration(step, "The step must be finite and positive")
        if isinstance(step, TimeDelta):
            return 0, _in_unit(step, step.unit), step.unit
        return 0, step, None
    return 0, 0.0, None
