from __future__ import annotations

import operator
from typing import Any, overload

import numpy as np

from .. import _sb_cpp as cpp
from ..point_cloud import PointCloud, PointCloudTensor
from ..typing import float32, float64, int32, int64, uint32, uint64
from .date_time import DateTimeTensor, TimeDelta, _duration, _in_unit, _ScalarDuration
from .series import (TimeSeries, TimeSeriesTensor, _checked_dates, _date_counts, _domain_error, _PandasLike, _as_series,
                     _queries)

_NUMBER_DTYPES = (float32, float64, int32, int64, uint32, uint64)


@overload
def to_point_clouds(series: TimeSeries | _PandasLike, times: Any = None, *, n: int | None = None,
                    step: float | _ScalarDuration | None = None) -> PointCloud: ...


@overload
def to_point_clouds(series: TimeSeriesTensor, times: Any = None, *, n: int | None = None,
                    step: float | _ScalarDuration | None = None) -> PointCloudTensor: ...


def to_point_clouds(series: TimeSeries | TimeSeriesTensor | _PandasLike, times: Any = None, *,
                    n: int | None = None,
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
    in two clouds. Windows of one size give clouds of ``n`` points each.
    With ``step``, the times are ``start + k * step`` in the domain of each
    series, for ``k = 0, 1, ...``, leaving out ``start`` if the domain
    leaves it out. With ``times``, the times are those, in their order, the
    same for every series, or for dates and times the same instants, so the
    cloud holds the values that evaluating the series at ``times`` gives.

    A ``float32`` series gives ``pcloud32`` clouds, and series of other
    numbers ``pcloud64`` clouds. A pandas ``Series`` or ``DataFrame`` is
    converted with :meth:`TimeSeries.from_pandas` first.

    Parameters
    ----------
    series : TimeSeries, TimeSeriesTensor, pandas.Series, or pandas.DataFrame
        Series of one channel of numbers. Combine several channels with
        :func:`~stablebear.tensor_create.stack` or
        :func:`~stablebear.tensor_create.concatenate`, or select one with
        ``series[i]``.
    times : TimeDeltaTensor, DateTimeTensor, or array_like, optional
        The times of the points, counted from the time 0 of each series,
        as :meth:`TimeSeries.__call__` takes them: durations, such as a
        :class:`TimeDeltaTensor`, for series with a unit, or numbers for
        series without one. Series with an :attr:`~TimeSeries.epoch` also
        take dates and times, such as a :class:`DateTimeTensor`, which each
        series places at ``date - epoch - origin`` in its own time.
    n : int, optional
        The number of evenly spaced points of each cloud.
    step : float, TimeDelta, tuple, numpy.timedelta64, datetime.timedelta, or pandas.Timedelta, optional
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
    TimeSeriesDomainError
        If a time is outside the domain of a series, with the ``index``
        of the first such series of a tensor.
    ValueError
        If a series has several channels, more than one of ``times``,
        ``n``, and ``step`` is given, ``n`` is not positive or is 1 for a
        domain with both ends closed, ``step`` is not finite and positive,
        ``times`` is not one-dimensional, or for dates and times on a
        series without an epoch.
    TypeError
        If the observations are not numbers, are booleans, or have more
        than one dimension, for a ``step`` or ``times`` that are numbers
        on a series with a unit or durations on one without, or for an
        object that is not a series or a pandas ``Series`` or
        ``DataFrame``.
    """
    if isinstance(series, TimeSeriesTensor):
        _check_numbers(series.dtype.value_type)
        n, step, times, dates, unit = _point_times(times, n, step, None)
        counts = None
        if dates is not None:
            counts, unit = _date_counts(dates)
            counts = np.asarray(counts).tolist()
        try:
            return PointCloudTensor(cpp.to_point_clouds(series._data, n, step, times, counts, unit))
        except cpp.TimeSeriesDomainError as error:
            if dates is None:
                raise series._domain_error(error) from None
            raise series._naming_dates(error, dates) from None
    series = _as_series(series)
    _check_numbers(series.dtype)
    # Dates are placed in the series here, so C++ gets times.
    n, step, times, _, unit = _point_times(times, n, step, series)
    try:
        return PointCloud(cpp.to_point_cloud(series._data, n, step, times, None, unit))
    except cpp.TimeSeriesDomainError as error:
        raise _domain_error(error, series.unit, None) from None


def _check_numbers(dtype):
    if dtype not in _NUMBER_DTYPES:
        raise TypeError(f"Point clouds come from series of numbers, not of {dtype}")


def _point_times(times, n, step, series):
    """The ``n``, ``step``, ``times``, ``dates``, and ``unit`` arguments of
    the C++ ``to_point_clouds``, 0 or ``None`` for a mode not chosen. Dates
    and times are placed in ``series`` here, or for a tensor of series, with
    ``series`` ``None``, returned as a ``datetime64`` array, which the caller
    passes as counts for C++ to place in each series."""
    if sum(x is not None for x in (times, n, step)) > 1:
        raise ValueError("Pass at most one of times, n, and step")
    if times is not None:
        # A DateTimeTensor is read as dates before _queries, which would
        # convert it one element at a time.
        if isinstance(times, DateTimeTensor):
            numbers, unit = times.to_datetime64(), None
        else:
            numbers, unit = _queries(times)
        if isinstance(numbers, np.ndarray) and numbers.dtype.kind in "MOUS":
            if series is None:
                dates = _checked_dates(_as_dates(numbers))
                if dates.ndim != 1:
                    raise ValueError(f"The times of the points must be one-dimensional, not of shape {dates.shape}")
                return 0, 0.0, None, dates, None
            numbers, unit = _times_of_dates(numbers, series)
        numbers = np.asarray(numbers, dtype=np.float64)
        if numbers.ndim != 1:
            raise ValueError(f"The times of the points must be one-dimensional, not of shape {numbers.shape}")
        return 0, 0.0, numbers.tolist(), None, unit
    if n is not None:
        n = operator.index(n)
        if n < 1:
            raise ValueError(f"n must be positive, not {n}")
        return n, 0.0, None, None, None
    if step is not None:
        step = _duration(step, "The step must be finite and positive")
        if isinstance(step, TimeDelta):
            return 0, _in_unit(step, step.unit), None, None, step.unit
        return 0, step, None, None, None
    return 0, 0.0, None, None, None


def _times_of_dates(dates, series):
    """``dates``, an array of dates and times, as times of ``series`` and
    its unit."""
    if series.epoch is None:
        raise ValueError("This series has no epoch to place dates and times; set one with with_epoch")
    dates = _checked_dates(_as_dates(dates))
    times = series._time_of_date(dates, 0)
    series._check_date_in_domain(dates, times, 0)
    return times, series.unit


def _as_dates(values):
    """``values``, an array, as ``datetime64``."""
    try:
        return np.asarray(values, dtype="datetime64")
    except (TypeError, ValueError):
        raise TypeError("Times must be all durations or all dates and times, not a mix") from None
