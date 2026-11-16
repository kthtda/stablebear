from __future__ import annotations

import operator
from typing import Literal, overload

from .. import _sb_cpp as cpp
from .date_time import TimeDelta, _duration, _in_unit, _ScalarDuration
from .series import TimeSeries, TimeSeriesTensor, _PandasLike, _as_series


@overload
def delay_embed(series: TimeSeries | _PandasLike, dimension: int, delay: float | _ScalarDuration, *,
                direction: Literal["backward", "forward"] = "backward") -> TimeSeries: ...


@overload
def delay_embed(series: TimeSeriesTensor, dimension: int, delay: float | _ScalarDuration, *,
                direction: Literal["backward", "forward"] = "backward") -> TimeSeriesTensor: ...


def delay_embed(series: TimeSeries | TimeSeriesTensor | _PandasLike, dimension: int,
                delay: float | _ScalarDuration, *,
                direction: Literal["backward", "forward"] = "backward") -> TimeSeries | TimeSeriesTensor:
    """Embed a time series, or each series of a tensor of them such as
    windows, in delay coordinates.

    The embedding of a channel ``x`` with ``dimension`` ``m`` and ``delay``
    ``d`` looks back by default: it evaluates at every time ``t`` from
    ``t_0 + (m - 1) * d`` to the end of the channel's domain, its last
    observation or, for a window (:func:`cut_windows`), the window's end, to the
    point ``(x(t - (m - 1) * d), ..., x(t - d), x(t))``. With
    ``direction="forward"``, it evaluates at every time ``t`` from ``t_0`` to
    the end of the domain minus ``(m - 1) * d``, to the point
    ``(x(t), x(t + d), ..., x(t + (m - 1) * d))``. Both are oldest first, so
    the forward point at ``t`` is the backward point at ``t + (m - 1) * d``.
    Each end is included when the channel's is, and each observation gains a
    leading axis of length ``m``. A pandas ``Series`` or ``DataFrame`` is
    converted with :meth:`TimeSeries.from_pandas` first.

    Its times are the start of its domain and the times in it where a
    component reaches an observation, ``t_i + k * d`` for ``k`` from 0 to
    ``m - 1``, or ``t_i - k * d`` looking forward, so it has at most ``m``
    times per observation of the channel. Shifted times that differ only by
    rounding and give the same point appear once, at the time they snap to:
    the channel's own timestamp if one is among them, such as 0.3 rather
    than ``0.2 + 0.1``, else the latest of them. The first time is always
    the start of the domain.

    Each point's components are the channel evaluated at its delayed times,
    with the channel's snapping. Between its times the embedding evaluates
    by previous value, as any series does, with its own snapping. It is a
    view: it shares the series' observations and copies none.

    A :class:`TimeSeriesTensor`, such as the windows :func:`cut_windows`
    gives, is embedded series by series into a tensor of the same shape.
    Each window is embedded on its own, so its embedding never reaches back
    before the window's start, unlike windows cut from an embedding.

    Parameters
    ----------
    series : TimeSeries, TimeSeriesTensor, pandas.Series, or pandas.DataFrame
        The series to embed.
    dimension : int
        The number of components ``m``, at least 1.
    delay : float, TimeDelta, tuple, np.timedelta64, datetime.timedelta, or pd.Timedelta
        The time between components, positive. A duration such as
        ``(3, "h")``, ``TimeDelta(3, "h")``, or ``np.timedelta64(3, "h")`` is
        for a series with a unit, and is converted to the unit of each
        series embedded. A number is for a series without a unit. Months
        and years have no fixed length and are rejected.
    direction : {"backward", "forward"}, default "backward"
        Whether the components look back from ``t`` or ahead of it.

    Returns
    -------
    TimeSeries or TimeSeriesTensor
        The embedding, or for a tensor of series a tensor of the same shape
        holding the embedding of each.

    Raises
    ------
    ValueError
        On an invalid ``dimension``, ``delay``, or ``direction``, or when a
        channel spans less than ``(m - 1) * d``. For a tensor, the message
        names the position of the first series that fails.
    TypeError
        For a number on a series with a unit, or a duration on a series
        without one, or for an object that is not a :class:`TimeSeries`, a
        :class:`TimeSeriesTensor`, or a pandas ``Series`` or ``DataFrame``.
    """
    dimension = operator.index(dimension)
    if dimension < 1:
        raise ValueError("The embedding dimension must be at least 1")
    delay = _duration(delay, "The embedding delay must be finite and positive")
    if direction not in ("backward", "forward"):
        raise ValueError(f"direction must be 'backward' or 'forward', not {direction!r}")
    if isinstance(series, TimeSeriesTensor):
        # C++ converts the delay into the unit of each series.
        unit = delay.unit if isinstance(delay, TimeDelta) else None
        return TimeSeriesTensor(cpp.delay_embed_tensor(series._data, dimension, _in_unit(delay, unit), unit,
                                                       direction == "forward"))
    series = _as_series(series)
    return TimeSeries._from_cpp(cpp.delay_embed(series._data, dimension, _in_unit(delay, series.unit),
                                                direction == "forward"))
