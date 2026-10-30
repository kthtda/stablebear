from __future__ import annotations

import datetime
import operator
from typing import Literal

import numpy as np

from .. import _sb_cpp as cpp
from .date_time import TimeDelta, _duration, _in_unit
from .series import TimeSeries


class TimeDelayEmbedding:
    """Delay embedding of a time series.

    The embedding of a channel ``x`` with ``dimension`` ``m`` and ``delay``
    ``d`` looks back by default: it evaluates at every time ``t`` from
    ``t_0 + (m - 1) * d`` to the end of the channel's domain, its last
    observation or, for a window (:class:`Window`), the window's end, to the
    point ``(x(t - (m - 1) * d), ..., x(t - d), x(t))``. With
    ``direction="forward"``, it evaluates at every time ``t`` from ``t_0`` to
    the end of the domain minus ``(m - 1) * d``, to the point
    ``(x(t), x(t + d), ..., x(t + (m - 1) * d))``. Both are oldest first, so
    the forward point at ``t`` is the backward point at ``t + (m - 1) * d``.
    Each end is included when the channel's is, and each observation gains a
    leading axis of length ``m``.

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

    Parameters
    ----------
    dimension : int
        The number of components ``m``, at least 1.
    delay : float, TimeDelta, np.timedelta64, datetime.timedelta, or pd.Timedelta
        The time between components, positive. A number is in the series'
        unit; a duration such as ``np.timedelta64(3, "h")`` is converted to
        the unit of each series embedded. Months and years have no fixed
        length and are rejected.
    direction : {"backward", "forward"}, default "backward"
        Whether the components look back from ``t`` or ahead of it.

    Raises
    ------
    ValueError
        On an invalid ``dimension``, ``delay``, or ``direction``, or when a
        channel spans less than ``(m - 1) * d``.
    """

    def __init__(self, dimension: int, delay: float | TimeDelta | np.timedelta64 | datetime.timedelta,
                 direction: Literal["backward", "forward"] = "backward"):
        self.dimension = operator.index(dimension)
        if self.dimension < 1:
            raise ValueError("The embedding dimension must be at least 1")
        self.delay = _duration(delay, "The embedding delay must be finite and positive")
        if direction not in ("backward", "forward"):
            raise ValueError(f"direction must be 'backward' or 'forward', not {direction!r}")
        self.direction = direction

    def __call__(self, series: TimeSeries) -> TimeSeries:
        return TimeSeries._from_cpp(cpp.delay_embed(series._data, self.dimension, _in_unit(self.delay, series.unit),
                                                    self.direction == "forward"))

    def __repr__(self):
        return (f"TimeDelayEmbedding(dimension={self.dimension}, delay={self.delay!r}, "
                f"direction={self.direction!r})")
