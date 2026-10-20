from __future__ import annotations

import math
import operator

from .. import _sb_cpp as cpp
from .series import TimeSeries


class TimeDelayEmbedding:
    """Delay embedding of a time series.

    The embedding of a channel ``x`` with ``dimension`` ``m`` and ``delay``
    ``d`` evaluates at every time ``t`` from ``t_0 + (m - 1) * d`` to the
    channel's last observation, both included, to the point
    ``(x(t - (m - 1) * d), ..., x(t - d), x(t))``, oldest first, so each
    observation gains a leading axis of length ``m``. Each component is the
    channel evaluated at its delayed time, with the channel's snapping, so
    the embedding agrees with evaluating the series even where a delayed
    time snaps to an observation. Its times are the ends
    of that domain and the times between where a component reaches an
    observation, ``t_i + k * d`` for ``k`` from 0 to ``m - 1``, so it has at
    most ``m`` times per observation of the channel. Shifted times that
    differ only by rounding and give the same point appear once, at the time
    they snap to: the channel's own timestamp if one is among them, such as
    0.3 rather than ``0.2 + 0.1``, else the latest of them. The first time is
    always ``t_0 + (m - 1) * d``.

    The embedding is a view: it shares the series' observations and
    evaluates them on demand.

    Parameters
    ----------
    dimension : int
        The number of components ``m``, at least 1.
    delay : float
        The time between components, positive and in the series' unit.

    Raises
    ------
    ValueError
        On an invalid ``dimension`` or ``delay``, or when
        ``t_0 + (m - 1) * d`` is after the last observation of a channel.
    """

    def __init__(self, dimension: int, delay: float):
        self.dimension = operator.index(dimension)
        self.delay = float(delay)
        if self.dimension < 1:
            raise ValueError("The embedding dimension must be at least 1")
        if math.isnan(self.delay) or math.isinf(self.delay) or self.delay <= 0:
            raise ValueError("The embedding delay must be finite and positive")

    def __call__(self, series: TimeSeries) -> TimeSeries:
        return TimeSeries._from_cpp(cpp.delay_embed(series._data, self.dimension, self.delay))

    def __repr__(self):
        return f"TimeDelayEmbedding(dimension={self.dimension}, delay={self.delay})"
