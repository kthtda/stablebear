from __future__ import annotations

import math
import operator

from .. import _sb_cpp as cpp
from ..typing import boolean, float32, float64, int32, int64, uint32, uint64
from .series import TimeSeries

_NUMERIC = (boolean, float32, float64, int32, int64, uint32, uint64)


class TimeDelayEmbedding:
    """Delay embedding of a numeric time series.

    The embedding of a channel ``x`` with ``dimension`` ``m`` and ``delay``
    ``d`` has an observation at each of the channel's own times ``t`` from
    ``t_0 + (m - 1) * d`` on, where every component has a value: the point
    ``(x(t - (m - 1) * d), ..., x(t - d), x(t))``, oldest first, so each
    observation gains a leading axis of length ``m``. Like any series, it is evaluated
    to the observation in effect.

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
        On an invalid ``dimension`` or ``delay``, or when no time of the
        series has its full history.
    TypeError
        If the series does not hold numbers.
    """

    def __init__(self, dimension: int, delay: float):
        self.dimension = operator.index(dimension)
        self.delay = float(delay)
        if self.dimension < 1:
            raise ValueError("The embedding dimension must be at least 1")
        if math.isnan(self.delay) or math.isinf(self.delay) or self.delay <= 0:
            raise ValueError("The embedding delay must be finite and positive")

    def __call__(self, series: TimeSeries) -> TimeSeries:
        if series.dtype not in _NUMERIC:
            raise TypeError(f"Delay embedding needs a numeric series, got {series.dtype}")
        return TimeSeries._from_cpp(cpp.delay_embed(series._data, self.dimension, self.delay))

    def __repr__(self):
        return f"TimeDelayEmbedding(dimension={self.dimension}, delay={self.delay})"
