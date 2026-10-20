from __future__ import annotations

import math

from .. import _sb_cpp as cpp
from .series import TimeSeries


class Window:
    """Consecutive windows of a time series.

    ``Window(size, step)(series)`` returns a series with one channel and one
    observation per window, at its start time ``a_k = t_0 + k * step``, while
    ``a_k + size`` is at most the last time at which every channel of
    ``series`` has an observation; ``t_0`` is the first such time. The series
    has the unit of ``series`` and the dtype ``sb.series(series.dtype)``.

    Window ``k`` is a series with the channels of ``series``, each holding
    exactly its observations at times ``t`` with
    ``a_k <= t <= a_k + size``, both ends included, at the times
    ``t - a_k`` in the channel's own unit, so each window starts at 0.
    Nothing is carried in from before ``a_k``, so a window can be empty.
    A bound computed in floating point that falls just short of an
    observation reaches it, with the channel's snapping, as evaluation does:
    the observation is in the window, and a window whose start reaches one
    starts at it, so its first time is exactly 0. Each window shares its
    observations with ``series`` and evaluates as ``series`` does at the
    corresponding times.

    Parameters
    ----------
    size : float
        The length of each window, positive and in the series' unit.
    step : float, optional
        The time between the starts of consecutive windows, positive.
        Defaults to ``size``, so windows touch without overlapping.
    overlap : float, optional
        The time two consecutive windows share, in ``[0, size)``; the step
        is then ``size - overlap``. Pass ``step`` or ``overlap``, not both.

    Raises
    ------
    ValueError
        On a ``size`` or ``step`` that is not finite and positive, an
        ``overlap`` outside ``[0, size)``, or both ``step`` and ``overlap``.
    """

    def __init__(self, size: float, step: float | None = None,
                 overlap: float | None = None):
        self.size = float(size)
        if not (math.isfinite(self.size) and self.size > 0):
            raise ValueError("The window size must be finite and positive")
        if step is not None and overlap is not None:
            raise ValueError("Pass step or overlap, not both")
        if overlap is not None:
            overlap = float(overlap)
            if not 0 <= overlap < self.size:
                raise ValueError("The overlap must be in [0, size)")
            step = self.size - overlap
        self.step = self.size if step is None else float(step)
        if not (math.isfinite(self.step) and self.step > 0):
            raise ValueError("The window step must be finite and positive")

    def __call__(self, series: TimeSeries) -> TimeSeries:
        return TimeSeries._from_cpp(cpp.windows(series._data, self.size, self.step))

    def __repr__(self):
        return f"Window(size={self.size}, step={self.step})"
