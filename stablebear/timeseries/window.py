from __future__ import annotations

import math
from typing import Literal, overload

from .. import _sb_cpp as cpp
from ..base_tensor import FloatTensor
from .series import TimeSeries, TimeSeriesTensor


class Window:
    """Consecutive windows of a time series.

    ``Window(size, step)(series)`` returns a 1-D tensor of dtype
    ``sb.series(series.dtype)`` with one window per start time
    ``a_k = t_0 + k * step``, while ``a_k + size`` is at most the last time
    at which every channel of ``series`` can be evaluated; ``t_0`` is the
    first such time. With ``partial=True``, windows continue while
    ``a_k`` is in the domain, and a window that would end after it ends where
    ``series`` does instead, as the last windows of Polars
    ``group_by_dynamic`` do. With ``return_bounds=True`` it returns the
    tensor and a float64 tensor of shape ``(n_windows, 2)`` whose row ``k``
    is ``(a_k, a_k + size)``, in the unit of ``series``; a partial window's
    row ends where the series does.

    Window ``k`` covers ``a_k`` to ``a_k + size``, with the ends that
    ``closed`` includes, and is a series with the channels of ``series``.
    Each channel holds the value in effect at ``a_k``, at time 0, then its
    observations at times ``t`` with ``a_k < t < a_k + size``, and one at
    ``a_k + size`` if the end is closed, at the times ``t - a_k`` in the
    channel's own unit. A window can be evaluated from 0 up to ``size``,
    after its last observation too, at 0 if the start is closed, and at
    ``size`` if the end is. With an open start the value at time 0 stays in
    effect after it, so it is still held, though evaluating at 0 raises.
    With the default ``closed="left"``, an observation at ``a_k + size``
    belongs only to the next window. A delay embedding of a window keeps
    its ends.
    A bound computed in floating point that falls just short of an
    observation reaches it, with the channel's snapping, as evaluation does:
    a window whose start reaches one is shifted by its time, and an end that
    reaches one excludes it, or includes it if closed. Each channel snaps
    with its own timestamps and settings, so channels of one window can
    snap differently; the bounds are therefore the times ``a_k`` and
    ``a_k + size`` as computed, before any snapping. For
    example, ``3 * 0.3`` computes just below 0.9: a window starting there
    starts at an observation at 0.9, but its bound reads
    ``0.8999999999999999``.
    Each window shares its observations with ``series`` and copies none; it
    evaluates by previous value on its own times, so it shows only the
    observations it holds.

    Parameters
    ----------
    size : float
        The length of each window, positive and in the series' unit.
    step : float, optional
        The time between the starts of consecutive windows, positive and in
        the series' unit. Defaults to ``size``, so windows touch without
        overlapping; a smaller step makes them overlap by ``size - step``,
        and a larger one leaves gaps between them.
    closed : {"left", "right", "both", "none"}, default "left"
        Which ends of each window are included, as in Polars
        ``group_by_dynamic``: ``"left"`` is ``[a_k, a_k + size)``,
        ``"right"`` is ``(a_k, a_k + size]``.
    partial : bool, default False
        Whether to keep the windows at the end that are shorter than
        ``size``.

    Raises
    ------
    ValueError
        On a ``size`` or ``step`` that is not finite and positive, or an
        unknown ``closed``. When
        called, if a channel has no observations or ``size`` is longer
        than the time during which every channel has observations and
        ``partial`` is false, so that no window fits.
    """

    _CLOSED = {"left": (True, False), "right": (False, True), "both": (True, True), "none": (False, False)}

    def __init__(self, size: float, step: float | None = None,
                 closed: Literal["left", "right", "both", "none"] = "left", partial: bool = False):
        self.size = float(size)
        if not (math.isfinite(self.size) and self.size > 0):
            raise ValueError("The window size must be finite and positive")
        self.step = self.size if step is None else float(step)
        if not (math.isfinite(self.step) and self.step > 0):
            raise ValueError("The window step must be finite and positive")
        if closed not in self._CLOSED:
            raise ValueError(f"closed must be 'left', 'right', 'both', or 'none', not {closed!r}")
        self.closed = closed
        self.partial = bool(partial)

    @overload
    def __call__(self, series: TimeSeries, return_bounds: Literal[False] = False) -> TimeSeriesTensor: ...

    @overload
    def __call__(self, series: TimeSeries, return_bounds: Literal[True]) -> tuple[TimeSeriesTensor, FloatTensor]: ...

    def __call__(self, series: TimeSeries, return_bounds: bool = False):
        windows, bounds = cpp.windows(series._data, self.size, self.step, *self._CLOSED[self.closed], self.partial)
        if return_bounds:
            return TimeSeriesTensor(windows), FloatTensor(bounds)
        return TimeSeriesTensor(windows)

    def __repr__(self):
        return f"Window(size={self.size}, step={self.step}, closed={self.closed!r}, partial={self.partial})"
