from __future__ import annotations

import datetime
from typing import Literal, overload

import numpy as np

from .. import _sb_cpp as cpp
from .date_time import _duration, _in_unit
from ..base_tensor import FloatTensor
from .series import TimeSeries, TimeSeriesTensor


class Window:
    """Consecutive windows of a time series.

    ``Window(size, step, anchor="start")(series)`` returns a 1-D tensor of
    dtype ``sb.series(series.dtype)`` with one window per start time
    ``a_k = t_0 + k * step``, while ``a_k + size`` is at most the last time
    at which every channel of ``series`` can be evaluated; ``t_0`` is the
    first such time. With ``partial=True``, windows continue while
    ``a_k`` is in the domain, and a window that would end after it ends where
    ``series`` does instead, as the last windows of Polars
    ``group_by_dynamic`` do. With ``anchor="end"``, the default, the windows
    are taken from the end of ``series`` going back instead: they end at
    ``b_k = t_1 - k * step``, ``t_1`` being the last time at which every
    channel can be evaluated, while ``b_k - size`` is at least ``t_0``. With
    ``partial=True`` they continue while ``b_k`` is in the domain, and a
    window that would start before ``t_0`` starts where ``series`` does.
    They still come in chronological order, so the last window ends at
    ``t_1``. With ``return_bounds=True`` it returns the tensor and a float64
    tensor of shape ``(n_windows, 2)`` whose row ``k`` is
    ``(a_k, a_k + size)``, in the unit of ``series``; a partial window's
    row starts or ends where the series does.

    Window ``k`` covers ``a_k`` to ``a_k + size``, with the ends that
    ``closed`` includes, and is a series with the channels of ``series``.
    Each channel holds the value in effect at ``a_k``, at time 0, then its
    observations at times ``t`` with ``a_k < t < a_k + size``, and one at
    ``a_k + size`` if the end is closed, at the times ``t - a_k`` in the
    channel's own unit. A window can be evaluated from 0 up to ``size``,
    after its last observation too, at 0 if the start is closed, and at
    ``size`` if the end is. With an open start the value at time 0 stays in
    effect after it, so it is still held, though evaluating at 0 raises.
    With ``closed="left"``, the default when anchored at the start, an
    observation at ``a_k + size`` belongs only to the next window; with
    ``closed="right"``, the default when anchored at the end, an
    observation at ``a_k`` belongs only to the previous one. A delay
    embedding of a window keeps its ends.
    A bound computed in floating point that falls just short of an
    observation reaches it, with the channel's snapping, as evaluation does:
    a window whose start reaches one is shifted by its time, and an end that
    reaches one excludes it, or includes it if closed. Each channel snaps
    with its own timestamps and settings, so channels of one window can
    snap differently; the bounds are therefore the times ``a_k`` and
    ``a_k + size`` as computed, before any snapping. For
    example, ``3 * 0.3`` computes just below 0.9: a window starting there
    starts at an observation at 0.9, but its bound reads
    ``0.8999999999999999``. The :attr:`~TimeSeries.origin` of each channel of
    a window is where its time 0 lies in ``series``, after its own snapping:
    0.9 in this example.
    Each window shares its observations with ``series`` and copies none; it
    evaluates by previous value on its own times, so it shows only the
    observations it holds.

    Parameters
    ----------
    size : float, np.timedelta64, datetime.timedelta, or pd.Timedelta
        The length of each window, positive. A number is in the series'
        unit; a duration such as ``np.timedelta64(25, "h")`` is converted
        to the unit of each series the windows are taken from. Months and
        years have no fixed length and are rejected.
    step : float, np.timedelta64, datetime.timedelta, or pd.Timedelta, optional
        The time between the starts of consecutive windows, positive, as
        for ``size``. Defaults to ``size``, so windows touch without
        overlapping; a smaller step makes them overlap by ``size - step``,
        and a larger one leaves gaps between them.
    closed : {"left", "right", "both", "none"}, optional
        Which ends of each window are included, as in Polars
        ``group_by_dynamic``: ``"left"`` is ``[a_k, a_k + size)``,
        ``"right"`` is ``(a_k, a_k + size]``. Defaults to the end at the
        anchor: ``"left"`` for ``anchor="start"`` and ``"right"`` for
        ``anchor="end"``, so the observation the windows are anchored at is
        in one.
    partial : bool, default False
        Whether to keep the windows away from the anchor that are shorter
        than ``size``.
    anchor : {"start", "end"}, default "end"
        Whether the windows are taken from the start of the series going
        forward, or from its end going back.

    Raises
    ------
    ValueError
        On a ``size`` or ``step`` that is not finite and positive, or an
        unknown ``closed`` or ``anchor``. When
        called, if a channel has no observations or ``size`` is longer
        than the time during which every channel has observations and
        ``partial`` is false, so that no window fits.
    """

    _CLOSED = {"left": (True, False), "right": (False, True), "both": (True, True), "none": (False, False)}

    def __init__(self, size: float | np.timedelta64 | datetime.timedelta,
                 step: float | np.timedelta64 | datetime.timedelta | None = None,
                 closed: Literal["left", "right", "both", "none"] | None = None, partial: bool = False,
                 anchor: Literal["start", "end"] = "end"):
        self.size = _duration(size, "The window size must be finite and positive")
        self.step = self.size if step is None else _duration(step, "The window step must be finite and positive")
        if anchor not in ("start", "end"):
            raise ValueError(f"anchor must be 'start' or 'end', not {anchor!r}")
        self.anchor = anchor
        if closed is None:
            closed = "left" if anchor == "start" else "right"
        if closed not in self._CLOSED:
            raise ValueError(f"closed must be 'left', 'right', 'both', or 'none', not {closed!r}")
        self.closed = closed
        self.partial = bool(partial)

    @overload
    def __call__(self, series: TimeSeries, return_bounds: Literal[False] = False) -> TimeSeriesTensor: ...

    @overload
    def __call__(self, series: TimeSeries, return_bounds: Literal[True]) -> tuple[TimeSeriesTensor, FloatTensor]: ...

    def __call__(self, series: TimeSeries, return_bounds: bool = False):
        windows, bounds = cpp.windows(series._data, _in_unit(self.size, series.unit), _in_unit(self.step, series.unit),
                                      *self._CLOSED[self.closed], self.partial, self.anchor == "end")
        if return_bounds:
            return TimeSeriesTensor(windows), FloatTensor(bounds)
        return TimeSeriesTensor(windows)

    def __repr__(self):
        return (f"Window(size={self.size!r}, step={self.step!r}, closed={self.closed!r}, partial={self.partial}, "
                f"anchor={self.anchor!r})")
