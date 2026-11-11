from __future__ import annotations

from typing import Literal

from .. import _sb_cpp as cpp
from .date_time import _duration, _in_unit, _ScalarDuration
from .series import TimeSeries, TimeSeriesTensor, _PandasLike, _as_series


_CLOSED = {"left": (True, False), "right": (False, True), "both": (True, True), "none": (False, False)}


def cut_windows(series: TimeSeries | _PandasLike, size: float | _ScalarDuration,
                step: float | _ScalarDuration | None = None, *,
                closed: Literal["left", "right", "both", "none"] | None = None, partial: bool = False,
                anchor: Literal["start", "end"] = "end") -> TimeSeriesTensor:
    """Cut a time series into consecutive windows.

    ``cut_windows(series, size, step, anchor="start")`` returns a 1-D tensor of
    dtype ``sb.ts(series.dtype)`` with one window per start time
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
    ``t_1``. A pandas ``Series`` or ``DataFrame`` is converted
    with :meth:`TimeSeries.from_pandas` first.

    Window ``k`` covers ``a_k`` to ``a_k + size``, with the ends that
    ``closed`` includes, and is a series with the channels of ``series``.
    Each channel holds the value in effect at ``a_k``, at time 0, then its
    observations at times ``t`` with ``a_k < t < a_k + size``, and one at
    ``a_k + size`` if the end is closed, at the times ``t - a_k``. A window can be evaluated from 0 up to ``size``,
    after its last observation too, at 0 if the start is closed, and at
    ``size`` if the end is. With an open start the value at time 0 stays in
    effect after it, so it is still held, though evaluating at 0 raises.
    With ``closed="left"``, the default when anchored at the start, an
    observation at ``a_k + size`` belongs only to the next window; with
    ``closed="right"``, the default when anchored at the end, an
    observation at ``a_k`` belongs only to the previous one. A delay
    embedding of a window keeps its ends.
    A bound computed in floating point rounds on the scale of the terms it
    is computed from, ``t_0`` and ``k * step`` or a bound and ``size``, so
    it can miss an observation on either side. An observation within the
    channel's ``snap_tolerance`` times the larger magnitude of those terms
    of a bound, or one the bound snaps to as evaluation does, is at it: a
    window whose start is at one holds it at time 0, and an end at one
    excludes it, or includes it if closed. Whether a window fits in the
    domain is decided alike, with the smallest tolerance of the channels.
    So windows of times such as ``np.arange(n) / 10`` in seconds hold the
    same observations as windows of the same times in milliseconds. Each
    channel snaps
    with its own timestamps and settings, so channels of one window can
    snap differently, but all of them have their time 0 at ``a_k`` as
    computed: the :attr:`~TimeSeries.origin` of window ``k`` is its bound
    ``a_k``. For example, ``3 * 0.3`` computes just below 0.9: a window
    starting there holds an observation at 0.9 at time 0, and its origin
    has the count ``0.8999999999999999``. An observation at a bound is thus
    placed in ``series`` off by at most that reach.
    Each window shares its observations with ``series`` and copies none; it
    evaluates by previous value on its own times, so it shows only the
    observations it holds.

    Parameters
    ----------
    series : TimeSeries, pandas.Series, or pandas.DataFrame
        The series to cut.
    size : float, TimeDelta, tuple, np.timedelta64, datetime.timedelta, or pd.Timedelta
        The length of each window, positive. A duration such as
        ``(25, "h")``, ``TimeDelta(25, "h")``, or ``np.timedelta64(25, "h")``
        is for a series with a unit, and is converted to the unit of each
        series the windows are taken from. A number is for a series without
        a unit. Months and years have no fixed length and are rejected.
    step : float, TimeDelta, tuple, np.timedelta64, datetime.timedelta, or pd.Timedelta, optional
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
    Returns
    -------
    TimeSeriesTensor
        The windows.

    Raises
    ------
    ValueError
        On a ``size`` or ``step`` that is not finite and positive, an
        unknown ``closed`` or ``anchor``, or if a channel has no
        observations or ``size`` is longer than the time during which every
        channel has observations and ``partial`` is false, so that no
        window fits.
    TypeError
        For numbers on a series with a unit, or durations on a series
        without one, or for an object that is not a :class:`TimeSeries` or
        a pandas ``Series`` or ``DataFrame``.
    """
    size = _duration(size, "The window size must be finite and positive")
    step = size if step is None else _duration(step, "The window step must be finite and positive")
    if anchor not in ("start", "end"):
        raise ValueError(f"anchor must be 'start' or 'end', not {anchor!r}")
    if closed is None:
        closed = "left" if anchor == "start" else "right"
    if closed not in _CLOSED:
        raise ValueError(f"closed must be 'left', 'right', 'both', or 'none', not {closed!r}")
    series = _as_series(series)
    windows, _ = cpp.windows(series._data, _in_unit(size, series.unit), _in_unit(step, series.unit),
                             *_CLOSED[closed], bool(partial), anchor == "end")
    return TimeSeriesTensor(windows)

