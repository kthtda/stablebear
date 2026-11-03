import datetime
import re

import numpy as np
import pytest

import stablebear as sb
from stablebear import FloatTensor
from stablebear.timeseries import DateTime, DateTimeTensor, Domain, TimeDelayEmbedding, TimeDelta, TimeDeltaTensor, TimeSeries, cut_windows
from stablebear.timeseries.series import TimeSeriesTensor

from _element_values import TWO_ELEMENTS_OF_EVERY_DTYPE
from _round_trip import ROUND_TRIPS


def _windows_source():
    # Each observation is 10 * t.
    return TimeSeries([0.0, 5.0, 15.0, 25.0, 40.0], times=TimeDeltaTensor([0.0, 0.5, 1.5, 2.5, 4.0], "s"))


def _spans(windows):
    return [[w.source_domain.start.count, w.source_domain.end.count] for w in windows]


def test_windows_are_a_tensor_of_series_of_the_source_dtype():
    windows = cut_windows(_windows_source(), size=TimeDelta(2, "s"), step=TimeDelta(2, "s"), closed="left",
                          partial=False, anchor="start")

    # The windows cover [0.0, 2.0) and [2.0, 4.0). The first holds 0.0, 0.5,
    # and 1.5; the second starts with the value at 2.0, observed at 1.5, and
    # holds 2.5.
    assert isinstance(windows, TimeSeriesTensor)
    assert windows.dtype is sb.series(sb.float64)
    assert windows.shape == (2,)
    assert windows[0].dtype is sb.float64
    assert windows[1].dtype is sb.float64
    assert np.array_equal(windows[0].times.count, [0.0, 0.5, 1.5])
    assert np.array_equal(np.asarray(windows[0].values), [0.0, 5.0, 15.0])
    assert np.array_equal(windows[1].times.count, [0.0, 0.5])
    assert np.array_equal(np.asarray(windows[1].values), [15.0, 25.0])


@pytest.mark.parametrize("window, starts", [
    pytest.param(dict(size=TimeDelta(2, "s"), step=TimeDelta(1, "s")), [0.0, 1.0, 2.0], id="step"),
    pytest.param(dict(size=TimeDelta(1, "s"), step=TimeDelta(1.5, "s")), [0.0, 1.5, 3.0], id="gaps"),
    pytest.param(dict(size=TimeDelta(2, "s")), [0.0, 2.0], id="default-step"),
])
def test_windows_start_every_step_while_they_fit(window, starts):
    windows = cut_windows(_windows_source(), **window, closed="left", partial=False, anchor="start")

    assert windows.shape == (len(starts),)
    assert _spans(windows) == [[start, start + window["size"].count] for start in starts]


@pytest.mark.parametrize("size, step, bounds", [
    pytest.param(TimeDelta(2, "s"), TimeDelta(1, "s"),
                 [[0.0, 2.0],
                  [1.0, 3.0],
                  [2.0, 4.0]], id="step"),
    pytest.param(TimeDelta(1, "s"), TimeDelta(1.5, "s"),
                 [[0.0, 1.0],
                  [1.5, 2.5],
                  [3.0, 4.0]], id="gaps"),
])
def test_windows_anchored_at_the_end_end_every_step_back_while_they_fit(size, step, bounds):
    windows = cut_windows(_windows_source(), size=size, step=step, closed="right", partial=False, anchor="end")

    # The last window ends where the series does, at 4.0, and the windows
    # come in chronological order.
    assert _spans(windows) == bounds


@pytest.mark.parametrize("anchor", ["start", "end"])
@pytest.mark.parametrize("series, match", [
    pytest.param(_windows_source(), "longer than the series", id="longer-than-the-series"),
    pytest.param(TimeSeries([], times=TimeDeltaTensor([], "s")), "without observations", id="empty"),
])
def test_windowing_raises_when_no_window_fits(series, match, anchor):
    with pytest.raises(ValueError, match=match):
        cut_windows(series, size=TimeDelta(5, "s"), step=TimeDelta(1, "s"), closed="left", partial=False, anchor=anchor)


@pytest.mark.parametrize("size, step, match", [
    pytest.param(TimeDelta(0, "s"), TimeDelta(1, "s"), "size", id="zero-size"),
    pytest.param(TimeDelta(-1, "s"), TimeDelta(1, "s"), "size", id="negative-size"),
    pytest.param(TimeDelta(1, "s"), TimeDelta(0, "s"), "step", id="zero-step"),
    pytest.param(TimeDelta(1, "s"), TimeDelta(-1, "s"), "step", id="negative-step"),
])
def test_window_raises_on_a_size_or_step_that_is_not_positive(size, step, match):
    with pytest.raises(ValueError, match=match):
        cut_windows(_windows_source(), size=size, step=step)


def test_window_raises_on_an_unknown_closed():
    with pytest.raises(ValueError, match="closed"):
        cut_windows(_windows_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"),
                    closed="middle")  # pyright: ignore[reportArgumentType]


def test_window_raises_on_an_unknown_anchor():
    with pytest.raises(ValueError, match="anchor"):
        cut_windows(_windows_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left",
                    anchor="middle")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("anchor, ends", [
    pytest.param("start", (True, False), id="start"),
    pytest.param("end", (False, True), id="end"),
])
def test_windows_are_closed_at_their_anchor_by_default(anchor, ends):
    window = cut_windows(_windows_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), anchor=anchor)[0]

    assert (window.domain.start_closed, window.domain.end_closed) == ends


def test_windows_are_anchored_at_the_end_by_default():
    windows = cut_windows(_windows_source(), size=TimeDelta(1.5, "s"), step=TimeDelta(1.5, "s"), closed="right",
                          partial=False)

    assert _spans(windows) == [[1.0, 2.5], [2.5, 4.0]]


def _closed_source():
    # Each observation is 10 * t.
    return TimeSeries([0.0, 5.0, 10.0, 15.0, 20.0], times=TimeDeltaTensor([0.0, 0.5, 1.0, 1.5, 2.0], "s"))


def test_a_left_closed_window_includes_its_start_but_not_its_end():
    # The window [1.0, 2.0).
    window = cut_windows(_closed_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left", partial=False,
                         anchor="start")[1]

    assert window(TimeDelta(0, "s")) == 10.0
    with pytest.raises(ValueError, match=re.escape("outside the domain [0, 1)")):
        window(TimeDelta(1, "s"))


def test_a_right_closed_window_includes_its_end_but_not_its_start():
    # The window (1.0, 2.0]. The observation at 1.0 stays in effect just
    # after the start.
    window = cut_windows(_closed_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="right",
                         partial=False, anchor="start")[1]

    with pytest.raises(ValueError, match=re.escape("outside the domain (0, 1]")):
        window(TimeDelta(0, "s"))
    assert window(TimeDelta(0.25, "s")) == 10.0
    assert window(TimeDelta(1, "s")) == 20.0


def test_a_window_closed_on_both_sides_includes_its_start_and_its_end():
    # The window [1.0, 2.0].
    window = cut_windows(_closed_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="both", partial=False,
                         anchor="start")[1]

    assert window(TimeDelta(0, "s")) == 10.0
    assert window(TimeDelta(1, "s")) == 20.0


def test_a_window_closed_on_neither_side_includes_neither_end():
    # The window (1.0, 2.0). The observation at 1.0 stays in effect just
    # after the start.
    window = cut_windows(_closed_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="none", partial=False,
                         anchor="start")[1]

    with pytest.raises(ValueError, match=re.escape("outside the domain (0, 1)")):
        window(TimeDelta(0, "s"))
    assert window(TimeDelta(0.25, "s")) == 10.0
    with pytest.raises(ValueError, match=re.escape("outside the domain (0, 1)")):
        window(TimeDelta(1, "s"))


@pytest.mark.parametrize("closed, start_closed, end_closed", [
    pytest.param("left", True, False, id="left"),
    pytest.param("right", False, True, id="right"),
    pytest.param("both", True, True, id="both"),
    pytest.param("none", False, False, id="none"),
])
def test_a_window_domain_runs_from_zero_to_its_size_with_its_closed_ends(closed, start_closed, end_closed):
    window = cut_windows(_windows_source(), size=TimeDelta(2, "s"), step=TimeDelta(2, "s"), closed=closed,
                         partial=False, anchor="start")[0]

    assert window.domain == Domain(TimeDelta(0, "s"), TimeDelta(2, "s"), start_closed, end_closed)


def test_a_window_origin_is_its_start_in_the_series():
    windows = cut_windows(_windows_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left",
                          partial=False, anchor="start")

    # The windows start at 0.0, 1.0, 2.0, and 3.0; none starts at an
    # observation it would snap to.
    origins = [windows[k].origin for k in range(windows.shape[0])]
    assert origins == [TimeDelta(k, "s") for k in range(4)]


def test_a_window_origin_places_its_observations_at_their_times_in_the_series():
    # Each observation is 10 * t.
    series = TimeSeries([3.0, 7.0, 9.0], times=TimeDeltaTensor([0.3, 0.7, 0.9], "s"))

    # The window [0.3, 0.8) holds the observations at 0.3 and 0.7, at times
    # 0 and 0.7 - 0.3, which rounds to 0.39999999999999997.
    window = cut_windows(series, size=TimeDelta(0.5, "s"), step=TimeDelta(0.5, "s"), closed="left", partial=False,
                         anchor="start")[0]

    placed = window.origin.count + window.times.count
    assert sb.allclose(placed, FloatTensor([0.3, 0.7], dtype=sb.float64), atol=0.0, rtol=4e-16)


def _partial_source():
    # Each observation is 10 * t.
    return TimeSeries([0.0, 10.0, 20.0, 30.0, 40.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 3.0, 4.0], "s"))


@pytest.mark.parametrize("partial, expected_bounds", [
    pytest.param(False, [[0.0, 3.0]], id="full"),
    pytest.param(True, [[0.0, 3.0], [3.0, 4.0]], id="partial"),
])
def test_partial_windows_continue_to_the_end_of_the_series(partial, expected_bounds):
    windows = cut_windows(_partial_source(), size=TimeDelta(3, "s"), step=TimeDelta(3, "s"), closed="left",
                          partial=partial, anchor="start")

    # A partial window ends where the series does.
    assert _spans(windows) == expected_bounds


def test_a_partial_window_ends_where_the_series_does():
    # The window at 3.0 would end at 6.0, after the series ends at 4.0.
    window = cut_windows(_partial_source(), size=TimeDelta(3, "s"), step=TimeDelta(3, "s"), closed="left", partial=True,
                         anchor="start")[1]

    assert np.array_equal(window.times.count, [0.0, 1.0])
    assert np.array_equal(np.asarray(window.values), [30.0, 40.0])
    assert window(TimeDelta(1, "s")) == 40.0
    with pytest.raises(ValueError, match=re.escape("outside the domain [0, 1]")):
        window(TimeDelta(1.5, "s"))


def test_a_window_anchored_at_the_end_holds_the_last_observation():
    # The window (1.0, 4.0] holds the value at 1.0 just after its start,
    # then the observations at 2.0, 3.0, and 4.0.
    window = cut_windows(_partial_source(), size=TimeDelta(3, "s"), step=TimeDelta(3, "s"), closed="right",
                         partial=False, anchor="end")[0]

    assert np.array_equal(window.times.count, [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(np.asarray(window.values), [10.0, 20.0, 30.0, 40.0])
    assert window(TimeDelta(3, "s")) == 40.0


def test_a_partial_window_anchored_at_the_end_starts_where_the_series_does():
    windows = cut_windows(_partial_source(), size=TimeDelta(3, "s"), step=TimeDelta(3, "s"), closed="right",
                          partial=True, anchor="end")

    # The window ending at 1.0 would start at -2.0, before the series starts
    # at 0.0, so it is [0.0, 1.0], closed at the start as the series is.
    assert _spans(windows) == [
        [0.0, 1.0],
        [1.0, 4.0],
    ]
    assert np.array_equal(windows[0].times.count, [0.0, 1.0])
    assert np.array_equal(np.asarray(windows[0].values), [0.0, 10.0])
    assert windows[0](TimeDelta(0, "s")) == 0.0
    assert windows[0](TimeDelta(1, "s")) == 10.0
    # The full window (1.0, 4.0] holds the value at 1.0 just after its start,
    # then the observations at 2.0, 3.0, and 4.0.
    assert np.array_equal(windows[1].times.count, [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(np.asarray(windows[1].values), [10.0, 20.0, 30.0, 40.0])
    assert windows[1](TimeDelta(3, "s")) == 40.0


def test_a_partial_window_domain_ends_where_the_series_does():
    # The window at 3.0 would end at 6.0; it ends at the series' closed end
    # at 4.0 instead.
    window = cut_windows(_partial_source(), size=TimeDelta(3, "s"), step=TimeDelta(3, "s"), closed="left", partial=True,
                         anchor="start")[1]

    assert window.domain == Domain(TimeDelta(0, "s"), TimeDelta(1, "s"), True, True)


def test_a_left_closed_partial_window_can_start_at_the_last_observation():
    windows = cut_windows(_partial_source(), size=TimeDelta(2, "s"), step=TimeDelta(2, "s"), closed="left",
                          partial=True, anchor="start")

    # The window at 4.0 holds only the last observation.
    assert [start for start, _ in _spans(windows)] == [0.0, 2.0, 4.0]
    assert np.array_equal(windows[2].times.count, [0.0])
    assert windows[2](TimeDelta(0, "s")) == 40.0


def test_a_right_closed_window_holds_the_last_observation_without_a_partial_window():
    windows = cut_windows(_partial_source(), size=TimeDelta(2, "s"), step=TimeDelta(2, "s"), closed="right",
                          partial=True, anchor="start")

    # The window (2.0, 4.0] ends at the last observation, and no window can
    # start there with an open start.
    assert [start for start, _ in _spans(windows)] == [0.0, 2.0]
    assert windows[1](TimeDelta(2, "s")) == 40.0


def test_a_window_can_be_evaluated_from_its_start_up_to_its_end():
    # The window [1.0, 3.0) holds the value at 1.0, observed at 0.5, then 1.5
    # and 2.5.
    window = cut_windows(_windows_source(), size=TimeDelta(2, "s"), step=TimeDelta(1, "s"), closed="left",
                         partial=False, anchor="start")[1]

    assert np.array_equal(window.times.count, [0.0, 0.5, 1.5])
    assert np.array_equal(np.asarray(window.values), [5.0, 15.0, 25.0])
    assert window(TimeDelta(0, "s")) == 5.0
    assert window(TimeDelta(1.75, "s")) == 25.0
    with pytest.raises(ValueError, match=r"outside the domain \[0, 2\)"):
        window(TimeDelta(2, "s"))


@pytest.mark.parametrize("duration", [
    pytest.param(TimeDelta(1750, "ms"), id="TimeDelta"),
    pytest.param(np.timedelta64(1750, "ms"), id="timedelta64"),
    pytest.param(datetime.timedelta(milliseconds=1750), id="timedelta"),
])
def test_a_window_at_a_duration_equals_the_window_at_its_time_in_the_unit(duration):
    # The window [1.0, 3.0) in seconds.
    window = cut_windows(_windows_source(), size=TimeDelta(2, "s"), step=TimeDelta(1, "s"), closed="left",
                         partial=False, anchor="start")[1]

    assert window(duration) == window(TimeDelta(1.75, "s"))


def test_an_observation_at_a_window_boundary_is_only_in_the_later_window():
    windows = cut_windows(_windows_source(), size=TimeDelta(1.5, "s"), step=TimeDelta(1.5, "s"), closed="left",
                          partial=False, anchor="start")

    # The observation at 1.5 starts the second window, not the first.
    assert np.array_equal(np.asarray(windows[0].values), [0.0, 5.0])
    assert np.array_equal(np.asarray(windows[1].values), [15.0, 25.0])


def test_a_window_between_observations_holds_the_value_at_its_start():
    # The window [3.0, 3.5) falls between the observations at 2.5 and 4.0.
    window = cut_windows(_windows_source(), size=TimeDelta(0.5, "s"), step=TimeDelta(0.5, "s"), closed="left",
                         partial=False, anchor="start")[6]

    assert np.array_equal(window.times.count, [0.0])
    assert np.array_equal(np.asarray(window.values), [25.0])
    assert window(TimeDelta(0.25, "s")) == 25.0


def test_window_bounds_just_below_an_observation_reach_it():
    series = TimeSeries([0.0, 3.0, 6.0, 9.0, 12.0], times=TimeDeltaTensor([0.0, 0.3, 0.6, 0.9, 1.2], "s"))

    windows = cut_windows(series, size=TimeDelta(0.3, "s"), step=TimeDelta(0.3, "s"), closed="left",
                          partial=False, anchor="start")

    # 3 * 0.3 computes just below 0.9, so window 2 ends and window 3 starts
    # there.
    spans = _spans(windows)
    assert spans[2][1] < 0.9
    assert spans[3][0] < 0.9
    # The observation at 0.9 is the end of window 2, so only window 3 holds
    # it, at time 0.
    assert np.array_equal(np.asarray(windows[2].values), [6.0])
    assert np.array_equal(windows[3].times.count, [0.0])
    assert np.array_equal(np.asarray(windows[3].values), [9.0])


def test_a_window_start_that_snaps_to_an_observation_keeps_its_start_as_origin():
    series = TimeSeries([0.0, 3.0, 6.0, 9.0, 12.0], times=TimeDeltaTensor([0.0, 0.3, 0.6, 0.9, 1.2], "s"))

    windows = cut_windows(series, size=TimeDelta(0.3, "s"), step=TimeDelta(0.3, "s"), closed="left",
                          partial=False, anchor="start")

    # Window 3 starts at 3 * 0.3, just below 0.9, and snaps to the
    # observation at 0.9, which it holds at time 0.
    assert windows[3].origin == TimeDelta(3 * 0.3, "s")
    assert np.array_equal(windows[3].times.count, [0.0])
    assert windows[3](TimeDelta(0, "s")) == 9.0


def _snapping_channels():
    # Channel a has an observation at 0.9 for window 3's start to snap to;
    # channel b does not.
    a = TimeSeries([0.0, 3.0, 6.0, 9.0, 12.0], times=TimeDeltaTensor([0.0, 0.3, 0.6, 0.9, 1.2], "s"))
    b = TimeSeries([0.0, -5.0, -10.0, -12.0], times=TimeDeltaTensor([0.0, 0.5, 1.0, 1.2], "s"))
    return TimeSeries.from_channels([a, b])


def test_channels_that_snap_differently_share_an_origin():
    window = cut_windows(_snapping_channels(), size=TimeDelta(0.3, "s"), step=TimeDelta(0.3, "s"), closed="left",
                         partial=False, anchor="start")[3]

    assert window.origin.array_equal(TimeDeltaTensor([3 * 0.3, 3 * 0.3], "s"))
    # Channel a holds its observation at 0.9 at time 0; b holds the one at 0.5.
    values = window(TimeDelta(0, "s"))
    assert (values[0][()], values[1][()]) == (9.0, -5.0)


def test_a_window_of_a_window_has_its_origin_in_the_first_series():
    series = TimeSeries(np.arange(9.0), times=TimeDeltaTensor(np.arange(9.0), "s"))
    window = cut_windows(series, size=TimeDelta(4, "s"), step=TimeDelta(4, "s"), closed="left", partial=False,
                         anchor="start")[1]

    # The window starts at 4.0; its windows start at 0.0 and 2.0 within it.
    inner = cut_windows(window, size=TimeDelta(2, "s"), step=TimeDelta(2, "s"), closed="left", partial=False,
                        anchor="start")

    assert (inner[0].origin, inner[1].origin) == (TimeDelta(4, "s"), TimeDelta(6, "s"))


def test_an_embedding_of_a_window_keeps_its_origin():
    window = cut_windows(_windows_source(), size=TimeDelta(2, "s"), step=TimeDelta(2, "s"), closed="left",
                         partial=False, anchor="start")[1]

    embedding = TimeDelayEmbedding(dimension=2, delay=TimeDelta(0.5, "s"))(window)

    assert embedding.origin == TimeDelta(2, "s")


def _dated_source():
    # The observations of _windows_source, counted from 2023-01-01.
    source = _windows_source()
    return TimeSeries(source.values, times=source.times, epoch="2023-01-01")


def test_windows_keep_the_epoch_of_their_series():
    windows = cut_windows(_dated_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left", partial=False,
                          anchor="start")

    assert windows[2].epoch == DateTime("2023-01-01")


@pytest.mark.parametrize("closed, start_closed, end_closed", [
    pytest.param("left", True, False, id="left"),
    pytest.param("right", False, True, id="right"),
])
def test_a_dated_window_domain_is_where_it_lies(closed, start_closed, end_closed):
    windows = cut_windows(_dated_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed=closed, partial=False,
                          anchor="start")

    domain = windows[2].domain

    assert domain.start.to_datetime64() == np.datetime64("2023-01-01T00:00:02")
    assert domain.end.to_datetime64() == np.datetime64("2023-01-01T00:00:03")
    assert (domain.start_closed, domain.end_closed) == (start_closed, end_closed)


def test_a_window_of_a_series_without_a_unit_gives_numbers():
    series = TimeSeries([0.0, 5.0, 15.0, 25.0, 40.0], times=[0.0, 0.5, 1.5, 2.5, 4.0])

    windows = cut_windows(series, size=2.0, step=2.0, closed="left", partial=False, anchor="start")

    assert windows[1].unit is None
    assert windows[1].origin == 2.0
    assert windows[1].times.array_equal(sb.FloatTensor([0.0, 0.5], dtype=sb.float64))
    assert windows[1].domain == Domain(0.0, 2.0, True, False)


def test_windows_of_a_number_on_a_series_with_a_unit_raise():
    with pytest.raises(TypeError, match="has a unit"):
        cut_windows(_windows_source(), size=2.0, step=2.0, closed="left", partial=False, anchor="start")


def test_windows_of_a_duration_on_a_series_without_a_unit_raise():
    series = TimeSeries([0.0, 5.0, 15.0, 25.0, 40.0], times=[0.0, 0.5, 1.5, 2.5, 4.0])
    with pytest.raises(TypeError, match="without a unit"):
        cut_windows(series, size=TimeDelta(2, "s"), step=TimeDelta(2, "s"), closed="left", partial=False,
                    anchor="start")


def test_windows_of_a_dated_window_lie_where_they_are_in_its_series():
    window = cut_windows(_dated_source(), size=TimeDelta(2, "s"), step=TimeDelta(2, "s"), closed="left", partial=False,
                         anchor="start")[1]

    windows = cut_windows(window, size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left", partial=False,
                          anchor="start")

    assert [windows[k].domain.start for k in range(2)] == [DateTime("2023-01-01T00:00:02"),
                                                          DateTime("2023-01-01T00:00:03")]


def test_a_dated_window_domain_keeps_a_unit_finer_than_microseconds():
    series = TimeSeries([0.0, 1.0, 2.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "ns"), epoch="2023-01-01")

    windows = cut_windows(series, size=TimeDelta(1, "ns"), step=TimeDelta(1, "ns"), closed="left", partial=False,
                          anchor="start")

    assert windows[1].domain.start == DateTime(np.datetime64("2023-01-01T00:00:00.000000001", "ns"))
    assert windows[1].domain.end == DateTime(np.datetime64("2023-01-01T00:00:00.000000002", "ns"))


def test_an_embedding_keeps_the_epoch_of_its_series():
    embedding = TimeDelayEmbedding(dimension=2, delay=TimeDelta(0.5, "s"))(_dated_source())

    assert embedding.epoch == DateTime("2023-01-01")


def test_a_window_with_an_epoch_keeps_its_origin():
    window = cut_windows(_windows_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left",
                         partial=False, anchor="start")[2]

    dated = window.with_epoch("2023-01-01")

    assert dated.epoch == DateTime("2023-01-01")
    assert dated.origin == TimeDelta(2, "s")


def test_a_window_origin_is_reported_in_the_query_unit():
    window = cut_windows(_windows_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left",
                         partial=False, anchor="start")[2]

    assert window.to("ms").origin == TimeDelta(2000, "ms")


def test_an_embedding_of_a_window_keeps_its_open_end():
    # The window [1.0, 3.0) holds 5, 15, and 25 at 0.0, 0.5, and 1.5. With a
    # delay of 0.4, the newer component reaches 1.5 at 1.9, between the
    # window's last observation and its end.
    window = cut_windows(_windows_source(), size=TimeDelta(2, "s"), step=TimeDelta(1, "s"), closed="left",
                         partial=False, anchor="start")[1]

    embedding = TimeDelayEmbedding(dimension=2, delay=TimeDelta(0.4, "s"))(window)

    assert np.array_equal(embedding.times.count, [0.4, 0.5, 0.9, 1.5, 1.9])
    assert np.array_equal(np.asarray(embedding(TimeDelta(1.95, "s"))), [25.0, 25.0])
    with pytest.raises(ValueError, match=r"outside the domain \[0.4, 2\)"):
        embedding(TimeDelta(2, "s"))


def test_an_embedding_of_a_window_keeps_its_open_start_and_closed_end():
    # The window (1.0, 3.0] holds 10 and 20 at 0.0 and 1.0, and ends at 2.0,
    # after its last observation.
    series = TimeSeries([0.0, 10.0, 20.0, 40.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 4.0], "s"))
    window = cut_windows(series, size=TimeDelta(2, "s"), step=TimeDelta(1, "s"), closed="right", partial=False,
                         anchor="start")[1]

    embedding = TimeDelayEmbedding(dimension=2, delay=TimeDelta(1, "s"))(window)

    assert np.array_equal(embedding.times.count, [1.0, 2.0])
    assert np.array_equal(np.asarray(embedding(TimeDelta(2, "s"))), [20.0, 20.0])
    with pytest.raises(ValueError, match=re.escape("outside the domain (1, 2]")):
        embedding(TimeDelta(1, "s"))


def test_an_embedding_of_a_window_takes_no_history_from_before_the_window():
    # The window [1.0, 3.0) holds 5, 15, and 25 at 0.0, 0.5, and 1.5. With a
    # delay of 1.0, the first point is at 1.0.
    window = cut_windows(_windows_source(), size=TimeDelta(2, "s"), step=TimeDelta(1, "s"), closed="left",
                         partial=False, anchor="start")[1]

    embedding = TimeDelayEmbedding(dimension=2, delay=TimeDelta(1, "s"))(window)

    assert np.array_equal(embedding.times.count, [1.0, 1.5])
    assert np.array_equal(np.asarray(embedding.values), [[5.0, 15.0], [15.0, 25.0]])
    with pytest.raises(ValueError, match=r"outside the domain \[1, 2\)"):
        embedding(TimeDelta(0.75, "s"))
    with pytest.raises(ValueError, match=r"outside the domain \[1, 2\)"):
        embedding(TimeDelta(2, "s"))


def test_windows_of_an_embedding_hold_its_points():
    # The embedding has points at 1.0, 1.5, 2.5, 3.5, and 4.0.
    embedding = TimeDelayEmbedding(dimension=2, delay=TimeDelta(1, "s"))(_windows_source())

    windows = cut_windows(embedding, size=TimeDelta(2, "s"), step=TimeDelta(1, "s"), closed="left",
                          partial=False, anchor="start")

    assert [start for start, _ in _spans(windows)] == [1.0, 2.0]
    assert np.array_equal(windows[0].times.count, [0.0, 0.5, 1.5])
    assert np.array_equal(np.asarray(windows[0].values), [[0.0, 5.0], [5.0, 15.0], [15.0, 25.0]])


def test_windows_cut_every_channel_at_the_same_bounds():
    # A's observations are 10 * t and B's are -10 * t, with t in seconds.
    a = TimeSeries([0.0, 10.0, 25.0], times=TimeDeltaTensor([0.0, 1.0, 2.5], "s"))
    b = TimeSeries([-5.0, -15.0, -30.0], times=TimeDeltaTensor([500.0, 1500.0, 3000.0], "ms"))
    series = TimeSeries.from_channels([a, b], unit="s")

    windows = cut_windows(series, size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left", partial=False,
                          anchor="start")

    # Both channels are observed from 0.5 to 2.5, so the windows are
    # [0.5, 1.5) and [1.5, 2.5).
    assert _spans(windows) == [[0.5, 1.5], [1.5, 2.5]]
    # B's observation at 1500 ms is the end of the first window.
    assert np.array_equal(windows[0][0].times.count, [0.0, 0.5])
    assert np.array_equal(windows[0][1].times.count, [0.0])
    assert np.array_equal(windows[1][0].times.count, [0.0])
    assert np.array_equal(windows[1][1].times.count, [0.0])
    assert windows[0](TimeDelta(0.5, "s"))[0][()] == 10.0
    assert windows[0](TimeDelta(0.5, "s"))[1][()] == -5.0
    with pytest.raises(ValueError, match=r"outside the domain \[0, 1\)"):
        windows[0](TimeDelta(1, "s"))


def _overlapping_windows():
    # Windows at 0.0, 1.0, and 2.0.
    series = TimeSeries([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
                        times=TimeDeltaTensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], "s"))
    return cut_windows(series, size=TimeDelta(2, "s"), step=TimeDelta(1, "s"), closed="left", partial=False,
                       anchor="start")


def test_windows_evaluate_at_times_in_window_order():
    windows = _overlapping_windows()

    values = windows(TimeDeltaTensor([0.0, 1.0], "s"))

    assert isinstance(values, FloatTensor)
    assert np.array_equal(np.asarray(values), [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]])
    assert np.array_equal(np.asarray(windows(TimeDelta(1.5, "s"))), [15.0, 25.0, 35.0])


def test_windows_evaluate_at_times_of_any_shape():
    windows = _overlapping_windows()

    values = windows(TimeDeltaTensor([[0.0, 1.0], [1.5, 0.5]], "s"))

    assert np.array_equal(np.asarray(values), [
        [[0.0, 10.0], [15.0, 5.0]],
        [[10.0, 20.0], [25.0, 15.0]],
        [[20.0, 30.0], [35.0, 25.0]],
    ])


def test_windows_evaluate_at_times_in_the_given_unit():
    windows = _overlapping_windows()

    values = windows([0.0, 1000.0], "ms")

    assert np.array_equal(np.asarray(values), [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]])


@pytest.mark.parametrize("duration", [
    pytest.param(TimeDelta(1500, "ms"), id="TimeDelta"),
    pytest.param(np.timedelta64(1500, "ms"), id="timedelta64"),
    pytest.param(datetime.timedelta(milliseconds=1500), id="timedelta"),
])
def test_windows_at_a_duration_equal_the_windows_at_its_time_in_the_unit(duration):
    windows = _overlapping_windows()

    assert windows(duration).array_equal(windows(TimeDelta(1.5, "s")))


@pytest.mark.parametrize("durations", [
    pytest.param(np.array([0, 500, 1000], dtype="timedelta64[ms]"), id="timedelta64"),
    pytest.param(TimeDeltaTensor(sb.FloatTensor([0, 500, 1000]), "ms"), id="TimeDeltaTensor"),
])
def test_windows_at_an_array_of_durations_equal_the_windows_at_their_times(durations):
    windows = _overlapping_windows()

    assert windows(durations).array_equal(windows(TimeDeltaTensor([0.0, 0.5, 1.0], "s")))


def test_windows_at_a_list_of_time_deltas_equal_the_windows_at_their_times():
    windows = _overlapping_windows()

    expected = windows(TimeDeltaTensor([0.5, 1.0], "s"))

    assert windows([TimeDelta(500, "ms"), TimeDelta(1, "s")]).array_equal(expected)


def test_windows_at_a_duration_take_no_unit():
    with pytest.raises(TypeError, match="no unit"):
        _overlapping_windows()(np.timedelta64(1, "s"), "s")


def test_windows_reject_an_unknown_unit():
    windows = _overlapping_windows()

    with pytest.raises(ValueError, match="Unknown time unit"):
        windows([0.0, 1.0], "fortnight")


def test_windows_at_numbers_of_a_unit_equal_the_windows_at_their_durations():
    windows = _overlapping_windows()

    assert windows(1500.0, "ms").array_equal(windows(TimeDelta(1500.0, "ms")))
    assert windows([0.0, 1500.0], "ms").array_equal(windows(TimeDeltaTensor([0.0, 1500.0], "ms")))


@pytest.mark.parametrize("t", [pytest.param(1.5, id="single"), pytest.param([0.0, 1.5], id="many")])
def test_windows_with_a_unit_reject_numbers_without_unit(t):
    with pytest.raises(TypeError, match="have a unit"):
        _overlapping_windows()(t)


def _windows_without_a_unit():
    series = TimeSeries([0.0, 5.0, 15.0, 25.0, 40.0], times=[0.0, 0.5, 1.5, 2.5, 4.0])
    return cut_windows(series, size=2.0, step=2.0, closed="left", partial=False, anchor="start")


def test_windows_without_a_unit_take_numbers():
    windows = _windows_without_a_unit()

    assert windows(0.5).array_equal(sb.FloatTensor([5.0, 25.0], dtype=sb.float64))
    assert windows([0.0, 0.5]).array_equal(sb.FloatTensor([[0.0, 5.0], [15.0, 25.0]], dtype=sb.float64))


@pytest.mark.parametrize("t, unit", [
    pytest.param(TimeDelta(0.5, "s"), None, id="duration"),
    pytest.param(TimeDeltaTensor([0.0, 0.5], "s"), None, id="durations"),
    pytest.param(0.5, "s", id="unit"),
])
def test_windows_without_a_unit_reject_durations_and_units(t, unit):
    with pytest.raises(TypeError, match="without a unit"):
        _windows_without_a_unit()(t, unit)


def test_windows_of_several_channels_evaluate_to_one_leaf_per_channel():
    # A's observations are 10 * t and B's are -10 * t, with t in seconds; B
    # starts 0.5 s after A. Both are observed from 0.5 to 2.5, so the windows
    # are [0.5, 1.5) and [1.5, 2.5).
    a = TimeSeries([0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
                   times=TimeDeltaTensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], "s"))
    b = TimeSeries([-5.0, -10.0, -15.0, -20.0, -25.0, -30.0],
                   times=TimeDeltaTensor([500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0], "ms"))
    series = TimeSeries.from_channels([a, b], unit="s")
    windows = cut_windows(series, size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left", partial=False,
                          anchor="start")

    values = windows(TimeDeltaTensor([0.0, 0.5], "s"))

    assert isinstance(values, sb.NestedTensor)
    assert np.array_equal(np.asarray(values[0]), [[5.0, 10.0], [15.0, 20.0]])
    assert np.array_equal(np.asarray(values[1]), [[-5.0, -10.0], [-15.0, -20.0]])


def test_windows_of_channels_of_different_shapes_evaluate_to_leaves_of_each_shape():
    # A's observations are 10 * t and B's are the points (-10 * t, -20 * t).
    # The windows are [0.0, 1.0) and [1.0, 2.0).
    times = TimeDeltaTensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], "s")
    a = TimeSeries([0.0, 5.0, 10.0, 15.0, 20.0, 25.0], times=times)
    b = TimeSeries([[0.0, 0.0], [-5.0, -10.0], [-10.0, -20.0],
                    [-15.0, -30.0], [-20.0, -40.0], [-25.0, -50.0]], times=times)
    series = TimeSeries.from_channels([a, b], unit="s")
    windows = cut_windows(series, size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left", partial=False,
                          anchor="start")

    values = windows(TimeDeltaTensor([0.0, 0.5], "s"))

    assert np.array_equal(np.asarray(values[0]), [[0.0, 5.0], [10.0, 15.0]])
    assert np.array_equal(np.asarray(values[1]), [
        [[0.0, 0.0], [-5.0, -10.0]],
        [[-10.0, -20.0], [-15.0, -30.0]],
    ])


def test_evaluating_windows_outside_a_window_raises_naming_its_position():
    # The windows are [0.0, 2.0) and the partial window [2.0, 3.0], which can
    # be evaluated from 0 to 1.
    series = TimeSeries([0.0, 10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "s"))
    windows = cut_windows(series, size=TimeDelta(2, "s"), step=TimeDelta(2, "s"), closed="left", partial=True,
                          anchor="start")

    message = "TimeSeries at position (1): Time 1.5 is outside the domain [0, 1]"
    with pytest.raises(ValueError, match=re.escape(message)):
        windows(TimeDeltaTensor([0.0, 1.5], "s"))
    message = "TimeSeries at position (0): Time 2.5 is outside the domain [0, 2)"
    with pytest.raises(ValueError, match=re.escape(message)):
        windows(TimeDeltaTensor([0.0, 2.5, 1.0], "s"))


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
@pytest.mark.parametrize("make_values", TWO_ELEMENTS_OF_EVERY_DTYPE)
def test_windows_of_every_dtype_survive_saving(make_values, round_trip):
    # The observations are a, b, b, a, so the windows [0.0, 1.0] and
    # [0.5, 1.5] hold a, b, b and b, b, a.
    two = make_values()
    series = TimeSeries(sb.concatenate([two, two[[1, 0]]]), times=TimeDeltaTensor([0.0, 0.5, 1.0, 1.5], "s"))
    windows = cut_windows(series, size=TimeDelta(1, "s"), step=TimeDelta(0.5, "s"), closed="both", partial=False,
                          anchor="start")

    loaded = round_trip(windows)

    assert isinstance(loaded, TimeSeriesTensor)
    assert loaded.dtype is windows.dtype
    assert loaded.array_equal(windows)


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
def test_window_ends_survive_saving(closed, round_trip):
    windows = cut_windows(_closed_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed=closed,
                          partial=False, anchor="start")

    loaded = round_trip(windows)

    assert loaded.array_equal(windows)


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
def test_window_origins_survive_saving(round_trip):
    windows = cut_windows(_windows_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left",
                          partial=False, anchor="start")

    loaded = round_trip(windows)

    origins = [loaded[k].origin for k in range(loaded.shape[0])]
    assert origins == [TimeDelta(start, "s") for start in [0, 1, 2, 3]]


def _windows_of_pattern(start):
    # The pattern 0, 1 repeats every 2 seconds from start.
    series = TimeSeries([0.0, 1.0, 0.0, 1.0, 0.0], times=TimeDeltaTensor(start + np.arange(5.0), "s"))
    return cut_windows(series, size=TimeDelta(2, "s"), step=TimeDelta(2, "s"), closed="left", partial=False,
                       anchor="start")


def test_windows_with_the_same_contents_from_different_places_are_array_equal():
    early = _windows_of_pattern(0.0)
    late = _windows_of_pattern(10.0)

    assert early[0].array_equal(late[0])
    assert (early[0].origin, late[0].origin) == (TimeDelta(0, "s"), TimeDelta(10, "s"))


def test_tensors_of_windows_with_the_same_contents_from_different_places_are_array_equal():
    assert _windows_of_pattern(0.0).array_equal(_windows_of_pattern(10.0))


def test_series_differing_only_in_epoch_are_array_equal():
    assert _dated_source().array_equal(_windows_source())


def test_tensors_of_windows_differing_only_in_epoch_are_array_equal():
    dated = cut_windows(_dated_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left",
                        partial=False, anchor="start")

    assert dated.array_equal(cut_windows(_windows_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"),
                                         closed="left", partial=False, anchor="start"))


def test_window_dates_include_their_origin():
    window = cut_windows(_dated_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left", partial=False,
                         anchor="start")[2]

    # Window 2 starts at 2 s and holds the observation at 2.5 s.
    dates = window.datetimes.to_datetime64()

    assert np.array_equal(dates, np.array(["2023-01-01T00:00:02", "2023-01-01T00:00:02.5"], dtype="datetime64[ms]"))


def test_windows_of_a_fractional_time_delta_convert_it_exactly():
    # A third of a second is 333333333.33... ns; rounded to microseconds it
    # would be 333333000 ns, and the observation at 333333100 ns would fall
    # in the second window.
    series = TimeSeries([1.0, 2.0, 3.0], times=TimeDeltaTensor([0.0, 333_333_100.0, 1_000_000_000.0], "ns"))

    windows = cut_windows(series, size=TimeDelta(1 / 3, "s"), step=TimeDelta(1 / 3, "s"), closed="left", partial=False,
                          anchor="start")

    assert windows[0].n_observations == 2


def _hourly_source():
    return TimeSeries(np.arange(10.0), times=TimeDeltaTensor(np.arange(10.0), "h"))


def test_windows_of_a_timedelta64_size_and_step_equal_those_of_their_time_deltas():
    windows = cut_windows(_hourly_source(), size=np.timedelta64(2, "h"), step=np.timedelta64(30, "m"),
                          closed="left", partial=False, anchor="start")

    assert windows.array_equal(cut_windows(_hourly_source(), size=TimeDelta(2, "h"), step=TimeDelta(30, "min"),
                                           closed="left", partial=False, anchor="start"))


def test_window_step_defaults_to_its_size():
    windows = cut_windows(_hourly_source(), size=TimeDelta(2, "h"), closed="left", partial=False, anchor="start")

    assert windows.array_equal(cut_windows(_hourly_source(), size=TimeDelta(2, "h"), step=TimeDelta(2, "h"),
                                           closed="left", partial=False, anchor="start"))


@pytest.mark.parametrize("size", [
    pytest.param(np.timedelta64(1, "M"), id="months"),
    pytest.param(np.timedelta64(1, "Y"), id="years"),
])
def test_window_rejects_durations_without_a_fixed_length(size):
    with pytest.raises(ValueError, match="fixed length"):
        cut_windows(_windows_source(), size=size)


def test_a_window_source_domain_is_its_domain_moved_by_its_origin():
    window = cut_windows(_windows_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left",
                         partial=False, anchor="start")[2]

    # Window 2 is evaluated on [0, 1) and lies on [2, 3) of the series.
    assert window.domain == Domain(TimeDelta(0, "s"), TimeDelta(1, "s"), True, False)
    assert window.source_domain == Domain(TimeDelta(2, "s"), TimeDelta(3, "s"), True, False)


def test_the_source_domain_of_a_series_built_from_observations_is_its_domain():
    series = _windows_source()

    assert series.source_domain == series.domain


def test_the_source_domain_of_a_dated_window_is_its_domain():
    window = cut_windows(_dated_source(), size=TimeDelta(1, "s"), step=TimeDelta(1, "s"), closed="left",
                         partial=False, anchor="start")[2]

    assert window.source_domain == window.domain


def test_the_source_domain_of_a_window_of_a_series_without_a_unit_is_numbers():
    series = TimeSeries([0.0, 5.0, 15.0, 25.0, 40.0], times=[0.0, 0.5, 1.5, 2.5, 4.0])

    window = cut_windows(series, size=2.0, step=2.0, closed="left", partial=False, anchor="start")[1]

    assert window.source_domain == Domain(2.0, 4.0, True, False)


def test_the_source_domain_of_channels_at_different_origins_is_the_intersection_of_where_each_lies():
    series = TimeSeries(np.arange(9.0), times=TimeDeltaTensor(np.arange(9.0), "s"))
    windows = cut_windows(series, size=TimeDelta(3, "s"), step=TimeDelta(2, "s"), closed="left", partial=False,
                          anchor="start")
    # The channels lie on [0, 3) and [2, 5) of the series.
    channels = TimeSeries.from_channels([windows[0], windows[1]])

    assert channels.source_domain == Domain(TimeDelta(2, "s"), TimeDelta(3, "s"), True, False)


def test_window_of_pairs_of_count_and_unit_equals_the_window_of_their_time_deltas():
    windows = cut_windows(_windows_source(), size=(2, "s"), step=(500, "ms"))

    assert windows.array_equal(cut_windows(_windows_source(), size=TimeDelta(2, "s"), step=TimeDelta(500, "ms")))


def test_windows_of_an_array_raise_naming_the_time_series_constructor():
    with pytest.raises(TypeError, match=re.escape("TimeSeries(values, times)")):
        cut_windows(np.arange(3.0), size=TimeDelta(1, "s"))  # pyright: ignore[reportArgumentType]
