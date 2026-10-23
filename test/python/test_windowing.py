import re

import numpy as np
import pytest

import stablebear as sb
from stablebear import FloatTensor
from stablebear.timeseries import TimeDelayEmbedding, Times, TimeSeries, Window
from stablebear.timeseries.series import TimeSeriesTensor

from _element_values import TWO_ELEMENTS_OF_EVERY_DTYPE
from _round_trip import ROUND_TRIPS


def _windows_source():
    # Each observation is 10 * t.
    return TimeSeries([0.0, 5.0, 15.0, 25.0, 40.0], times=[0.0, 0.5, 1.5, 2.5, 4.0])


def test_windows_are_a_tensor_of_series_of_the_source_dtype():
    windows = Window(size=2.0, step=2.0, closed="left", partial=False)(_windows_source())

    # The windows cover [0.0, 2.0) and [2.0, 4.0). The first holds 0.0, 0.5,
    # and 1.5; the second starts with the value at 2.0, observed at 1.5, and
    # holds 2.5.
    assert isinstance(windows, TimeSeriesTensor)
    assert windows.dtype is sb.series(sb.float64)
    assert windows.shape == (2,)
    assert windows[0].dtype is sb.float64
    assert windows[1].dtype is sb.float64
    assert np.array_equal(np.asarray(windows[0].times), [0.0, 0.5, 1.5])
    assert np.array_equal(np.asarray(windows[0].values), [0.0, 5.0, 15.0])
    assert np.array_equal(np.asarray(windows[1].times), [0.0, 0.5])
    assert np.array_equal(np.asarray(windows[1].values), [15.0, 25.0])


@pytest.mark.parametrize("window, starts", [
    pytest.param(Window(size=2.0, step=1.0, closed="left", partial=False), [0.0, 1.0, 2.0], id="step"),
    pytest.param(Window(size=1.0, step=1.5, closed="left", partial=False), [0.0, 1.5, 3.0], id="gaps"),
    pytest.param(Window(size=2.0, closed="left", partial=False), [0.0, 2.0], id="default-step"),
])
def test_windows_start_every_step_while_they_fit(window, starts):
    windows, bounds = window(_windows_source(), return_bounds=True)

    assert windows.shape == (len(starts),)
    assert isinstance(bounds, FloatTensor)
    assert bounds.dtype is sb.float64
    assert bounds.shape == (len(starts), 2)
    assert np.array_equal(np.asarray(bounds)[:, 0], starts)
    assert np.array_equal(np.asarray(bounds)[:, 1], np.add(starts, window.size))


@pytest.mark.parametrize("window, bounds", [
    pytest.param(Window(size=2.0, step=1.0, closed="right", partial=False, anchor="end"),
                 [[0.0, 2.0],
                  [1.0, 3.0],
                  [2.0, 4.0]], id="step"),
    pytest.param(Window(size=1.0, step=1.5, closed="right", partial=False, anchor="end"),
                 [[0.0, 1.0],
                  [1.5, 2.5],
                  [3.0, 4.0]], id="gaps"),
])
def test_windows_anchored_at_the_end_end_every_step_back_while_they_fit(window, bounds):
    _, window_bounds = window(_windows_source(), return_bounds=True)

    # The last window ends where the series does, at 4.0, and the windows
    # come in chronological order.
    assert np.array_equal(np.asarray(window_bounds), bounds)


@pytest.mark.parametrize("anchor", ["start", "end"])
@pytest.mark.parametrize("series, match", [
    pytest.param(_windows_source(), "longer than the series", id="longer-than-the-series"),
    pytest.param(TimeSeries([], times=[]), "without observations", id="empty"),
])
def test_windowing_raises_when_no_window_fits(series, match, anchor):
    with pytest.raises(ValueError, match=match):
        Window(size=5.0, step=1.0, closed="left", partial=False, anchor=anchor)(series)


@pytest.mark.parametrize("size, step, match", [
    pytest.param(0.0, 1.0, "size", id="zero-size"),
    pytest.param(-1.0, 1.0, "size", id="negative-size"),
    pytest.param(np.inf, 1.0, "size", id="infinite-size"),
    pytest.param(-np.inf, 1.0, "size", id="negative-infinite-size"),
    pytest.param(np.nan, 1.0, "size", id="nan-size"),
    pytest.param(1.0, 0.0, "step", id="zero-step"),
    pytest.param(1.0, -1.0, "step", id="negative-step"),
    pytest.param(1.0, np.inf, "step", id="infinite-step"),
    pytest.param(1.0, -np.inf, "step", id="negative-infinite-step"),
    pytest.param(1.0, np.nan, "step", id="nan-step"),
])
def test_window_raises_on_a_size_or_step_that_is_not_finite_and_positive(size, step, match):
    with pytest.raises(ValueError, match=match):
        Window(size=size, step=step)


def test_window_raises_on_an_unknown_closed():
    with pytest.raises(ValueError, match="closed"):
        Window(size=1.0, step=1.0, closed="middle")


def test_window_raises_on_an_unknown_anchor():
    with pytest.raises(ValueError, match="anchor"):
        Window(size=1.0, step=1.0, closed="left", anchor="middle")


@pytest.mark.parametrize("anchor, closed", [
    pytest.param("start", "left", id="start"),
    pytest.param("end", "right", id="end"),
])
def test_windows_are_closed_at_their_anchor_by_default(anchor, closed):
    assert Window(size=1.0, step=1.0, anchor=anchor).closed == closed


def _closed_source():
    # Each observation is 10 * t.
    return TimeSeries([0.0, 5.0, 10.0, 15.0, 20.0], times=[0.0, 0.5, 1.0, 1.5, 2.0])


def test_a_left_closed_window_includes_its_start_but_not_its_end():
    # The window [1.0, 2.0).
    window = Window(size=1.0, step=1.0, closed="left", partial=False)(_closed_source())[1]

    assert window(0.0) == 10.0
    with pytest.raises(ValueError, match=re.escape("outside the domain [0, 1)")):
        window(1.0)


def test_a_right_closed_window_includes_its_end_but_not_its_start():
    # The window (1.0, 2.0]. The observation at 1.0 stays in effect just
    # after the start.
    window = Window(size=1.0, step=1.0, closed="right", partial=False)(_closed_source())[1]

    with pytest.raises(ValueError, match=re.escape("outside the domain (0, 1]")):
        window(0.0)
    assert window(0.25) == 10.0
    assert window(1.0) == 20.0


def test_a_window_closed_on_both_sides_includes_its_start_and_its_end():
    # The window [1.0, 2.0].
    window = Window(size=1.0, step=1.0, closed="both", partial=False)(_closed_source())[1]

    assert window(0.0) == 10.0
    assert window(1.0) == 20.0


def test_a_window_closed_on_neither_side_includes_neither_end():
    # The window (1.0, 2.0). The observation at 1.0 stays in effect just
    # after the start.
    window = Window(size=1.0, step=1.0, closed="none", partial=False)(_closed_source())[1]

    with pytest.raises(ValueError, match=re.escape("outside the domain (0, 1)")):
        window(0.0)
    assert window(0.25) == 10.0
    with pytest.raises(ValueError, match=re.escape("outside the domain (0, 1)")):
        window(1.0)


def _partial_source():
    # Each observation is 10 * t.
    return TimeSeries([0.0, 10.0, 20.0, 30.0, 40.0], times=[0.0, 1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("partial, expected_bounds", [
    pytest.param(False, [[0.0, 3.0]], id="full"),
    pytest.param(True, [[0.0, 3.0], [3.0, 4.0]], id="partial"),
])
def test_partial_windows_continue_to_the_end_of_the_series(partial, expected_bounds):
    _, bounds = Window(size=3.0, step=3.0, closed="left", partial=partial)(_partial_source(), return_bounds=True)

    # A partial window's bounds end where the series does.
    assert np.array_equal(np.asarray(bounds), expected_bounds)


def test_a_partial_window_ends_where_the_series_does():
    # The window at 3.0 would end at 6.0, after the series ends at 4.0.
    window = Window(size=3.0, step=3.0, closed="left", partial=True)(_partial_source())[1]

    assert np.array_equal(np.asarray(window.times), [0.0, 1.0])
    assert np.array_equal(np.asarray(window.values), [30.0, 40.0])
    assert window(1.0) == 40.0
    with pytest.raises(ValueError, match=re.escape("outside the domain [0, 1]")):
        window(1.5)


def test_a_window_anchored_at_the_end_holds_the_last_observation():
    # The window (1.0, 4.0] holds the value at 1.0 just after its start,
    # then the observations at 2.0, 3.0, and 4.0.
    window = Window(size=3.0, step=3.0, closed="right", partial=False, anchor="end")(_partial_source())[0]

    assert np.array_equal(np.asarray(window.times), [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(np.asarray(window.values), [10.0, 20.0, 30.0, 40.0])
    assert window(3.0) == 40.0


def test_a_partial_window_anchored_at_the_end_starts_where_the_series_does():
    windows, bounds = Window(size=3.0, step=3.0, closed="right", partial=True, anchor="end")(
        _partial_source(), return_bounds=True)

    # The window ending at 1.0 would start at -2.0, before the series starts
    # at 0.0, so it is [0.0, 1.0], closed at the start as the series is.
    assert np.array_equal(np.asarray(bounds), [
        [0.0, 1.0],
        [1.0, 4.0],
    ])
    assert np.array_equal(np.asarray(windows[0].times), [0.0, 1.0])
    assert np.array_equal(np.asarray(windows[0].values), [0.0, 10.0])
    assert windows[0](0.0) == 0.0
    assert windows[0](1.0) == 10.0
    # The full window (1.0, 4.0] holds the value at 1.0 just after its start,
    # then the observations at 2.0, 3.0, and 4.0.
    assert np.array_equal(np.asarray(windows[1].times), [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(np.asarray(windows[1].values), [10.0, 20.0, 30.0, 40.0])
    assert windows[1](3.0) == 40.0


def test_a_left_closed_partial_window_can_start_at_the_last_observation():
    windows, bounds = Window(size=2.0, step=2.0, closed="left", partial=True)(_partial_source(), return_bounds=True)

    # The window at 4.0 holds only the last observation.
    assert np.array_equal(np.asarray(bounds)[:, 0], [0.0, 2.0, 4.0])
    assert np.array_equal(np.asarray(windows[2].times), [0.0])
    assert windows[2](0.0) == 40.0


def test_a_right_closed_window_holds_the_last_observation_without_a_partial_window():
    windows, bounds = Window(size=2.0, step=2.0, closed="right", partial=True)(_partial_source(), return_bounds=True)

    # The window (2.0, 4.0] ends at the last observation, and no window can
    # start there with an open start.
    assert np.array_equal(np.asarray(bounds)[:, 0], [0.0, 2.0])
    assert windows[1](2.0) == 40.0


def test_a_window_can_be_evaluated_from_its_start_up_to_its_end():
    # The window [1.0, 3.0) holds the value at 1.0, observed at 0.5, then 1.5
    # and 2.5.
    window = Window(size=2.0, step=1.0, closed="left", partial=False)(_windows_source())[1]

    assert np.array_equal(np.asarray(window.times), [0.0, 0.5, 1.5])
    assert np.array_equal(np.asarray(window.values), [5.0, 15.0, 25.0])
    assert window(0.0) == 5.0
    assert window(1.75) == 25.0
    with pytest.raises(ValueError, match=r"outside the domain \[0, 2\)"):
        window(2.0)


def test_an_observation_at_a_window_boundary_is_only_in_the_later_window():
    windows = Window(size=1.5, step=1.5, closed="left", partial=False)(_windows_source())

    # The observation at 1.5 starts the second window, not the first.
    assert np.array_equal(np.asarray(windows[0].values), [0.0, 5.0])
    assert np.array_equal(np.asarray(windows[1].values), [15.0, 25.0])


def test_a_window_between_observations_holds_the_value_at_its_start():
    # The window [3.0, 3.5) falls between the observations at 2.5 and 4.0.
    window = Window(size=0.5, step=0.5, closed="left", partial=False)(_windows_source())[6]

    assert np.array_equal(np.asarray(window.times), [0.0])
    assert np.array_equal(np.asarray(window.values), [25.0])
    assert window(0.25) == 25.0


def test_window_bounds_just_below_an_observation_reach_it():
    series = TimeSeries([0.0, 3.0, 6.0, 9.0, 12.0], times=[0.0, 0.3, 0.6, 0.9, 1.2])

    windows, bounds = Window(size=0.3, step=0.3, closed="left", partial=False)(series, return_bounds=True)

    # 3 * 0.3 computes just below 0.9, so window 2 ends and window 3 starts
    # there.
    assert np.asarray(bounds)[2, 1] < 0.9
    assert np.asarray(bounds)[3, 0] < 0.9
    # The observation at 0.9 is the end of window 2, so only window 3 holds
    # it, at time 0.
    assert np.array_equal(np.asarray(windows[2].values), [6.0])
    assert np.array_equal(np.asarray(windows[3].times), [0.0])
    assert np.array_equal(np.asarray(windows[3].values), [9.0])


def test_an_embedding_of_a_window_keeps_its_open_end():
    # The window [1.0, 3.0) holds 5, 15, and 25 at 0.0, 0.5, and 1.5. With a
    # delay of 0.4, the newer component reaches 1.5 at 1.9, between the
    # window's last observation and its end.
    window = Window(size=2.0, step=1.0, closed="left", partial=False)(_windows_source())[1]

    embedding = TimeDelayEmbedding(dimension=2, delay=0.4)(window)

    assert np.array_equal(np.asarray(embedding.times), [0.4, 0.5, 0.9, 1.5, 1.9])
    assert np.array_equal(np.asarray(embedding(1.95)), [25.0, 25.0])
    with pytest.raises(ValueError, match=r"outside the domain \[0.4, 2\)"):
        embedding(2.0)


def test_an_embedding_of_a_window_keeps_its_open_start_and_closed_end():
    # The window (1.0, 3.0] holds 10 and 20 at 0.0 and 1.0, and ends at 2.0,
    # after its last observation.
    series = TimeSeries([0.0, 10.0, 20.0, 40.0], times=[0.0, 1.0, 2.0, 4.0])
    window = Window(size=2.0, step=1.0, closed="right", partial=False)(series)[1]

    embedding = TimeDelayEmbedding(dimension=2, delay=1.0)(window)

    assert np.array_equal(np.asarray(embedding.times), [1.0, 2.0])
    assert np.array_equal(np.asarray(embedding(2.0)), [20.0, 20.0])
    with pytest.raises(ValueError, match=re.escape("outside the domain (1, 2]")):
        embedding(1.0)


def test_an_embedding_of_a_window_takes_no_history_from_before_the_window():
    # The window [1.0, 3.0) holds 5, 15, and 25 at 0.0, 0.5, and 1.5. With a
    # delay of 1.0, the first point is at 1.0.
    window = Window(size=2.0, step=1.0, closed="left", partial=False)(_windows_source())[1]

    embedding = TimeDelayEmbedding(dimension=2, delay=1.0)(window)

    assert np.array_equal(np.asarray(embedding.times), [1.0, 1.5])
    assert np.array_equal(np.asarray(embedding.values), [[5.0, 15.0], [15.0, 25.0]])
    with pytest.raises(ValueError, match=r"outside the domain \[1, 2\)"):
        embedding(0.75)
    with pytest.raises(ValueError, match=r"outside the domain \[1, 2\)"):
        embedding(2.0)


def test_windows_of_an_embedding_hold_its_points():
    # The embedding has points at 1.0, 1.5, 2.5, 3.5, and 4.0.
    embedding = TimeDelayEmbedding(dimension=2, delay=1.0)(_windows_source())

    windows, bounds = Window(size=2.0, step=1.0, closed="left", partial=False)(embedding, return_bounds=True)

    assert np.array_equal(np.asarray(bounds)[:, 0], [1.0, 2.0])
    assert np.array_equal(np.asarray(windows[0].times), [0.0, 0.5, 1.5])
    assert np.array_equal(np.asarray(windows[0].values), [[0.0, 5.0], [5.0, 15.0], [15.0, 25.0]])


def test_windows_cut_every_channel_at_the_same_bounds():
    # A's observations are 10 * t and B's are -10 * t, with t in seconds.
    a = TimeSeries([0.0, 10.0, 25.0], times=Times([0.0, 1.0, 2.5], unit="s"))
    b = TimeSeries([-5.0, -15.0, -30.0], times=Times([500.0, 1500.0, 3000.0], unit="ms"))
    series = TimeSeries.from_channels([a, b], unit="s")

    windows, bounds = Window(size=1.0, step=1.0, closed="left", partial=False)(series, return_bounds=True)

    # Both channels are observed from 0.5 to 2.5, so the windows are
    # [0.5, 1.5) and [1.5, 2.5).
    assert np.array_equal(np.asarray(bounds)[:, 0], [0.5, 1.5])
    # B's observation at 1500 ms is the end of the first window.
    assert np.array_equal(np.asarray(windows[0].times[0]), [0.0, 0.5])
    assert np.array_equal(np.asarray(windows[0].times[1]), [0.0])
    assert np.array_equal(np.asarray(windows[1].times[0]), [0.0])
    assert np.array_equal(np.asarray(windows[1].times[1]), [0.0])
    assert windows[0](0.5)[0][()] == 10.0
    assert windows[0](0.5)[1][()] == -5.0
    with pytest.raises(ValueError, match=r"outside the domain \[0, 1\)"):
        windows[0](1.0)


def _overlapping_windows():
    # Windows at 0.0, 1.0, and 2.0.
    series = TimeSeries([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
                        times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    return Window(size=2.0, step=1.0, closed="left", partial=False)(series)


def test_windows_evaluate_at_times_in_window_order():
    windows = _overlapping_windows()

    values = windows([0.0, 1.0])

    assert isinstance(values, FloatTensor)
    assert np.array_equal(np.asarray(values), [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]])
    assert np.array_equal(np.asarray(windows(1.5)), [15.0, 25.0, 35.0])


def test_windows_evaluate_at_times_of_any_shape():
    windows = _overlapping_windows()

    values = windows([[0.0, 1.0], [1.5, 0.5]])

    assert np.array_equal(np.asarray(values), [
        [[0.0, 10.0], [15.0, 5.0]],
        [[10.0, 20.0], [25.0, 15.0]],
        [[20.0, 30.0], [35.0, 25.0]],
    ])


def test_windows_evaluate_at_times_in_the_given_unit():
    windows = _overlapping_windows()

    values = windows([0.0, 1000.0], "ms")

    assert np.array_equal(np.asarray(values), [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]])


def test_windows_reject_an_unknown_unit():
    windows = _overlapping_windows()

    with pytest.raises(ValueError, match="Unknown time unit"):
        windows([0.0, 1.0], "fortnight")


def test_windows_of_several_channels_evaluate_to_one_leaf_per_channel():
    # A's observations are 10 * t and B's are -10 * t, with t in seconds; B
    # starts 0.5 s after A. Both are observed from 0.5 to 2.5, so the windows
    # are [0.5, 1.5) and [1.5, 2.5).
    a = TimeSeries([0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
                   times=Times([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], unit="s"))
    b = TimeSeries([-5.0, -10.0, -15.0, -20.0, -25.0, -30.0],
                   times=Times([500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0], unit="ms"))
    series = TimeSeries.from_channels([a, b], unit="s")
    windows = Window(size=1.0, step=1.0, closed="left", partial=False)(series)

    values = windows([0.0, 0.5])

    assert isinstance(values, sb.NestedTensor)
    assert np.array_equal(np.asarray(values[0]), [[5.0, 10.0], [15.0, 20.0]])
    assert np.array_equal(np.asarray(values[1]), [[-5.0, -10.0], [-15.0, -20.0]])


def test_windows_of_channels_of_different_shapes_evaluate_to_leaves_of_each_shape():
    # A's observations are 10 * t and B's are the points (-10 * t, -20 * t).
    # The windows are [0.0, 1.0) and [1.0, 2.0).
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    a = TimeSeries([0.0, 5.0, 10.0, 15.0, 20.0, 25.0], times=times)
    b = TimeSeries([[0.0, 0.0], [-5.0, -10.0], [-10.0, -20.0],
                    [-15.0, -30.0], [-20.0, -40.0], [-25.0, -50.0]], times=times)
    series = TimeSeries.from_channels([a, b], unit="s")
    windows = Window(size=1.0, step=1.0, closed="left", partial=False)(series)

    values = windows([0.0, 0.5])

    assert np.array_equal(np.asarray(values[0]), [[0.0, 5.0], [10.0, 15.0]])
    assert np.array_equal(np.asarray(values[1]), [
        [[0.0, 0.0], [-5.0, -10.0]],
        [[-10.0, -20.0], [-15.0, -30.0]],
    ])


def test_evaluating_windows_outside_a_window_raises_naming_its_position():
    # The windows are [0.0, 2.0) and the partial window [2.0, 3.0], which can
    # be evaluated from 0 to 1.
    series = TimeSeries([0.0, 10.0, 20.0, 30.0], times=[0.0, 1.0, 2.0, 3.0])
    windows = Window(size=2.0, step=2.0, closed="left", partial=True)(series)

    message = "TimeSeries at position (1): Time 1.5 is outside the domain [0, 1]"
    with pytest.raises(ValueError, match=re.escape(message)):
        windows([0.0, 1.5])
    message = "TimeSeries at position (0): Time nan is outside the domain [0, 2)"
    with pytest.raises(ValueError, match=re.escape(message)):
        windows([0.0, np.nan, 1.0])


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
@pytest.mark.parametrize("make_values", TWO_ELEMENTS_OF_EVERY_DTYPE)
def test_windows_of_every_dtype_survive_saving(make_values, round_trip):
    # The observations are a, b, b, a, so the windows [0.0, 1.0] and
    # [0.5, 1.5] hold a, b, b and b, b, a.
    two = make_values()
    series = TimeSeries(sb.concatenate([two, two[[1, 0]]]), times=[0.0, 0.5, 1.0, 1.5])
    windows = Window(size=1.0, step=0.5, closed="both", partial=False)(series)

    loaded = round_trip(windows)

    assert isinstance(loaded, TimeSeriesTensor)
    assert loaded.dtype is windows.dtype
    assert loaded.array_equal(windows)


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
def test_window_ends_survive_saving(closed, round_trip):
    windows = Window(size=1.0, step=1.0, closed=closed, partial=False)(_closed_source())

    loaded = round_trip(windows)

    assert loaded.array_equal(windows)
