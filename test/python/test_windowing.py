import numpy as np
import pytest

import stablebear as sb
from stablebear import FloatTensor
from stablebear.timeseries import TimeDelayEmbedding, Times, TimeSeries, Window
from stablebear.timeseries.series import TimeSeriesTensor


def _windows_source():
    # Each observation is 10 * t.
    return TimeSeries([0.0, 5.0, 15.0, 25.0, 40.0], times=[0.0, 0.5, 1.5, 2.5, 4.0])


def test_windows_are_a_tensor_of_series_of_the_source_dtype():
    windows = Window(size=2.0, step=2.0)(_windows_source())

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
    pytest.param(Window(size=2.0, step=1.0), [0.0, 1.0, 2.0], id="step"),
    pytest.param(Window(size=1.0, step=1.5), [0.0, 1.5, 3.0], id="gaps"),
    pytest.param(Window(size=2.0), [0.0, 2.0], id="default-step"),
])
def test_windows_start_every_step_while_they_fit(window, starts):
    windows, bounds = window(_windows_source(), return_bounds=True)

    assert windows.shape == (len(starts),)
    assert isinstance(bounds, FloatTensor)
    assert bounds.dtype is sb.float64
    assert bounds.shape == (len(starts), 2)
    assert np.array_equal(np.asarray(bounds)[:, 0], starts)
    assert np.array_equal(np.asarray(bounds)[:, 1], np.add(starts, window.size))


@pytest.mark.parametrize("series, match", [
    pytest.param(_windows_source(), "longer than the series", id="longer-than-the-series"),
    pytest.param(TimeSeries([], times=[]), "without observations", id="empty"),
])
def test_windowing_raises_when_no_window_fits(series, match):
    with pytest.raises(ValueError, match=match):
        Window(size=5.0, step=1.0)(series)


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


def test_a_window_can_be_evaluated_from_its_start_up_to_its_end():
    # The window [1.0, 3.0) holds the value at 1.0, observed at 0.5, then 1.5
    # and 2.5.
    window = Window(size=2.0, step=1.0)(_windows_source())[1]

    assert np.array_equal(np.asarray(window.times), [0.0, 0.5, 1.5])
    assert np.array_equal(np.asarray(window.values), [5.0, 15.0, 25.0])
    assert window(0.0) == 5.0
    assert window(1.75) == 25.0
    with pytest.raises(ValueError, match=r"outside the domain \[0, 2\)"):
        window(2.0)


def test_an_observation_at_a_window_boundary_is_only_in_the_later_window():
    windows = Window(size=1.5, step=1.5)(_windows_source())

    # The observation at 1.5 starts the second window, not the first.
    assert np.array_equal(np.asarray(windows[0].values), [0.0, 5.0])
    assert np.array_equal(np.asarray(windows[1].values), [15.0, 25.0])


def test_a_window_between_observations_holds_the_value_at_its_start():
    # The window [3.0, 3.5) falls between the observations at 2.5 and 4.0.
    window = Window(size=0.5, step=0.5)(_windows_source())[6]

    assert np.array_equal(np.asarray(window.times), [0.0])
    assert np.array_equal(np.asarray(window.values), [25.0])
    assert window(0.25) == 25.0


def test_window_bounds_just_below_an_observation_reach_it():
    series = TimeSeries([0.0, 3.0, 6.0, 9.0, 12.0], times=[0.0, 0.3, 0.6, 0.9, 1.2])

    windows, bounds = Window(size=0.3, step=0.3)(series, return_bounds=True)

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
    window = Window(size=2.0, step=1.0)(_windows_source())[1]

    embedding = TimeDelayEmbedding(dimension=2, delay=0.4)(window)

    assert np.array_equal(np.asarray(embedding.times), [0.4, 0.5, 0.9, 1.5, 1.9])
    assert np.array_equal(np.asarray(embedding(1.95)), [25.0, 25.0])
    with pytest.raises(ValueError, match=r"outside the domain \[0.4, 2\)"):
        embedding(2.0)


def test_an_embedding_of_a_window_takes_no_history_from_before_the_window():
    # The window [1.0, 3.0) holds 5, 15, and 25 at 0.0, 0.5, and 1.5. With a
    # delay of 1.0, the first point is at 1.0.
    window = Window(size=2.0, step=1.0)(_windows_source())[1]

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

    windows, bounds = Window(size=2.0, step=1.0)(embedding, return_bounds=True)

    assert np.array_equal(np.asarray(bounds)[:, 0], [1.0, 2.0])
    assert np.array_equal(np.asarray(windows[0].times), [0.0, 0.5, 1.5])
    assert np.array_equal(np.asarray(windows[0].values), [[0.0, 5.0], [5.0, 15.0], [15.0, 25.0]])


def test_windows_cut_every_channel_at_the_same_bounds():
    # A's observations are 10 * t and B's are -10 * t, with t in seconds.
    a = TimeSeries([0.0, 10.0, 25.0], times=Times([0.0, 1.0, 2.5], unit="s"))
    b = TimeSeries([-5.0, -15.0, -30.0], times=Times([500.0, 1500.0, 3000.0], unit="ms"))
    series = TimeSeries.from_channels([a, b], unit="s")

    windows, bounds = Window(size=1.0, step=1.0)(series, return_bounds=True)

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
