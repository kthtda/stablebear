import numpy as np
import pytest

import stablebear as sb
from stablebear import FloatTensor
from stablebear.timeseries import TimeSeries, Window
from stablebear.timeseries.series import TimeSeriesTensor


def _windows_source():
    # Each observation is 10 * t.
    return TimeSeries([0.0, 5.0, 15.0, 25.0, 40.0], times=[0.0, 0.5, 1.5, 2.5, 4.0])


def test_windows_are_a_tensor_of_series_of_the_source_dtype():
    windows = Window(size=2.0, step=2.0)(_windows_source())

    # The windows start at 0.0 and 2.0. The first holds 0.0, 0.5, and 1.5,
    # and ends with the value at 2.0, observed at 1.5; the second starts with
    # that value and holds 2.5 and 4.0.
    assert isinstance(windows, TimeSeriesTensor)
    assert windows.dtype is sb.series(sb.float64)
    assert windows.shape == (2,)
    assert windows[0].dtype is sb.float64
    assert windows[1].dtype is sb.float64
    assert np.array_equal(np.asarray(windows[0].times), [0.0, 0.5, 1.5, 2.0])
    assert np.array_equal(np.asarray(windows[0].values), [0.0, 5.0, 15.0, 15.0])
    assert np.array_equal(np.asarray(windows[1].times), [0.0, 0.5, 2.0])
    assert np.array_equal(np.asarray(windows[1].values), [15.0, 25.0, 40.0])


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


def test_a_window_starts_and_ends_with_the_values_at_its_bounds():
    # The window from 1.0 to 3.0 holds 1.5 and 2.5 between the values at 1.0,
    # observed at 0.5, and at 3.0, observed at 2.5.
    window = Window(size=2.0, step=1.0)(_windows_source())[1]

    assert np.array_equal(np.asarray(window.times), [0.0, 0.5, 1.5, 2.0])
    assert np.array_equal(np.asarray(window.values), [5.0, 15.0, 25.0, 25.0])


def test_windows_include_both_ends():
    windows = Window(size=1.5, step=1.5)(_windows_source())

    # The observation at 1.5 ends the first window and starts the second.
    assert np.array_equal(np.asarray(windows[0].values), [0.0, 5.0, 15.0])
    assert np.array_equal(np.asarray(windows[1].values), [15.0, 25.0, 25.0])


def test_a_window_between_observations_holds_the_value_at_its_bounds():
    # The window from 3.0 to 3.5 falls between the observations at 2.5 and 4.0.
    window = Window(size=0.5, step=0.5)(_windows_source())[6]

    assert np.array_equal(np.asarray(window.times), [0.0, 0.5])
    assert np.array_equal(np.asarray(window.values), [25.0, 25.0])


def test_window_bounds_just_below_an_observation_reach_it():
    series = TimeSeries([0.0, 3.0, 6.0, 9.0, 12.0], times=[0.0, 0.3, 0.6, 0.9, 1.2])

    windows, bounds = Window(size=0.3, step=0.3)(series, return_bounds=True)

    # 3 * 0.3 computes just below 0.9, so window 2 ends and window 3 starts
    # there.
    assert np.asarray(bounds)[2, 1] < 0.9
    assert np.asarray(bounds)[3, 0] < 0.9
    # Window 2 ends at the observation at 0.9, and window 3 starts at it, at
    # time 0.
    assert np.array_equal(np.asarray(windows[2].values), [6.0, 9.0])
    assert np.asarray(windows[3].times)[0] == 0.0
    assert np.array_equal(np.asarray(windows[3].values), [9.0, 12.0])


def test_a_window_reads_its_own_observations_without_snapping():
    # In the window from -1.0, (-0.3 + 1.0) - 1.0 and (0.2 + 1.0) - 1.0 round
    # below -0.3 and 0.2, and snapping is off.
    series = TimeSeries([-10.0, -3.0, 2.0, 5.0], times=[-1.0, -0.3, 0.2, 0.5], snap_tolerance=0.0)

    window = Window(size=1.5, step=1.5)(series)[0]

    assert np.array_equal(np.asarray(window.values), [-10.0, -3.0, 2.0, 5.0])
