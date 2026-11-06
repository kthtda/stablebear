import numpy as np
import pytest

import stablebear as sb
from stablebear.timeseries import Domain, TimeDelta, TimeDeltaTensor, TimeSeries, to_point_clouds


def _series():
    """The values 0, 10, 20 at times 0, 1, 2 s, on the domain [0, 2] s."""
    return TimeSeries([0.0, 10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "s"))


def _points(cloud):
    return np.asarray(cloud).ravel().tolist()


@pytest.mark.parametrize("start_closed, end_closed, expected", [
    pytest.param(True, True, [0.0, 10.0, 20.0], id="closed"),         # times 0, 1, 2
    pytest.param(True, False, [0.0, 0.0, 10.0], id="closed-start"),   # times 0, 2/3, 4/3
    pytest.param(False, True, [0.0, 10.0, 20.0], id="closed-end"),    # times 2/3, 4/3, 2
    pytest.param(False, False, [0.0, 10.0, 10.0], id="open"),         # times 0.5, 1, 1.5
])
def test_n_points_are_evenly_spaced_over_the_domain_including_its_closed_ends(start_closed, end_closed, expected):
    series = _series().restrict(Domain(TimeDelta(0.0, "s"), TimeDelta(2.0, "s"), start_closed=start_closed,
                                       end_closed=end_closed))

    cloud = to_point_clouds(series, n=3)

    assert _points(cloud) == expected


@pytest.mark.parametrize("start_closed, end_closed, expected", [
    pytest.param(True, False, [0.0], id="closed-start"),  # time 0
    pytest.param(False, True, [20.0], id="closed-end"),   # time 2
    pytest.param(False, False, [10.0], id="open"),        # time 1
])
def test_one_point_lies_at_the_closed_end_or_in_the_middle(start_closed, end_closed, expected):
    series = _series().restrict(Domain(TimeDelta(0.0, "s"), TimeDelta(2.0, "s"), start_closed=start_closed,
                                       end_closed=end_closed))

    cloud = to_point_clouds(series, n=1)

    assert _points(cloud) == expected


def test_n_and_step_give_the_same_cloud_where_they_agree():
    by_n = to_point_clouds(_series(), n=5)
    by_step = to_point_clouds(_series(), step=TimeDelta(0.5, "s"))

    assert _points(by_n) == _points(by_step) == [0.0, 0.0, 10.0, 10.0, 20.0]


def test_each_series_of_a_tensor_gives_a_cloud_of_n_points_over_its_own_domain():
    series_tensor = sb.zeros((2,), dtype=sb.ts(sb.float64))
    # The values are 10 and 100 times the times of the observations.
    series_tensor[0] = TimeSeries([0.0, 5.0, 15.0, 20.0], times=TimeDeltaTensor([0.0, 0.5, 1.5, 2.0], "s"))
    series_tensor[1] = TimeSeries([0.0, 50.0, 100.0, 250.0, 300.0, 400.0],
                                  times=TimeDeltaTensor([0.0, 0.5, 1.0, 2.5, 3.0, 4.0], "s"))

    clouds = to_point_clouds(series_tensor, n=3)

    assert _points(clouds[0]) == [0.0, 5.0, 20.0]     # times 0, 1, 2
    assert _points(clouds[1]) == [0.0, 100.0, 400.0]  # times 0, 2, 4


def test_step_that_does_not_divide_the_domain_stops_inside_it():
    cloud = to_point_clouds(_series(), step=TimeDelta(1500.0, "ms"))

    assert _points(cloud) == [0.0, 10.0]  # times 0, 1.5


def test_step_leaves_out_an_open_start():
    series = _series().restrict(Domain(TimeDelta(0.0, "s"), TimeDelta(2.0, "s"), start_closed=False,
                                       end_closed=True))

    cloud = to_point_clouds(series, step=TimeDelta(1.0, "s"))

    assert _points(cloud) == [10.0, 20.0]  # times 1, 2


def test_step_of_a_series_without_a_unit_is_a_number():
    series = TimeSeries([0.0, 10.0, 20.0], times=[0.0, 1.0, 2.0])

    cloud = to_point_clouds(series, step=2.0)

    assert _points(cloud) == [0.0, 20.0]  # times 0, 2


@pytest.mark.parametrize("kwargs, error, message", [
    pytest.param({"n": 2, "step": TimeDelta(1.0, "s")}, ValueError, "at most one", id="both"),
    pytest.param({"n": 0}, ValueError, "positive", id="n-zero"),
    pytest.param({"n": 1}, ValueError, "both ends", id="n-one-closed"),
    pytest.param({"step": TimeDelta(0.0, "s")}, ValueError, "finite and positive", id="step-zero"),
    pytest.param({"step": 1.0}, TypeError, "durations", id="step-number-with-unit"),
])
def test_invalid_grid_raises(kwargs, error, message):
    with pytest.raises(error, match=message):
        to_point_clouds(_series(), **kwargs)


def test_duration_step_on_a_series_without_a_unit_raises():
    series = TimeSeries([0.0, 10.0, 20.0], times=[0.0, 1.0, 2.0])

    with pytest.raises(TypeError, match="plain numbers"):
        to_point_clouds(series, step=TimeDelta(1.0, "s"))
