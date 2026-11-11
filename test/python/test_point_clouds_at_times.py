import numpy as np
import pytest

import stablebear as sb
from stablebear.timeseries import (DateTimeTensor, OutsideDomainError, TimeDelta, TimeDeltaTensor, TimeSeries, cut_windows,
                                   to_point_clouds)


def _series():
    """The values 0, 10, 20 at times 0, 1, 2 s."""
    return TimeSeries([0.0, 10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "s"))


def _points(cloud):
    return np.asarray(cloud).ravel().tolist()


def test_points_at_given_times_are_the_values_in_effect_then():
    cloud = to_point_clouds(_series(), TimeDeltaTensor([1500.0, 0.0, 500.0], "ms"))

    assert _points(cloud) == [10.0, 0.0, 0.0]  # times 1.5, 0, 0.5 s


def test_points_at_given_times_keep_their_order():
    cloud = to_point_clouds(_series(), TimeDeltaTensor([2.0, 0.0, 1.0], "s"))

    assert _points(cloud) == [20.0, 0.0, 10.0]


def test_given_times_of_a_series_without_a_unit_are_numbers():
    series = TimeSeries([0.0, 10.0, 20.0], times=[0.0, 1.0, 2.0])

    cloud = to_point_clouds(series, [0.5, 2.0])

    assert _points(cloud) == [0.0, 20.0]


def test_given_times_give_every_series_of_a_tensor_a_cloud_at_those_times():
    series_tensor = sb.zeros((2,), dtype=sb.ts(sb.float64))
    series_tensor[0] = _series()
    series_tensor[1] = TimeSeries([5.0, 15.0], times=TimeDeltaTensor([0.0, 1.5], "s"))

    times = TimeDeltaTensor([0.0, 1.0], "s")

    clouds = to_point_clouds(series_tensor, times)

    assert _points(clouds[0]) == _points(series_tensor[0](times))
    assert _points(clouds[1]) == _points(series_tensor[1](times))


@pytest.mark.parametrize("times, error, message", [
    pytest.param(TimeDeltaTensor([1.0, 3.0], "s"), ValueError, "outside the domain", id="outside"),
    pytest.param(TimeDeltaTensor([[0.0, 1.0]], "s"), ValueError, "one-dimensional", id="two-dimensional"),
    pytest.param([0.0, 1.0], TypeError, "durations", id="numbers-with-unit"),
])
def test_invalid_given_times_raise(times, error, message):
    with pytest.raises(error, match=message):
        to_point_clouds(_series(), times)


def test_duration_times_on_a_series_without_a_unit_raise():
    series = TimeSeries([0.0, 10.0, 20.0], times=[0.0, 1.0, 2.0])

    with pytest.raises(TypeError, match="plain numbers"):
        to_point_clouds(series, TimeDeltaTensor([0.0, 1.0], "s"))


def test_given_times_and_n_raise():
    with pytest.raises(ValueError, match="at most one"):
        to_point_clouds(_series(), TimeDeltaTensor([0.0], "s"), n=2)


def test_dates_place_points_at_their_durations_from_the_epoch():
    series = _series().with_epoch("2023-01-01")

    by_date = to_point_clouds(series, DateTimeTensor(["2023-01-01T00:00:02", "2023-01-01T00:00:01"]))
    by_duration = to_point_clouds(series, TimeDeltaTensor([2.0, 1.0], "s"))

    assert _points(by_date) == _points(by_duration)


def test_dates_give_every_series_of_a_tensor_a_cloud_at_those_instants():
    early = _series().with_epoch("2023-01-01T00:00:00")
    late = _series().with_epoch("2023-01-01T00:00:01")
    dates = DateTimeTensor(["2023-01-01T00:00:02", "2023-01-01T00:00:01"])

    clouds = to_point_clouds(sb.tensor([early, late]), dates)

    assert _points(clouds[0]) == _points(to_point_clouds(early, dates))
    assert _points(clouds[1]) == _points(to_point_clouds(late, dates))


def test_a_date_outside_a_series_of_a_tensor_raises_with_its_index_and_domain():
    early = _series().with_epoch("2023-01-01T00:00:00")
    late = _series().with_epoch("2023-01-01T00:00:01")

    # 00:00:00 is before the domain of late, which starts at 00:00:01.
    with pytest.raises(OutsideDomainError) as excinfo:
        to_point_clouds(sb.tensor([early, late]), DateTimeTensor(["2023-01-01T00:00:00"]))

    assert excinfo.value.index == (1,)
    assert excinfo.value.domain == late.domain


def test_a_tensor_of_a_window_gives_a_point_at_its_start_date():
    # Observed every 6 minutes and stored in hours, the window from 00:18
    # has its origin, three steps of 0.1 h, at 0.30000000000000004, while
    # 00:18 counts as 0.3 h from the epoch.
    times = np.arange("2023-01-01T00:00", "2023-01-01T02:06", np.timedelta64(6, "m"), dtype="datetime64[m]")
    series = TimeSeries(np.arange(21.0), times=DateTimeTensor(times)).to("h")
    window = cut_windows(series, size=TimeDelta(1.0, "h"), step=TimeDelta(6.0, "min"), closed="left", partial=False,
                         anchor="start")[3]

    clouds = to_point_clouds(sb.tensor([window]), DateTimeTensor(["2023-01-01T00:18"]))

    assert _points(clouds[0]) == [3.0]
