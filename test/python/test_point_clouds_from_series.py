import numpy as np
import pytest

import stablebear as sb
from stablebear.timeseries import TimeDelta, TimeDeltaTensor, TimeSeries, cut_windows, delay_embed, to_point_clouds


def _clouds(tensor):
    """The clouds of a 1-D PointCloudTensor as lists of points."""
    return [np.asarray(tensor[i]).tolist() for i in range(tensor.shape[0])]


def test_point_cloud_of_a_delay_embedding_has_its_points():
    series = TimeSeries([0.0, 10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "s"))
    embedding = delay_embed(series, dimension=2, delay=TimeDelta(1.0, "s"))

    cloud = to_point_clouds(embedding)

    assert isinstance(cloud, sb.PointCloud)
    assert np.array_equal(np.asarray(cloud), [
        [0.0, 10.0],   # 1: x(0), x(1)
        [10.0, 20.0],  # 2: x(1), x(2)
        [20.0, 30.0],  # 3: x(2), x(3)
    ])


def test_point_clouds_of_windows_have_a_point_per_observation_in_the_window():
    series = TimeSeries([0.0, 10.0, 20.0, 40.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 4.0], "s"))
    # Each window (a, a + 2] gives a point for every observation inside it,
    # but none for the value it stores at its open start, at time 0.
    #
    # source   0    1    2    3    4
    # values   0    10   20        40
    # window 0 (---------]
    # window 1      (---------]
    # window 2           (---------]
    windows = cut_windows(series, size=TimeDelta(2.0, "s"), step=TimeDelta(1.0, "s"), closed="right",
                          partial=False, anchor="start")

    clouds = to_point_clouds(windows)

    assert isinstance(clouds, sb.PointCloudTensor)
    assert clouds.dtype is sb.pcloud64
    assert _clouds(clouds) == [
        [[10.0], [20.0]],
        [[20.0]],
        [[40.0]],
    ]


def test_point_clouds_of_partial_windows_are_smaller():
    series = TimeSeries([0.0, 10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "s"))
    # The last window [2, 4) ends where the series does, at 2.
    windows = cut_windows(series, size=TimeDelta(2.0, "s"), step=TimeDelta(1.0, "s"), closed="left",
                          partial=True, anchor="start")

    clouds = to_point_clouds(windows)

    assert _clouds(clouds) == [
        [[0.0], [10.0]],
        [[10.0], [20.0]],
        [[20.0]],
    ]


def test_point_cloud_of_scalars_is_one_dimensional():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([1.0, 2.0], "s"))

    cloud = to_point_clouds(series)

    assert cloud.shape == (2, 1)


def test_point_cloud_of_vectors_has_their_dimension():
    series = TimeSeries([[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]], times=TimeDeltaTensor([1.0, 2.0], "s"))

    cloud = to_point_clouds(series)

    assert np.array_equal(np.asarray(cloud), [
        [10.0, 20.0, 30.0],
        [20.0, 40.0, 60.0],
    ])


# Integers give float64 coordinates, even 32-bit ones: float32 holds
# integers exactly only up to 2^24, and float64 holds every 32-bit integer.
@pytest.mark.parametrize("values, expected_dtype", [
    pytest.param(np.array([10.0, 20.0], dtype=np.float32), sb.float32, id="float32"),
    pytest.param(np.array([10.0, 20.0], dtype=np.float64), sb.float64, id="float64"),
    pytest.param(np.array([10, 20], dtype=np.int32), sb.float64, id="int32"),
    pytest.param(np.array([10, 20], dtype=np.int64), sb.float64, id="int64"),
    pytest.param(np.array([10, 20], dtype=np.uint32), sb.float64, id="uint32"),
    pytest.param(np.array([10, 20], dtype=np.uint64), sb.float64, id="uint64"),
])
def test_point_cloud_is_float32_only_for_float32_series(values, expected_dtype):
    series = TimeSeries(values, times=TimeDeltaTensor([1.0, 2.0], "s"))

    cloud = to_point_clouds(series)

    assert cloud.dtype is expected_dtype
    assert np.array_equal(np.asarray(cloud), [[10.0], [20.0]])


@pytest.mark.parametrize("values_dtype, expected_dtype", [
    pytest.param(np.float32, sb.pcloud32, id="float32"),
    pytest.param(np.float64, sb.pcloud64, id="float64"),
])
def test_point_clouds_of_windows_have_the_precision_of_their_series(values_dtype, expected_dtype):
    series = TimeSeries(np.array([0.0, 10.0, 20.0], dtype=values_dtype), times=TimeDeltaTensor([0.0, 1.0, 2.0], "s"))
    windows = cut_windows(series, size=TimeDelta(1.0, "s"), step=TimeDelta(1.0, "s"), closed="both",
                          partial=False, anchor="start")

    assert to_point_clouds(windows).dtype is expected_dtype


def _two_channels():
    a = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([1.0, 2.0, 3.0], "s"))
    b = TimeSeries([-10.0, -20.0, -30.0], times=TimeDeltaTensor([1.0, 2.0, 3.0], "s"))
    return TimeSeries.from_channels([a, b], unit="s")


@pytest.mark.parametrize("make_series", [
    pytest.param(_two_channels, id="series"),
    pytest.param(lambda: cut_windows(_two_channels(), size=TimeDelta(1.0, "s"), step=TimeDelta(1.0, "s"),
                                     closed="both", partial=False, anchor="start"), id="windows"),
])
def test_point_clouds_of_several_channels_raise_naming_stack_channels(make_series):
    series = make_series()

    with pytest.raises(ValueError, match="stack_channels"):
        to_point_clouds(series)


@pytest.mark.parametrize("make_series", [
    pytest.param(lambda: TimeSeries([True, False], times=TimeDeltaTensor([1.0, 2.0], "s")), id="booleans"),
    pytest.param(lambda: TimeSeries([[[10.0, 20.0],
                                      [30.0, 40.0]],

                                     [[20.0, 40.0],
                                      [60.0, 80.0]]], times=TimeDeltaTensor([1.0, 2.0], "s")),
                 id="matrices"),
    pytest.param(lambda: TimeSeries(sb.zeros((2,), dtype=sb.pcf64), times=TimeDeltaTensor([1.0, 2.0], "s")),
                 id="pcfs"),
])
def test_point_clouds_of_unsupported_observations_raise(make_series):
    series = make_series()

    with pytest.raises(TypeError):
        to_point_clouds(series)


def test_point_clouds_of_an_array_raise():
    with pytest.raises(TypeError, match="TimeSeries"):
        to_point_clouds(np.array([10.0, 20.0]))  # pyright: ignore[reportCallIssue, reportArgumentType]
