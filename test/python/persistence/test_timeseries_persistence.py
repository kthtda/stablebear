import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import (
    Barcode,
    BarcodeTensor,
    barcode_to_accumulated_persistence,
    barcode_to_betti_curve,
    barcode_to_stable_rank,
    compute_persistent_homology,
)
from stablebear.timeseries import Times, TimeSeries


def _clouds(dtype, np_dtype):
    # Different point counts and barcodes; the square has one degree-1 bar.
    return sb.PointCloudTensor([
        np.array([[0, 0], [1, 0]], dtype=np_dtype),
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np_dtype),
        np.array([[0, 0], [2, 0], [5, 0]], dtype=np_dtype),
    ], dtype=dtype)


@pytest.mark.parametrize("dtype, np_dtype, barcode_dtype", [
    (sb.pcloud32, np.float32, sb.barcode32),
    (sb.pcloud64, np.float64, sb.barcode64),
])
def test_series_persistence_matches_each_snapshot(dtype, np_dtype, barcode_dtype):
    clouds = _clouds(dtype, np_dtype)
    times = Times([0.0, 1.3, 4.0], unit="ms")
    series = TimeSeries(clouds, times=times)

    result = compute_persistent_homology(series, max_dim=1)

    assert isinstance(result, TimeSeries)
    assert result.n_channels == 1
    assert result.dtype is barcode_dtype
    assert result.shape == (2,)
    assert result.times == times
    assert result.unit == "ms"
    for i in range(3):
        expected = compute_persistent_homology(clouds[i], max_dim=1)
        assert result.values[i].is_isomorphic_to(expected, atol=0, rtol=0)


def test_series_of_one_cloud_keeps_its_time_axis():
    series = TimeSeries(_clouds(sb.pcloud64, np.float64)[:1], times=[0.0])

    result = compute_persistent_homology(series, max_dim=1)

    assert len(result) == 1
    assert result.shape == (2,)


def test_each_channel_gets_its_own_barcodes_and_times():
    clouds = _clouds(sb.pcloud64, np.float64)
    cam = TimeSeries(clouds, times=Times([0.0, 1.3, 4.0], unit="ms"))
    # Not cam's first two clouds, so reading cam's elements for lidar shows.
    lidar = TimeSeries(clouds[[2, 1]], times=Times([0.5, 2.0], unit="ms"))
    series = TimeSeries.from_channels([cam, lidar])

    result = compute_persistent_homology(series, max_dim=1)

    assert result.n_channels == 2
    for i, source in enumerate((cam, lidar)):
        channel = result[i]
        assert channel.times == source.times
        expected = compute_persistent_homology(source, max_dim=1)
        assert channel.values.is_isomorphic_to(expected.values, atol=0, rtol=0)


@pytest.mark.parametrize("dtype, np_dtype, barcode_dtype", [
    (sb.distmat32, np.float32, sb.barcode32),
    (sb.distmat64, np.float64, sb.barcode64),
])
def test_series_of_distance_matrices(dtype, np_dtype, barcode_dtype):
    matrices = sb.zeros((2,), dtype=dtype)
    matrices[0] = sb.DistanceMatrix(np.array([
        [0, 1],
        [1, 0],
    ], dtype=np_dtype))
    matrices[1] = sb.DistanceMatrix(np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0],
    ], dtype=np_dtype))
    times = Times([0.0, 2.5], unit="s")
    series = TimeSeries(matrices, times=times)

    result = compute_persistent_homology(series, max_dim=1)

    assert result.dtype is barcode_dtype
    assert result.times == times
    for i in range(2):
        expected = compute_persistent_homology(matrices[i], max_dim=1)
        assert result.values[i].is_isomorphic_to(expected, atol=0, rtol=0)


def test_empty_series_gives_a_typed_empty_result():
    series = TimeSeries(sb.zeros((0,), dtype=sb.pcloud32), times=[])

    result = compute_persistent_homology(series, max_dim=1)

    assert result.dtype is sb.barcode32
    assert len(result) == 0
    assert result.shape == (2,)
    assert result.unit == "s"


def test_keyword_input_and_parameters_reach_the_series_path():
    clouds = _clouds(sb.pcloud64, np.float64)
    series = TimeSeries(clouds, times=[0.0, 1.0, 2.0])

    result = compute_persistent_homology(X=series, max_dim=0, reduced=True)

    for i in range(3):
        expected = compute_persistent_homology(clouds[i], max_dim=0, reduced=True)
        assert result.values[i].is_isomorphic_to(expected, atol=0, rtol=0)


@pytest.mark.parametrize("np_dtype, barcode_dtype", [
    (np.float32, sb.barcode32),
    (np.float64, sb.barcode64),
])
def test_float_series_of_matrices_reads_rows_as_points(np_dtype, barcode_dtype):
    # Channel 0: two clouds of 2 points in 2-D; channel 1: one of 3 points in 1-D.
    flat = np.array([[[0, 0], [1, 0]], [[0, 0], [3, 0]]], dtype=np_dtype)
    line = np.array([[[0], [2], [5]]], dtype=np_dtype)
    numeric = TimeSeries.from_channels([
        TimeSeries(flat, times=[0.0, 1.0]),
        TimeSeries(line, times=[0.5]),
    ])
    clouds = TimeSeries.from_channels([
        TimeSeries(sb.PointCloudTensor(flat), times=[0.0, 1.0]),
        TimeSeries(sb.PointCloudTensor(line), times=[0.5]),
    ])

    result = compute_persistent_homology(numeric, max_dim=1)
    expected = compute_persistent_homology(clouds, max_dim=1)

    assert result.dtype is barcode_dtype
    assert result.shape == ((2,), (2,))
    for i in range(2):
        assert result[i].times == expected[i].times
        assert result[i].values.is_isomorphic_to(expected[i].values, atol=0, rtol=0)


@pytest.mark.parametrize("make_series", [
    pytest.param(lambda: TimeSeries([1.0, 2.0], times=[0.0, 1.0]), id="float-scalar"),
    pytest.param(lambda: TimeSeries(sb.zeros((2,), dtype=sb.barcode64), times=[0.0, 1.0]),
                 id="barcode"),
])
def test_series_that_holds_no_point_clouds_raises(make_series):
    with pytest.raises(TypeError):
        compute_persistent_homology(make_series(), max_dim=1)


def _barcode_series():
    # Two channels with different times, each with two distinct barcodes.
    x = TimeSeries(BarcodeTensor([
        Barcode(np.array([[0.0, 1.0]])),
        Barcode(np.array([[0.0, 2.0], [0.5, 1.5]])),
    ]), times=Times([0.0, 1.0], unit="s"))
    y = TimeSeries(BarcodeTensor([
        Barcode(np.array([[0.0, 4.0], [1.0, 2.0], [1.0, 3.0]])),
        Barcode(np.array([[0.25, 0.75]])),
    ]), times=Times([500.0, 1500.0], unit="ms"))
    return TimeSeries.from_channels([x, y], unit="s")


@pytest.mark.parametrize("summary", [
    barcode_to_stable_rank,
    barcode_to_betti_curve,
    barcode_to_accumulated_persistence,
])
def test_summary_of_a_series_keeps_channels_and_times(summary):
    series = _barcode_series()

    result = summary(series)

    assert isinstance(result, TimeSeries)
    assert result.n_channels == 2
    for i in range(2):
        assert result[i].times == series[i].times
        assert result[i].values.array_equal(summary(series[i].values))


def test_accumulated_persistence_of_a_series_forwards_max_death():
    series = _barcode_series()

    # Both channels have bars dying after 1.25, which max_death removes.
    result = barcode_to_accumulated_persistence(series, max_death=1.25)

    for i in range(2):
        expected = barcode_to_accumulated_persistence(series[i].values, max_death=1.25)
        assert result[i].values.array_equal(expected)
        assert not result[i].values.array_equal(
            barcode_to_accumulated_persistence(series[i].values)
        )


def test_stable_rank_of_a_point_cloud_series_raises():
    series = TimeSeries(_clouds(sb.pcloud64, np.float64), times=[0.0, 1.0, 2.0])

    with pytest.raises(TypeError):
        barcode_to_stable_rank(series)


def test_series_persistence_does_not_depend_on_the_thread_count():
    series = TimeSeries(_clouds(sb.pcloud64, np.float64), times=[0.0, 1.0, 2.0])
    expected = compute_persistent_homology(series, max_dim=1)

    previous = sb.system.get_cpu_limit()
    sb.system.limit_cpus(1)
    try:
        assert sb.system.get_cpu_limit() == 1
        result = compute_persistent_homology(series, max_dim=1)
    finally:
        sb.system.limit_cpus(previous)

    assert sb.system.get_cpu_limit() == previous
    assert result.values.is_isomorphic_to(expected.values, atol=0, rtol=0)
