import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import compute_persistent_homology
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
