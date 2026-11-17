import numpy as np
import pytest

import stablebear as sb


@pytest.mark.parametrize("dtype", [sb.pcf32, sb.pcf64])
def test_sum_matches_pcf_addition(dtype):
    f1 = sb.Pcf([
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=dtype)
    f2 = sb.Pcf([
        [0.0, 2.0],
        [2.0, 0.0],
    ], dtype=dtype)
    f3 = sb.Pcf([
        [0.0, 3.0],
        [3.0, 0.0],
    ], dtype=dtype)
    functions = sb.tensor([f1, f2, f3])

    total = sb.sum(functions)

    assert total[0] == f1 + f2 + f3


def test_mean_of_1d_returns_tensor():
    A = sb.zeros((10,))
    avg = sb.mean(A)

    assert isinstance(avg, sb.PcfTensor)
    assert avg.shape == (1,)
    assert np.array_equal(avg[0].to_numpy(), np.zeros((1, 2)))


def test_mean_of_simple():
    A = sb.zeros((2,))

    A[0] = sb.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))

    avg = sb.mean(A)
    avg_data = avg[0].to_numpy()

    assert avg_data.shape == (2, 2)
    assert avg_data[0][0] == pytest.approx(0.0, 1e-4)
    assert avg_data[0][1] == pytest.approx(0.5, 1e-4)
    assert avg_data[1][0] == pytest.approx(2.0, 1e-4)
    assert avg_data[1][1] == pytest.approx(1.5, 1e-4)


def test_mean_f64():
    A = sb.zeros((2,), dtype=sb.pcf64)
    A[0] = sb.Pcf(np.array([[0.0, 4.0], [2.0, 0.0]], dtype=np.float64))
    A[1] = sb.Pcf(np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float64))

    avg = sb.mean(A)
    assert isinstance(avg, sb.PcfTensor)
    avg_data = avg[0].to_numpy()
    assert avg_data[0][1] == pytest.approx(3.0, abs=1e-7)


def test_mean_2d_dim0():
    X = sb.zeros((2, 3))
    X[0, 0] = sb.Pcf(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32))
    X[1, 0] = sb.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]], dtype=np.float32))

    avg = sb.mean(X, dim=0)
    assert avg.shape == (3,)
    # Column 0: mean of 2 and 4 on [0,1) -> 3
    assert avg[0](0.5) == pytest.approx(3.0, abs=1e-5)


def test_mean_2d_dim1():
    X = sb.zeros((2, 3))
    X[0, 0] = sb.Pcf(np.array([[0.0, 6.0], [1.0, 0.0]], dtype=np.float32))
    X[0, 1] = sb.Pcf(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32))
    X[0, 2] = sb.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]], dtype=np.float32))

    avg = sb.mean(X, dim=1)
    assert avg.shape == (2,)
    # Row 0: mean of 6, 2, 4 on [0,1) -> 4
    assert avg[0](0.5) == pytest.approx(4.0, abs=1e-5)


def test_resolve_pcf_inputs_invalid():
    from stablebear.base_tensor import _resolve_pcf_inputs
    from stablebear.reductions import _REDUCTIONS_BACKEND_MAP

    with pytest.raises(ValueError, match="not supported"):
        _resolve_pcf_inputs(_REDUCTIONS_BACKEND_MAP, "not a tensor")  # pyright: ignore[reportArgumentType]


def test_max_time_1d():
    A = sb.zeros((2,))
    A[0] = sb.Pcf(np.array([[0.0, 1.0], [5.0, 0.0]], dtype=np.float32))
    A[1] = sb.Pcf(np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32))

    result = sb.max_time(A)
    assert isinstance(result, sb.FloatTensor)
    assert float(result) == pytest.approx(5.0, abs=1e-5)


def test_max_time_f64():
    A = sb.zeros((2,), dtype=sb.pcf64)
    A[0] = sb.Pcf(np.array([[0.0, 1.0], [7.0, 0.0]], dtype=np.float64))
    A[1] = sb.Pcf(np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float64))

    result = sb.max_time(A)
    assert isinstance(result, sb.FloatTensor)
    assert float(result) == pytest.approx(7.0, abs=1e-10)


def test_max_time_2d():
    X = sb.zeros((2, 3))
    X[0, 0] = sb.Pcf(np.array([[0.0, 1.0], [10.0, 0.0]], dtype=np.float32))
    X[0, 1] = sb.Pcf(np.array([[0.0, 1.0], [6.0, 0.0]], dtype=np.float32))
    X[0, 2] = sb.Pcf(np.array([[0.0, 1.0], [4.0, 0.0]], dtype=np.float32))
    X[1, 0] = sb.Pcf(np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32))
    X[1, 1] = sb.Pcf(np.array([[0.0, 1.0], [8.0, 0.0]], dtype=np.float32))
    X[1, 2] = sb.Pcf(np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32))

    result = sb.max_time(X, dim=0)
    assert result.shape == (3,)
    result_np = np.asarray(result)
    assert result_np[0] == pytest.approx(10.0, abs=1e-5)
    assert result_np[1] == pytest.approx(8.0, abs=1e-5)
    assert result_np[2] == pytest.approx(4.0, abs=1e-5)

    result1 = sb.max_time(X, dim=1)
    assert result1.shape == (2,)
    result1_np = np.asarray(result1)
    assert result1_np[0] == pytest.approx(10.0, abs=1e-5)
    assert result1_np[1] == pytest.approx(8.0, abs=1e-5)


def test_max_time_with_default_pcf():
    """Default-constructed PCFs (single breakpoint at t=0) should not crash."""
    X = sb.zeros((3,))
    result = sb.max_time(X)
    assert float(result) == pytest.approx(0.0)


def test_max_time_mixed_default_and_nondefault():
    """max_time should reflect the non-zero PCFs, not be dragged down by defaults."""
    X = sb.zeros((3,))
    X[1] = sb.Pcf(np.array([[0.0, 1.0], [5.0, 0.0]], dtype=np.float32))

    result = sb.max_time(X)
    assert float(result) == pytest.approx(5.0, abs=1e-5)


def test_float_on_single_element_float_tensor():
    t = sb.FloatTensor(np.array([3.14]))
    assert float(t) == pytest.approx(3.14)


def test_int_on_single_element_float_tensor():
    t = sb.FloatTensor(np.array([7.9]))
    assert int(t) == 7


def test_float_rejects_multi_element_tensor():
    t = sb.FloatTensor(np.array([1.0, 2.0]))
    with pytest.raises(TypeError, match="single-element"):
        float(t)


def _points():
    # The distances are 5 between the first two points, sqrt(2) from the
    # first to the third, and sqrt(13) from the second to the third.
    return np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])


@pytest.mark.parametrize("dtype", [sb.float32, sb.float64])
def test_max_of_distance_matrix_gives_the_largest_distance_from_each_point(dtype):
    D = sb.DistanceMatrix(np.array([[0.0, 5.0, 1.0],
                                    [5.0, 0.0, 3.0],
                                    [1.0, 3.0, 0.0]]), dtype=dtype)

    out = sb.max(D)

    assert isinstance(out, sb.FloatTensor)
    assert out.dtype == dtype
    np.testing.assert_array_equal(np.asarray(out), [5.0, 5.0, 3.0])


def test_max_of_max_of_distance_matrix_is_its_largest_distance():
    D = sb.pdist(sb.PointCloud(_points()))

    out = sb.max(sb.max(D))

    assert out.shape == (1,)
    assert float(out) == 5.0


def test_max_of_distance_matrix_of_one_point_is_zero():
    out = sb.max(sb.DistanceMatrix(np.zeros((1, 1))))

    np.testing.assert_array_equal(np.asarray(out), [0.0])


@pytest.mark.parametrize("dim, expected", [(0, [4.0, 5.0, 6.0]), (1, [5.0, 6.0]), (-1, [5.0, 6.0])])
def test_max_of_float_tensor_reduces_along_dim(dim, expected):
    out = sb.max(sb.FloatTensor(np.array([[1.0, 5.0, 3.0],
                                          [4.0, 2.0, 6.0]])), dim=dim)

    assert isinstance(out, sb.FloatTensor)
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_max_of_int_tensor_is_an_int_tensor():
    out = sb.max(sb.IntTensor(np.array([[1, 5], [7, 2]])), dim=0)

    assert isinstance(out, sb.IntTensor)
    np.testing.assert_array_equal(np.asarray(out), [7, 5])


def test_max_of_framework_array_matches_numpy(fw):
    values = np.array([[1.0, 5.0, 3.0],
                       [4.0, 2.0, 6.0]])

    out = sb.max(fw.tensor(values), dim=1)

    assert isinstance(out, fw.tensor_type)
    np.testing.assert_array_equal(fw.to_numpy(out), [5.0, 6.0])
    assert tuple(sb.max(out, dim=0).shape) == (1,)


def test_gradient_of_largest_distance_of_points(fw):
    def loss(P):
        return fw.sum(sb.max(sb.max(sb.pdist(sb.LpDistance(2)(P)))))

    value, grad = fw.value_and_grad(loss, _points())

    # The largest distance is that between the first two points, which
    # each move it by the unit vector between them.
    assert value == 5.0
    np.testing.assert_allclose(grad, [[-0.6, -0.8], [0.6, 0.8], [0.0, 0.0]])


def test_max_along_missing_dim_raises():
    with pytest.raises(IndexError, match="out of range"):
        sb.max(sb.FloatTensor(np.ones((2, 3))), dim=2)


def test_max_of_empty_dim_raises():
    with pytest.raises(ValueError, match="empty"):
        sb.max(sb.FloatTensor(np.ones((0, 3))), dim=0)


def test_max_of_pcf_tensor_raises():
    F = sb.PcfTensor([sb.Pcf([[0.0, 1.0], [1.0, 0.0]])])
    with pytest.raises(TypeError, match="DistanceMatrix"):
        sb.max(F)  # pyright: ignore[reportCallIssue, reportArgumentType]


def _clouds():
    return [_points(), 2.0 * _points()]


def _largest_from_each_point():
    # Of _clouds: the largest distance from each point of each cloud.
    return np.array([[5.0, 5.0, np.sqrt(13.0)],
                     [10.0, 10.0, 2.0 * np.sqrt(13.0)]])


@pytest.mark.parametrize("dim", [-1, -2, 1, 2])
def test_max_of_distance_matrix_tensor_gives_the_largest_distance_from_each_point(dim):
    D = sb.pdist(sb.PointCloudTensor(_clouds()))

    out = sb.max(D, dim=dim)

    assert isinstance(out, sb.FloatTensor)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(out), _largest_from_each_point())


def test_max_of_float32_distance_matrix_tensor_is_float32():
    D = sb.DistanceMatrixTensor(np.zeros((2, 3, 3), dtype=np.float32))

    assert sb.max(D, dim=-1).dtype == sb.float32


def test_max_of_tracked_distance_matrix_tensor_matches_untracked(fw):
    D = sb.pdist(sb.PointCloudTensor([fw.tensor(cloud) for cloud in _clouds()]))

    out = sb.max(D, dim=-1)

    assert isinstance(out, fw.tensor_type)
    assert tuple(out.shape) == (2, 3)
    np.testing.assert_allclose(fw.to_numpy(out), _largest_from_each_point())


def test_gradient_of_largest_distances_of_batch_of_points(fw):
    def loss(P):
        D = sb.pdist(sb.PointCloudTensor([P, 2.0 * P]))
        return fw.sum(sb.max(sb.max(D, dim=-1), dim=-1))

    value, grad = fw.value_and_grad(loss, _points())

    # The largest distances, 5 and 10, are those between the first two
    # points, which move them by 1 + 2 times the unit vector between them.
    assert value == 15.0
    np.testing.assert_allclose(grad, [[-1.8, -2.4], [1.8, 2.4], [0.0, 0.0]])


def test_max_of_distance_matrix_tensor_along_its_shape_raises():
    D = sb.pdist(sb.PointCloudTensor(_clouds()))
    with pytest.raises(NotImplementedError, match="dim=-1"):
        sb.max(D, dim=0)


def test_max_of_distance_matrix_tensor_of_matrices_of_different_sizes_raises():
    D = sb.pdist(sb.PointCloudTensor([_points(), _points()[:2]]))
    with pytest.raises(ValueError, match="one size"):
        sb.max(D, dim=-1)
