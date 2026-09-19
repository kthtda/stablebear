import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


@pytest.mark.parametrize(
    ("dtype", "np_dtype"),
    [(sb.float32, np.float32), (sb.float64, np.float64)],
)
def test_point_cloud_is_a_rank_two_facade(dtype, np_dtype):
    expected = np.arange(12, dtype=np_dtype).reshape(4, 3)
    cloud = sb.PointCloud(expected, dtype=dtype)

    assert cloud.shape == (4, 3)
    assert cloud.size == 12
    assert cloud.dtype == dtype
    assert not hasattr(cloud, "ndim")
    assert not hasattr(cloud, "squeeze")
    assert not hasattr(cloud, "reshape")
    assert cloud[:, 1].array_equal(expected[:, 1])

    cloud[2, 1] = -1
    expected[2, 1] = -1
    np.testing.assert_array_equal(np.asarray(cloud), expected)


@pytest.mark.parametrize("shape", [(3,), (2, 3, 4)])
def test_point_cloud_rejects_non_rank_two_data(shape):
    with pytest.raises(ValueError, match="2 dimensions"):
        sb.PointCloud(np.zeros(shape))


@pytest.mark.parametrize("shape", [(3,), (2, 3, 4)])
def test_point_cloud_tensor_rejects_non_rank_two_assignment(shape):
    tensor = sb.zeros((1,), dtype=sb.pcloud64)
    with pytest.raises(ValueError, match="2 dimensions"):
        tensor[0] = np.zeros(shape)


def test_can_create_point_clouds():
    X = sb.zeros((2,), dtype=sb.pcloud64)

    assert isinstance(X, sb.PointCloudTensor)
    assert X.dtype == sb.pcloud64

    X[0] = np.random.randn(10, 2)
    X[1] = np.random.randn(20, 2)

    assert X[0].shape == (10, 2)
    assert X[1].shape == (20, 2)

    Y = sb.zeros((2, 3), dtype=sb.pcloud32)

    assert isinstance(Y, sb.PointCloudTensor)
    assert Y.dtype == sb.pcloud32

    Y[0, 0] = np.random.randn(30, 20)
    Y[1, 1] = np.random.randn(40, 10)

    assert Y[0, 0].shape == (30, 20)
    assert Y[1, 1].shape == (40, 10)

def test_single_cloud_is_subscriptable():
    # A 0-d PointCloudTensor wraps a single cloud; it should be indexable as
    # its (n_points, dim) array so the natural pc[:, 0] / pc[:, 1] plotting
    # idiom works directly (see issue #133).
    arr = np.random.RandomState(0).rand(6, 2)
    pc = sb.PointCloudTensor(arr)
    assert pc.ndim == 0

    assert pc[:, 0].array_equal(arr[:, 0])
    assert pc[:, 1].array_equal(arr[:, 1])
    assert pc[0].array_equal(arr[0])
    assert pc[1:3].array_equal(arr[1:3])

    # Whole-cloud element access is unchanged.
    assert pc[()].array_equal(arr)
    assert pc[...].array_equal(arr)


def test_tensor_of_clouds_indexing_unchanged():
    # Rank >= 1 tensors still index over clouds, not into them.
    arr = np.random.RandomState(1).rand(5, 2)
    T = sb.zeros((3,), dtype=sb.pcloud64)
    T[0] = arr

    assert isinstance(T[0], sb.PointCloud)
    assert T[0].shape == (5, 2)
    assert T[0][:, 1].array_equal(arr[:, 1])

    sub = T[1:]
    assert isinstance(sub, sb.PointCloudTensor)
    assert sub.shape == (2,)

    # Selecting one cloud from a higher-rank tensor, then column-indexing it.
    grid = sb.zeros((2, 3), dtype=sb.pcloud64)
    grid[0, 1] = arr
    assert grid[0, 1][:, 0].array_equal(arr[:, 0])
    assert grid[0, 1][3].array_equal(arr[3])

    cloud = grid[0, 1]
    cloud[3, 0] = -1
    assert grid[0, 1][3, 0] == -1

    copied = cloud.copy()
    copied[3, 0] = 12
    assert grid[0, 1][3, 0] == -1


def test_stored_is_same_as_numpy():
    shape = (10, 20, 30)
    pclouds = sb.zeros(shape, dtype=sb.pcloud64)
    X = np.random.randn(10, 2).astype(np.float64)

    pclouds[0, 1, 2] = X
    assert pclouds[0, 1, 2].array_equal(X)

    pclouds = sb.zeros(shape, dtype=sb.pcloud32)
    X = np.random.randn(10, 2).astype(np.float32)

    pclouds[0, 1, 2] = X
    assert pclouds[0, 1, 2].array_equal(X)


@pytest.mark.parametrize(
    ("dtype", "np_dtype"),
    [(sb.pcloud32, np.float32), (sb.pcloud64, np.float64)],
)
def test_point_selection_owns_indices_but_retains_source_view(dtype, np_dtype):
    coordinates = np.asarray([[10], [20], [30]], dtype=np_dtype)
    points = sb.PointCloudTensor([coordinates], dtype=dtype)
    selections = sb.NestedTensor(
        [sb.tensor([2, 0], dtype=sb.uint64)]
    )

    selected = points[selections]
    sibling = selected[...]
    expected = coordinates[[2, 0]].copy()

    selections[0][0] = 1
    assert selections[0][0] == 1
    npt.assert_array_equal(np.asarray(selected[0]), expected)

    selections[0] = sb.tensor([1], dtype=sb.uint64)
    npt.assert_array_equal(np.asarray(selections[0]), [1])
    npt.assert_array_equal(np.asarray(selected[0]), expected)
    npt.assert_array_equal(np.asarray(sibling[0]), expected)

    del selections

    npt.assert_array_equal(np.asarray(selected[0]), expected)
    npt.assert_array_equal(np.asarray(sibling[0]), expected)

    points[0][2, 0] = 99
    expected[0, 0] = 99
    npt.assert_array_equal(np.asarray(selected[0]), expected)
    npt.assert_array_equal(np.asarray(sibling[0]), expected)
