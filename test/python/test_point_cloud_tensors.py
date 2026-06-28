import numpy as np
import pytest

import stablebear as sb


# ---------------------------------------------------------------------------
# Bug #44: a fresh zeros(dtype=pcloud*) cell read back as a 0-d scalar 0.0
# (the C++ default-constructed Tensor) instead of the documented "empty point
# cloud". The sibling matrix/barcode dtypes already normalize their fresh cells,
# so a never-assigned point-cloud cell must read back as an empty (0, 2) cloud.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype, np_float", [(sb.pcloud64, np.float64),
                                             (sb.pcloud32, np.float32)])
def test_fresh_pcloud_cell_is_empty_cloud(dtype, np_float):
    t = sb.zeros((2,), dtype=dtype)
    fresh = t[0]
    assert fresh.shape == (0, 2)
    assert np.asarray(fresh).dtype == np_float
    # equals an explicitly-assigned empty cloud
    t[1] = np.zeros((0, 2), dtype=np_float)
    assert fresh.array_equal(t[1])


def test_fresh_pcloud_cell_does_not_mask_real_cloud():
    t = sb.zeros((2,), dtype=sb.pcloud64)
    t[0] = np.arange(6.0).reshape(3, 2)
    assert t[0].shape == (3, 2)
    assert t[1].shape == (0, 2)   # the still-unassigned cell


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

    Y[0, 0] = np.random.randn(30, 2, 20)
    Y[1, 1] = np.random.randn(40, 15, 10)

    assert Y[0, 0].shape == (30, 2, 20)
    assert Y[1, 1].shape == (40, 15, 10)


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

    assert isinstance(T[0], sb.FloatTensor)
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
