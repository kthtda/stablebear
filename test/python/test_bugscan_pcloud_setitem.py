import numpy as np
import pytest

import stablebear as sb


# ---------------------------------------------------------------------------
# Bug #40: assigning a NumPy array into a slice / Ellipsis / partial-int region
# of a PointCloudTensor stored the WHOLE array as a single cloud in every
# selected cell (via _decay_value + broadcast-fill). The array must instead be
# distributed one cloud per cell -- leading axes index cells, trailing axes
# (n_points, dim) form each cloud -- like PointCloudTensor(arr). The full-index
# path (t[i] = one_cloud) was already correct and must stay so.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype, np_dtype",
    [(sb.pcloud64, np.float64), (sb.pcloud32, np.float32)],
)
def test_slice_assign_distributes_clouds(dtype, np_dtype):
    arr = np.arange(2 * 4 * 2, dtype=np_dtype).reshape(2, 4, 2)
    t = sb.zeros((2,), dtype=dtype)
    t[:] = arr
    assert tuple(t[0].shape) == (4, 2)
    assert t[0].array_equal(arr[0])
    assert t[1].array_equal(arr[1])
    assert not t[0].array_equal(arr)          # not the whole array per cell


def test_ellipsis_assign_distributes_clouds_2d():
    arr = np.arange(3 * 5 * 4 * 2, dtype=np.float64).reshape(3, 5, 4, 2)
    t = sb.zeros((3, 5), dtype=sb.pcloud64)
    t[...] = arr
    assert tuple(t[0, 0].shape) == (4, 2)
    assert t[0, 0].array_equal(arr[0, 0])
    assert t[2, 4].array_equal(arr[2, 4])


def test_partial_int_assign_distributes_clouds():
    arr = np.arange(3 * 5 * 4 * 2, dtype=np.float64).reshape(3, 5, 4, 2)
    t = sb.zeros((3, 5), dtype=sb.pcloud64)
    t[0] = arr[0]                              # selects the (5,) row of clouds
    assert tuple(t[0, 0].shape) == (4, 2)
    assert t[0, 0].array_equal(arr[0, 0])
    assert t[0, 4].array_equal(arr[0, 4])


def test_full_int_assign_is_a_single_cloud():
    arr = np.arange(2 * 4 * 2, dtype=np.float64).reshape(2, 4, 2)
    t = sb.zeros((2,), dtype=sb.pcloud64)
    t[0] = arr[0]                             # full index -> one cloud
    assert tuple(t[0].shape) == (4, 2)
    assert t[0].array_equal(arr[0])


def test_mismatched_leading_shape_raises():
    t = sb.zeros((2,), dtype=sb.pcloud64)
    with pytest.raises(ValueError):
        t[:] = np.zeros((3, 4, 2))           # 3 clouds into a 2-cell region


def test_assigned_clouds_do_not_alias_source():
    arr = np.zeros((2, 3, 2), dtype=np.float64)
    t = sb.zeros((2,), dtype=sb.pcloud64)
    t[:] = arr
    arr[0, 0, 0] = 99.0                       # mutate the source array
    assert float(t[0][0, 0]) == 0.0          # cell is an independent copy
