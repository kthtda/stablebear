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
    assert cloud[:, 1].array_equal(expected[:, 1])

    cloud[2, 1] = -1
    expected[2, 1] = -1
    np.testing.assert_array_equal(np.asarray(cloud), expected)


@pytest.mark.parametrize("indexed", [False, True])
def test_coordinate_indexing_matches_numpy_views_and_copies(indexed, monkeypatch):
    source = sb.PointCloudTensor(np.arange(24.).reshape(1, 6, 4))
    points = (source[sb.NestedTensor([sb.indices([4, 1, 4])])]
              if indexed else source)
    backend_type = type(points._data._get_element([0]))

    def reject_source_copy(*args):
        pytest.fail("extracting a cloud must not copy its coordinate source")

    monkeypatch.setattr(backend_type, "to_float_tensor", reject_source_copy)
    cloud = points[0]
    expected = np.asarray(cloud).copy()
    source_before = np.asarray(source[0]).copy()
    view = cloud[::-1, None, 1:]
    expected_view = expected[::-1, None, 1:]
    npt.assert_array_equal(np.asarray(view), expected_view)
    assert points._data.has_indices() == indexed

    # Chained basic slices write through, even after the shared transition.
    row = view[0, 0]
    row[0] = 99
    expected_view[0, 0, 0] = 99
    row += 2
    expected_view[0, 0] += 2
    npt.assert_array_equal(np.asarray(cloud), expected)
    npt.assert_array_equal(np.asarray(view), expected_view)
    if indexed:
        npt.assert_array_equal(np.asarray(source[0]), source_before)

    column = cloud[:].T[0]
    column.reshape((1, -1))[0, 0] = 71
    expected[0, 0] = 71
    npt.assert_array_equal(np.asarray(cloud), expected)
    with pytest.raises(ValueError, match="read-only"):
        column.broadcast_to((2, len(column)))[0, 0] = 0

    # Multiple advanced indices use NumPy's paired indexing, not outer indexing.
    indices = ([0, 1], [1, 2])
    gathered = cloud[indices]
    npt.assert_array_equal(np.asarray(gathered), expected[indices])
    gathered[:] = -1
    masked = cloud[expected > 10]
    masked[:] = -2
    copied = cloud.copy()
    copied[0, 0] = -3
    npt.assert_array_equal(np.asarray(cloud), expected)

    array = np.asarray(cloud)
    array[0, 0] = 72
    assert cloud[0, 0] == 72


def test_rejected_coordinate_view_write_keeps_indexed_source_connection():
    source = sb.PointCloudTensor(np.arange(12.).reshape(1, 6, 2))
    points = source[sb.NestedTensor([sb.indices([4, 1, 4])])]
    row = points[0][0]
    with pytest.raises(ValueError):
        row[:] = [1, 2, 3]
    assert points._data.has_indices()
    source[0][4, 0] = 70
    assert row[0] == 70


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
        [sb.indices([2, 0])]
    )

    selected = points[selections]
    sibling = selected[...]
    expected = coordinates[[2, 0]].copy()

    selections[0][0] = 1
    assert selections[0][0] == 1
    npt.assert_array_equal(np.asarray(selected[0]), expected)

    selections[0] = sb.indices([1])
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


@pytest.mark.parametrize(
    ("dtype", "np_dtype"),
    [(sb.pcloud32, np.float32), (sb.pcloud64, np.float64)],
)
def test_point_selection_matches_leading_outer_dimensions(dtype, np_dtype):
    coordinates = np.arange(10, dtype=np_dtype).reshape(5, 2)
    scalar_points = sb.PointCloudTensor(coordinates, dtype=dtype)
    scalar_selections = sb.NestedTensor([
        sb.indices([3, 3, 1]),
        sb.indices([]),
        sb.indices([4]),
    ])

    scalar_selected = scalar_points[scalar_selections]

    assert scalar_selected.shape == (3,)
    assert scalar_selected.dtype == dtype
    npt.assert_array_equal(
        np.asarray(scalar_selected[0]), coordinates[[3, 3, 1]]
    )
    assert scalar_selected[1].shape == (0, 2)
    npt.assert_array_equal(np.asarray(scalar_selected[2]), coordinates[[4]])

    grid = np.arange(2 * 3 * 5 * 2, dtype=np_dtype).reshape(2, 3, 5, 2)
    grid_points = sb.PointCloudTensor(grid, dtype=dtype)
    grid_selections = sb.NestedTensor([
        sb.indices([1, 0]) for _ in range(2 * 3 * 2 * 2)
    ]).reshape((2, 3, 2, 2))

    grid_selected = grid_points[grid_selections]

    assert grid_selected.shape == (2, 3, 2, 2)
    npt.assert_array_equal(
        np.asarray(grid_selected[1, 2, 1, 0]), grid[1, 2, [1, 0]]
    )


def test_point_selection_preserves_scalar_and_empty_outer_shapes():
    coordinates = np.arange(5).reshape(5, 1)
    scalar_points = sb.PointCloudTensor(coordinates)
    scalar_selection = sb.NestedTensor(
        sb.indices([3, 0])
    )
    expected = sb.PointCloud([[3.0], [0.0]])

    scalar_selected = scalar_points[scalar_selection]

    assert scalar_selected.shape == ()
    assert scalar_selected[()].array_equal(expected)

    empty_points = sb.zeros((2, 0), dtype=sb.pcloud64)
    empty_selections = sb.NestedTensor(
        [], dtype=sb.uint64, depth=2
    ).reshape((2, 0, 4))

    empty_selected = empty_points[empty_selections]

    assert empty_selected.shape == (2, 0, 4)
    assert empty_selected.dtype == sb.pcloud64


def test_point_selection_rejects_nonmatching_outer_shapes():
    leaf = sb.indices([0])

    points = sb.PointCloudTensor(
        np.arange(2 * 2 * 3).reshape(2, 2, 3, 1)
    )
    too_few_axes = sb.NestedTensor([leaf, leaf])
    with pytest.raises(ValueError, match="leading dimensions.*exactly match"):
        points[too_few_axes]

    points = sb.PointCloudTensor(np.arange(2 * 3).reshape(2, 3, 1))
    wrong_leading_dimensions = sb.NestedTensor([leaf, leaf, leaf])
    with pytest.raises(ValueError, match="leading dimensions.*exactly match"):
        points[wrong_leading_dimensions]

    singleton_axis = sb.PointCloudTensor(
        np.arange(2 * 1 * 3).reshape(2, 1, 3, 1)
    )
    would_broadcast = sb.NestedTensor([leaf for _ in range(2 * 3)]).reshape(
        (2, 3)
    )
    with pytest.raises(ValueError, match="leading dimensions.*exactly match"):
        singleton_axis[would_broadcast]


def test_point_selection_validates_every_child_before_returning():
    points_array = np.arange(2 * 3).reshape(2, 3, 1)
    points = sb.PointCloudTensor(points_array)
    single_cloud = sb.PointCloudTensor(points_array[0])

    wrong_dtype = sb.NestedTensor(sb.tensor([0], dtype=sb.int64))
    with pytest.raises(TypeError, match="uint64 leaves"):
        single_cloud[wrong_dtype]

    too_deep = sb.NestedTensor([
        sb.NestedTensor(sb.indices([0]))
    ])
    with pytest.raises(ValueError, match="Tensor<Tensor<uint64>>"):
        single_cloud[too_deep]

    rank_zero = sb.NestedTensor(sb.indices(np.array(0, dtype=np.uint64)))
    with pytest.raises(ValueError, match="must have 1 dimension, got 0"):
        single_cloud[rank_zero]

    rank_two_late = sb.NestedTensor([
        sb.indices([0]),
        sb.indices([[0]]),
    ])
    with pytest.raises(ValueError, match="must have 1 dimension, got 2"):
        points[rank_two_late]

    out_of_bounds_late = sb.NestedTensor([
        sb.indices([0]),
        sb.indices([3]),
    ])
    with pytest.raises(IndexError, match="index 3.*cloud with 3 points"):
        points[out_of_bounds_late]

    maximum_index = sb.NestedTensor(
        sb.indices([np.iinfo(np.uint64).max])
    )
    with pytest.raises(IndexError, match="out of bounds"):
        single_cloud[maximum_index]

    empty_cloud = sb.PointCloudTensor(np.empty((0, 1)))
    first_point = sb.NestedTensor(sb.indices([0]))
    with pytest.raises(IndexError, match="cloud with 0 points"):
        empty_cloud[first_point]

    # Failed validation leaves both the source and its selections unchanged.
    npt.assert_array_equal(np.asarray(points[0]), points_array[0])
    npt.assert_array_equal(np.asarray(out_of_bounds_late[1]), [3])


def test_point_selection_accepts_outer_views_and_reindexes_indexed_results():
    coordinates = np.arange(4 * 4).reshape(4, 4, 1)
    points = sb.PointCloudTensor(coordinates)[::2]
    selections = sb.NestedTensor([
        sb.indices([3, 1, 3]),
        sb.indices([0]),
        sb.indices([2, 0]),
        sb.indices([1]),
    ])[::2]

    selected = points[selections]
    npt.assert_array_equal(np.asarray(selected[0]), coordinates[0, [3, 1, 3]])
    npt.assert_array_equal(np.asarray(selected[1]), coordinates[2, [2, 0]])

    second_selections = sb.NestedTensor([
        sb.indices([2, 0]),
        sb.indices([1]),
    ])
    selected_again = selected[second_selections]

    npt.assert_array_equal(np.asarray(selected_again[0]), coordinates[0, [3, 3]])
    npt.assert_array_equal(np.asarray(selected_again[1]), coordinates[2, [0]])


def test_point_selection_revalidates_after_source_replacement():
    coordinates = np.arange(3).reshape(3, 1)
    points = sb.PointCloudTensor([coordinates])
    selections = sb.NestedTensor([
        sb.indices([2])
    ])
    selected = points[selections]

    points[0] = coordinates[:1]

    with pytest.raises(IndexError, match="index 2.*cloud with 1 points"):
        selected[0]


@pytest.mark.parametrize(
    ("dtype", "np_dtype"),
    [(sb.pcloud32, np.float32), (sb.pcloud64, np.float64)],
)
def test_to_dense_returns_independent_ordinary_storage(dtype, np_dtype):
    coordinates = np.asarray([[10], [20], [30]], dtype=np_dtype)
    points = sb.PointCloudTensor([coordinates], dtype=dtype)
    selections = sb.NestedTensor([sb.indices([2, 0])])
    selected = points[selections]

    dense = selected.to_dense()

    assert type(dense._data) is type(points._data)
    npt.assert_array_equal(np.asarray(dense[0]), coordinates[[2, 0]])

    points[0][2, 0] = 99
    assert dense[0][0, 0] == 30

    dense[0][1, 0] = 88
    assert selected[0][1, 0] == 10

    dense_copy = dense.to_dense()
    dense_copy[0][0, 0] = 77
    assert dense[0][0, 0] == 30
