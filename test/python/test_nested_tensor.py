import copy

import pytest

import numpy as np
import numpy.testing as npt

import stablebear as sb


def _leaf(values):
    return sb.tensor(np.asarray(values, dtype=np.int64))


def test_depth_three_with_different_child_and_leaf_shapes():
    # depth-three-example-start
    # Depth 2 means an outer tensor containing numeric tensors: two tensor levels.
    # Shapes and numbers of axes can differ without changing the depth.
    pair = sb.NestedTensor([
        np.array([1, 2, 3], dtype=np.int64),               # shape (3,)
        np.array([[4, 5], [6, 7]], dtype=np.int64),       # shape (2, 2)
    ])                                                      # shape (2,)
    assert pair[0][2] == 3
    assert pair[1][1, 0] == 6

    grid = sb.NestedTensor([
        [
            np.array(42, dtype=np.int64),                   # shape ()
            np.arange(8, dtype=np.int64).reshape(2, 1, 4),  # shape (2, 1, 4)
            np.array([11, 12], dtype=np.int64),             # shape (2,)
        ],
        [
            np.array([[13, 14, 15]], dtype=np.int64),       # shape (1, 3)
            np.array([[16], [17]], dtype=np.int64),         # shape (2, 1)
            np.array([], dtype=np.int64),                  # shape (0,)
        ],
    ])                                                     # shape (2, 3)
    assert grid[0, 0][()] == 42
    assert grid[0, 1][1, 0, 3] == 7
    assert grid[1, 0][0, 2] == 15

    scalar = sb.NestedTensor(np.array([9, 10], dtype=np.int64))  # shape ()

    # No children to infer a type from: explicitly declare Tensor<Tensor<int64>>.
    # Only the outer shape (0,) exists; there are no numeric leaves or inner shapes.
    empty = sb.NestedTensor([], dtype=sb.int64, depth=2)

    # Infer depth 3 and dtype int64 from the depth-2 children.
    nested = sb.NestedTensor([pair, grid, scalar, empty])

    assert nested.shape == (4,)
    assert nested.ndim == 1
    assert nested.depth == 3
    assert nested[0].shape == (2,)
    assert nested[1].shape == (2, 3)
    assert nested[2].shape == ()
    assert nested[3].shape == (0,)

    assert nested[0][0].shape == (3,)
    assert nested[0][1].shape == (2, 2)
    assert nested[1][0, 0].shape == ()
    assert nested[1][0, 1].shape == (2, 1, 4)
    assert nested[1][0, 2].shape == (2,)
    assert nested[1][1, 0].shape == (1, 3)
    assert nested[1][1, 1].shape == (2, 1)
    assert nested[1][1, 2].shape == (0,)
    assert nested[2][()].shape == (2,)

    # Each bracket indexes one tensor level; commas index axes within a level.
    assert nested[0][1][1, 0] == 6
    assert nested[1][0, 0][()] == 42
    assert nested[1][0, 1][1, 0, 3] == 7
    assert nested[1][1, 0][0, 2] == 15
    assert nested[2][()][1] == 10
    # depth-three-example-end

    assert nested.dtype == sb.int64
    assert repr(nested) == "Tensor<Tensor<Tensor<int64>>>"
    for child in nested:
        assert child.depth == 2
        assert child.dtype == sb.int64
    npt.assert_array_equal(np.asarray(nested[0][0]), [1, 2, 3])
    npt.assert_array_equal(np.asarray(nested[0][1]), [[4, 5], [6, 7]])
    npt.assert_array_equal(np.asarray(nested[1][0, 0]), np.array(42))
    npt.assert_array_equal(
        np.asarray(nested[1][0, 1]), np.arange(8).reshape(2, 1, 4)
    )
    npt.assert_array_equal(np.asarray(nested[1][0, 2]), [11, 12])
    npt.assert_array_equal(np.asarray(nested[1][1, 0]), [[13, 14, 15]])
    npt.assert_array_equal(np.asarray(nested[1][1, 1]), [[16], [17]])
    npt.assert_array_equal(np.asarray(nested[1][1, 2]), np.array([], dtype=np.int64))
    npt.assert_array_equal(np.asarray(nested[2][()]), [9, 10])
    assert nested[3].size == 0
    with pytest.raises(IndexError):
        nested[3][0]

    # Construction isolates the supplied values at every nesting level.
    nested[0][1][1, 0] = 60
    assert nested[0][1][1, 0] == 60
    assert pair[1][1, 0] == 6
    grid[0, 1][1, 0, 3] = 70
    assert grid[0, 1][1, 0, 3] == 70
    assert nested[1][0, 1][1, 0, 3] == 7


def test_outer_slice_shares_leaf_mutations_and_child_replacement():
    nested = sb.NestedTensor([_leaf([1, 2]), _leaf([3, 4])])
    view = nested[:1]
    sibling = nested[...]

    view[0][0] = 99

    assert nested[0][0] == 99
    assert sibling[0][0] == 99

    replacement = _leaf([7, 8, 9])
    view[0] = replacement
    replacement[0] = -1

    npt.assert_array_equal(np.asarray(nested[0]), [7, 8, 9])
    npt.assert_array_equal(np.asarray(sibling[0]), [7, 8, 9])


def test_transposed_outer_view_shares_storage():
    nested = sb.NestedTensor(
        [
            [_leaf([1, 2]), _leaf([3, 4])],
            [_leaf([5, 6]), _leaf([7, 8])],
        ]
    )

    transposed = nested.transpose()
    transposed[0, 1][1] = 60

    assert nested[1, 0][1] == 60


def test_empty_outer_view_preserves_depth():
    nested = sb.NestedTensor([_leaf([1, 2]), _leaf([3, 4])])

    empty = nested[:0]

    assert empty.shape == (0,)
    assert empty.depth == nested.depth
    assert empty.dtype == nested.dtype


def test_value_construction_and_assignment_copy_caller_owned_children():
    child = _leaf([1, 2])
    nested = sb.NestedTensor([child, child])

    child[0] = 99
    assert nested[0][0] == 1
    assert nested[1][0] == 1

    nested[0][0] = 77
    assert nested[0][0] == 77
    assert nested[1][0] == 1

    replacement = _leaf([5, 6])
    nested[1] = replacement
    npt.assert_array_equal(np.asarray(nested[0]), [77, 2])
    npt.assert_array_equal(np.asarray(nested[1]), [5, 6])

    replacement[0] = 88
    assert replacement[0] == 88
    npt.assert_array_equal(np.asarray(nested[0]), [77, 2])
    npt.assert_array_equal(np.asarray(nested[1]), [5, 6])


def test_copy_and_deepcopy_are_recursively_independent():
    nested = sb.NestedTensor([_leaf([1, 2]), _leaf([3, 4])])
    copied = nested.copy()
    deepcopied = copy.deepcopy(nested)

    copied[0][0] = 10
    deepcopied[1][1] = 40

    npt.assert_array_equal(np.asarray(nested[0]), [1, 2])
    npt.assert_array_equal(np.asarray(nested[1]), [3, 4])
    npt.assert_array_equal(np.asarray(copied[0]), [10, 2])
    npt.assert_array_equal(np.asarray(copied[1]), [3, 4])
    npt.assert_array_equal(np.asarray(deepcopied[0]), [1, 2])
    npt.assert_array_equal(np.asarray(deepcopied[1]), [3, 40])


@pytest.mark.parametrize("dtype", [sb.float32, sb.float64, sb.int32, sb.int64, sb.uint32, sb.uint64])
def test_existing_nested_value_construction_is_recursively_independent(dtype):
    leaf = sb.tensor(np.array([1, 2]), dtype=dtype)
    original = sb.NestedTensor([sb.NestedTensor([leaf])])
    constructed = sb.NestedTensor(original)
    repeated = sb.NestedTensor([original, original])

    constructed[0][0][0] = 10
    repeated[0][0][0][1] = 20
    assert constructed[0][0][0] == 10
    assert repeated[0][0][0][1] == 20
    assert original[0][0][0] == 1
    assert original[0][0][1] == 2
    assert repeated[1][0][0][1] == 2

    repeated[1] = original
    original[0][0][1] = 30
    assert original[0][0][1] == 30
    assert repeated[1][0][0][1] == 2
    assert repeated[0][0][0][1] == 20


@pytest.mark.parametrize("operation", [
    lambda x: x.reshape((2, 2)).transpose(),
    lambda x: x.reshape((2, 2)).swapaxes(0, 1),
    lambda x: x.expand_dims(0).squeeze(),
    lambda x: x.flatten(),
    lambda x: x[::2],
])
def test_deep_outer_views_share_storage_and_copies_isolate(operation):
    nested = sb.NestedTensor([sb.NestedTensor([_leaf([i])]) for i in range(4)])
    view = operation(nested)
    index = (0,) * view.ndim
    copied = sb.NestedTensor(view)
    deepcopied = copy.deepcopy(view)
    view[index][0][0] = 99
    assert view[index][0][0] == 99
    assert nested[0][0][0] == 99
    assert copied[index][0][0] == 0
    assert deepcopied[index][0][0] == 0


def test_scalar_and_empty_value_construction_preserves_metadata():
    for original in [sb.NestedTensor(_leaf([1, 2])),
                     sb.NestedTensor([], dtype=sb.int64, depth=4)]:
        for copied in [sb.NestedTensor(original), original.copy(), copy.deepcopy(original)]:
            assert copied.shape == original.shape
            assert copied.depth == original.depth
            assert copied.dtype == original.dtype
    scalar = sb.NestedTensor(_leaf([1, 2]))
    copied = sb.NestedTensor(scalar)
    copied[()][0] = 99
    assert copied[()][0] == 99
    assert scalar[()][0] == 1


def test_internal_node_wrapping_preserves_leaf_buffer():
    nested = sb.NestedTensor([sb.NestedTensor([_leaf([1, 2])])])
    original_buffer = np.asarray(nested[0][0]._data)
    child_buffer = np.asarray(nested[:1][0][0]._data)
    copied_buffer = np.asarray(nested.copy()[0][0]._data)
    assert np.shares_memory(original_buffer, child_buffer)
    assert not np.shares_memory(original_buffer, copied_buffer)
