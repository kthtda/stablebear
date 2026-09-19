import copy

import numpy as np
import numpy.testing as npt

import stablebear as sb


def _leaf(values):
    return sb.tensor(np.asarray(values, dtype=np.int64))


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
