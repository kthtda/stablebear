import pickle

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


def indices(values):
    return np.asarray(values, dtype=np.uint64)


def assert_index_tensor(actual, expected, shape):
    assert actual.shape == shape
    assert actual.dtype == sb.uint64

    for index in np.ndindex(*shape):
        expected_selection = expected
        for i in index:
            expected_selection = expected_selection[i]

        actual_selection = actual[index]
        assert isinstance(actual_selection, sb.IntTensor)
        npt.assert_array_equal(
            np.asarray(actual_selection),
            indices(expected_selection),
        )


@pytest.mark.parametrize(
    ("values", "shape"),
    [
        pytest.param([3, 3, 7], (), id="scalar"),
        pytest.param([], (), id="empty-scalar"),
        pytest.param([[3, 3, 7], [1], []], (3,), id="ragged"),
        pytest.param([[], []], (2,), id="empty-selections"),
        pytest.param(
            [[[3, 2], [2, 6, 7]], [[4], [9, 6]]],
            (2, 2),
            id="multidimensional",
        ),
    ],
)
def test_index_tensor_from_lists(values, shape):
    assert_index_tensor(sb.IndexTensor(values), values, shape)


def test_index_tensor_slicing():
    tensor = sb.IndexTensor([[[3, 2], [2, 6, 7]], [[4], [9, 6]]])

    reversed_rows = tensor[::-1]
    assert isinstance(reversed_rows, sb.IndexTensor)
    assert_index_tensor(
        reversed_rows,
        [[[4], [9, 6]], [[3, 2], [2, 6, 7]]],
        (2, 2),
    )


def test_empty_outer_index_tensor():
    tensor = sb.IndexTensor([[], []])
    empty_outer = tensor[:0]
    assert_index_tensor(empty_outer, [], (0,))
    assert_index_tensor(empty_outer.copy(), [], (0,))


def test_index_tensor_copies_selections_on_construction_and_assignment():
    source = sb.IntTensor(indices([1, 2]))
    tensor = sb.IndexTensor([source])
    source[0] = 9
    npt.assert_array_equal(np.asarray(tensor[0]), indices([1, 2]))

    replacement = sb.IntTensor(indices([4, 5, 6]))
    tensor[0] = replacement
    replacement[0] = 9
    npt.assert_array_equal(np.asarray(tensor[0]), indices([4, 5, 6]))


def test_index_tensor_copy_and_pickle_are_independent():
    original = sb.IndexTensor([indices([1, 2]), indices([3])])
    copied = original.copy()
    restored = pickle.loads(pickle.dumps(original))

    original[0][0] = 9
    npt.assert_array_equal(np.asarray(copied[0]), indices([1, 2]))
    npt.assert_array_equal(np.asarray(restored[0]), indices([1, 2]))


@pytest.mark.parametrize(
    ("selection", "error", "message"),
    [
        pytest.param(np.array([[1, 2]], dtype=np.uint64), ValueError, "rank 1", id="rank"),
        pytest.param(np.array([1.5]), TypeError, "integers", id="dtype"),
        pytest.param(np.array([-1], dtype=np.int64), ValueError, "nonnegative", id="negative"),
    ],
)
def test_index_tensor_rejects_invalid_selections(selection, error, message):
    with pytest.raises(error, match=message):
        sb.IndexTensor(selection)


def test_index_tensor_rejects_boolean_lists():
    with pytest.raises(TypeError, match="flat integer sequences"):
        sb.IndexTensor([True, False])
