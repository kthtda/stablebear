"""Ordinary tensor operations on property-indexed point-cloud tensors."""

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


DTYPES = [
    pytest.param(sb.pcloud32, np.float32, id="pcloud32"),
    pytest.param(sb.pcloud64, np.float64, id="pcloud64"),
]


def indexed_points(dtype, np_dtype):
    clouds = [
        np.arange(10, dtype=np_dtype).reshape(5, 2) + 100 * i
        for i in range(4)
    ]
    source = sb.PointCloudTensor(clouds, dtype=dtype)
    selections = sb.NestedTensor([
        sb.tensor(rows, dtype=sb.uint64)
        for rows in ([4, 1], [0, 3], [2, 2], [1, 4])
    ])
    return source, source[selections]


@pytest.mark.parametrize(("dtype", "np_dtype"), DTYPES)
def test_comparison_and_mixed_joins(dtype, np_dtype):
    _, indexed = indexed_points(dtype, np_dtype)
    dense = indexed.to_dense()

    npt.assert_array_equal(np.asarray(indexed == dense), True)
    npt.assert_array_equal(np.asarray(dense == indexed), True)
    assert indexed.array_equal(dense)
    assert dense.array_equal(indexed)

    stacked = sb.stack([indexed, dense], axis=1)
    concatenated = sb.concatenate([dense, indexed], axis=0)
    assert stacked.shape == (4, 2)
    assert concatenated.shape == (8,)
    for i in range(4):
        npt.assert_array_equal(np.asarray(stacked[i, 0]), np.asarray(indexed[i]))
        npt.assert_array_equal(np.asarray(stacked[i, 1]), np.asarray(dense[i]))

    stacked[0, 0][0, 0] = -1
    assert indexed[0][0, 0] != -1
    with pytest.raises(ValueError):
        sb.stack([indexed, dense[:2]])


@pytest.mark.parametrize(("dtype", "np_dtype"), DTYPES)
def test_splits_remain_shared_indexed_views(dtype, np_dtype):
    source, indexed = indexed_points(dtype, np_dtype)
    sibling = indexed[...]
    source_0 = source[0][4, 0]
    source_1 = source[1][0, 0]
    source_2 = source[2][2, 0]
    source_3 = source[3][1, 0]

    parts = sb.split(indexed, [1, 3])
    uneven = sb.array_split(indexed, 3)
    assert parts[0].shape == (1,)
    assert parts[1].shape == (2,)
    assert parts[2].shape == (1,)
    assert uneven[0].shape == (2,)
    assert uneven[1].shape == (1,)
    assert uneven[2].shape == (1,)
    assert type(parts[0]._data) is type(indexed._data)
    assert type(uneven[0]._data) is type(indexed._data)

    # split at [1, 3] produces indexed[0:1], indexed[1:3], indexed[3:4].
    first_part_cloud = parts[0][0]
    middle_part_cloud = parts[1][0]
    last_part_cloud = parts[2][0]
    npt.assert_array_equal(np.asarray(first_part_cloud), np.asarray(indexed[0]))
    npt.assert_array_equal(np.asarray(middle_part_cloud), np.asarray(indexed[1]))
    npt.assert_array_equal(np.asarray(last_part_cloud), np.asarray(indexed[3]))

    first_part_cloud[0, 0] = -10
    assert indexed[0][0, 0] == -10
    assert sibling[0][0, 0] == -10

    middle_part_cloud[0, 0] = -11
    assert indexed[1][0, 0] == -11
    assert sibling[1][0, 0] == -11

    last_part_cloud[0, 0] = -12
    assert indexed[3][0, 0] == -12
    assert sibling[3][0, 0] == -12

    # Splitting four elements into three gives [0:2], [2:3], [3:4].
    uneven_middle_cloud = uneven[1][0]
    npt.assert_array_equal(np.asarray(uneven_middle_cloud), np.asarray(indexed[2]))
    uneven_middle_cloud[0, 0] = -13
    assert indexed[2][0, 0] == -13
    assert sibling[2][0, 0] == -13

    assert source[0][4, 0] == source_0
    assert source[1][0, 0] == source_1
    assert source[2][2, 0] == source_2
    assert source[3][1, 0] == source_3


@pytest.mark.parametrize(("dtype", "np_dtype"), DTYPES)
def test_mask_and_integer_selection(dtype, np_dtype):
    _, indexed = indexed_points(dtype, np_dtype)
    mask = sb.BoolTensor(np.array([True, False, True, False]))

    # The mask selects indexed[0] and indexed[2], in that order. The result is
    # dense and independent, so mutating selected[0] must not change indexed[0].
    selected = indexed[mask]
    assert selected.shape == (2,)
    npt.assert_array_equal(np.asarray(selected[0]), np.asarray(indexed[0]))
    npt.assert_array_equal(np.asarray(selected[1]), np.asarray(indexed[2]))
    selected[0][0, 0] = -20
    assert indexed[0][0, 0] != -20

    # The integer indices [3, 1] gather indexed[3] followed by indexed[1].
    gathered = indexed[np.array([3, 1])]
    npt.assert_array_equal(np.asarray(gathered[0]), np.asarray(indexed[3]))
    npt.assert_array_equal(np.asarray(gathered[1]), np.asarray(indexed[1]))


@pytest.mark.parametrize(("dtype", "np_dtype"), DTYPES)
@pytest.mark.parametrize(
    ("selector", "replaced"),
    [
        pytest.param(
            sb.BoolTensor(np.array([True, False, True, False])),
            (0, 2),
            id="bool-tensor",
        ),
        pytest.param(
            np.array([True, False, True, False]),
            (0, 2),
            id="bool-numpy",
        ),
        pytest.param(
            [True, False, True, False],
            (0, 2),
            id="bool-list",
        ),
        pytest.param(np.array([1, 3]), (1, 3), id="integer-numpy"),
        pytest.param([1, 3], (1, 3), id="integer-list"),
    ],
)
def test_mask_and_integer_assignment(
    dtype, np_dtype, selector, replaced
):
    source, indexed = indexed_points(dtype, np_dtype)
    sibling = indexed[:]
    source_before = [np.asarray(cloud).copy() for cloud in source]
    untouched = sorted(set(range(indexed.shape[0])) - set(replaced))
    untouched_before = {
        i: np.asarray(sibling[i]).copy() for i in untouched
    }

    replacement = np.asarray([[-1, -2]], dtype=np_dtype)
    indexed[selector] = replacement

    # Only the selected outer elements change. The sibling view observes the
    # writes, while the materialized source remains independent.
    for i in replaced:
        npt.assert_array_equal(np.asarray(sibling[i]), replacement)
    for i in untouched:
        npt.assert_array_equal(np.asarray(sibling[i]), untouched_before[i])

    # Materializing indexed state for writes must leave every source cloud
    # unchanged.
    for actual, expected in zip(source, source_before):
        npt.assert_array_equal(np.asarray(actual), expected)
