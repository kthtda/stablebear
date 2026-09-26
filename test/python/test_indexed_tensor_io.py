import io
import pickle

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb
import stablebear._sb_cpp as cpp


_DTYPES = [
    pytest.param(sb.pcloud32, np.float32, cpp._IndexedPointCloud32Tensor,
                 id="pcloud32"),
    pytest.param(sb.pcloud64, np.float64, cpp._IndexedPointCloud64Tensor,
                 id="pcloud64"),
]


def _assert_same_clouds(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    for index in np.ndindex(*actual.shape):
        npt.assert_array_equal(np.asarray(actual[index]), np.asarray(expected[index]))


def test_indexed_binary_and_pickle_roundtrip_retains_storage(make_pcloud_or_distmat_tensor):
    points = make_pcloud_or_distmat_tensor(np.arange(40).reshape(2, 5, 4))
    selections = sb.NestedTensor(
        [
            [sb.indices([3, 1, 3]), sb.indices([]), sb.indices([4, 0])],
            [sb.indices([2]), sb.indices([0, 0]), sb.indices([1, 4, 2])],
        ]
    )
    indexed = points[selections][:, ::-1]
    cpp_type = type(indexed._data)

    stream = io.BytesIO()
    sb.save(indexed, stream)
    payload = stream.getvalue()
    del points, selections

    restored = sb.load(io.BytesIO(payload))
    assert type(restored._data) is cpp_type
    assert restored._data.has_indices()
    _assert_same_clouds(restored, indexed)

    unpickled = pickle.loads(pickle.dumps(indexed))
    assert type(unpickled._data) is cpp_type
    assert unpickled._data.has_indices()
    _assert_same_clouds(unpickled, indexed)

    # A loaded result still performs one shared-state transition on its first
    # write. Existing outer and cell views follow that transition, while other
    # logical cells and repeated selected rows remain independent.
    sibling = restored[...]
    cell = restored[0, 2]
    other_cell = np.asarray(restored[0, 0]).copy()
    repeated_row = restored[0, 2][2, 1]
    restored[0, 2][0, 1] = 99
    assert sibling[0, 2][0, 1] == 99
    assert cell[0, 1] == 99
    assert restored[0, 2][2, 1] == repeated_row
    npt.assert_array_equal(np.asarray(restored[0, 0]), other_cell)

    rematerialized_roundtrip = sb.load(io.BytesIO(_saved_bytes(restored)))
    assert type(rematerialized_roundtrip._data) is cpp_type
    _assert_same_clouds(rematerialized_roundtrip, restored)


def test_indexed_scalar_and_empty_outer_roundtrip(make_pcloud_or_distmat_tensor):
    scalar = make_pcloud_or_distmat_tensor(np.arange(12).reshape(6, 2))
    dtype = scalar.dtype

    scalar_selection = sb.NestedTensor(sb.indices([5, 2, 5]))
    scalar_indexed = scalar[scalar_selection]
    cpp_type = type(scalar_indexed._data)
    scalar_restored = sb.load(io.BytesIO(_saved_bytes(scalar_indexed)))
    assert type(scalar_restored._data) is cpp_type
    assert scalar_restored.shape == ()
    _assert_same_clouds(scalar_restored, scalar_indexed)

    selected = scalar[
        sb.NestedTensor([sb.indices([5, 2, 5]), sb.indices([])])
    ]
    restored = sb.load(io.BytesIO(_saved_bytes(selected)))
    assert type(restored._data) is cpp_type
    _assert_same_clouds(restored, selected)

    empty_points = sb.zeros((0,), dtype=dtype)
    empty_selections = sb.NestedTensor([], dtype=sb.uint64, depth=2)
    empty_indexed = empty_points[empty_selections]
    empty_restored = sb.load(io.BytesIO(_saved_bytes(empty_indexed)))
    assert type(empty_restored._data) is cpp_type
    assert empty_restored.shape == (0,)
    assert empty_restored.dtype == dtype


@pytest.mark.parametrize(("dtype", "np_dtype", "cpp_type"), _DTYPES)
def test_indexed_payload_does_not_expand_selected_coordinates(
    dtype, np_dtype, cpp_type
):
    coordinates = np.arange(160, dtype=np_dtype).reshape(20, 8)
    points = sb.PointCloudTensor(coordinates, dtype=dtype)
    rows = np.arange(10, dtype=np.uint64)
    # Repeating the selection makes the dense payload duplicate its coordinate
    # rows, while the indexed payload stores one source plus row indices.
    selections = sb.NestedTensor([sb.indices(rows) for _ in range(20)])
    indexed = points[selections]

    indexed_payload = _saved_bytes(indexed)
    dense_payload = _saved_bytes(indexed.to_dense())

    assert type(indexed._data) is cpp_type
    assert len(indexed_payload) < len(dense_payload) // 2


def _saved_bytes(tensor):
    stream = io.BytesIO()
    sb.save(tensor, stream)
    return stream.getvalue()
