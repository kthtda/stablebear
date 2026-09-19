import numpy as np
import pytest

import stablebear as sb
import stablebear.persistence as pers


def test_persistence_ripser_compute_euclidean_barcode_from_pcloud_returns_correct_dtype_and_shape():
    Xnp = np.random.randn(10, 2).astype(np.float64)
    X = sb.FloatTensor(Xnp)

    bcs = pers.compute_persistent_homology(
        X,
        max_dim=3,
        complex_type=pers.ComplexType.VietorisRips,
        distance_type=pers.DistanceType.Euclidean,
    )

    assert isinstance(bcs, pers.BarcodeTensor)
    assert bcs.dtype == sb.barcode64
    assert bcs.shape == (4,)

    Xnp = np.random.randn(10, 2).astype(np.float32)
    X = sb.FloatTensor(Xnp)

    bcs = pers.compute_persistent_homology(
        X,
        max_dim=3,
        complex_type=pers.ComplexType.VietorisRips,
        distance_type=pers.DistanceType.Euclidean,
    )

    assert isinstance(bcs, pers.BarcodeTensor)
    assert bcs.dtype == sb.barcode32
    assert bcs.shape == (4,)


def test_persistence_ripser_compute_euclidean_barcode_from_pcloud_tensor_returns_correct_dtype_and_shape():
    X = sb.zeros((3, 2, 7), dtype=sb.pcloud64)

    bcs = pers.compute_persistent_homology(
        X,
        max_dim=3,
        complex_type=pers.ComplexType.VietorisRips,
        distance_type=pers.DistanceType.Euclidean,
    )

    assert isinstance(bcs, pers.BarcodeTensor)
    assert bcs.dtype == sb.barcode64
    assert bcs.shape == (3, 2, 7, 4)

    X = sb.zeros((3, 2, 7), dtype=sb.pcloud32)

    bcs = pers.compute_persistent_homology(
        X,
        max_dim=3,
        complex_type=pers.ComplexType.VietorisRips,
        distance_type=pers.DistanceType.Euclidean,
    )

    assert isinstance(bcs, pers.BarcodeTensor)
    assert bcs.dtype == sb.barcode32
    assert bcs.shape == (3, 2, 7, 4)


def _make_rectangle_point_cloud():
    # Distance space is "two 3-4-5 triangles". This gives nontrivial H0 and H1 with bars at integers.
    X = np.zeros((4, 2))
    X[0, :] = [0.0, 0.0]
    X[1, :] = [0.0, 4.0]
    X[2, :] = [3.0, 0.0]
    X[3, :] = [3.0, 4.0]
    return X


def _assert_barcode_tensors_isomorphic(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    for index in np.ndindex(*actual.shape):
        assert actual[index].is_isomorphic_to(expected[index])


def _indexed_rectangle_tensors(pcloud_dtype, np_dtype):
    rectangle = _make_rectangle_point_cloud().astype(np_dtype)
    leaf = lambda rows: sb.tensor(rows, dtype=sb.uint64)
    selections = sb.NestedTensor(
        [
            [leaf([3, 0, 2, 1]), leaf([1, 1, 0, 3]), leaf([2, 0, 2, 3])],
            [leaf([0, 3, 1, 2]), leaf([2, 2, 1, 0]), leaf([3, 1, 0, 2])],
        ]
    )
    source = sb.PointCloudTensor(
        [rectangle, rectangle * np_dtype(2)], dtype=pcloud_dtype
    )
    return source[selections]


def test_persistence_ripser_unreduced_homology():
    X = _make_rectangle_point_cloud()

    bcs = pers.compute_persistent_homology(
        X,
        max_dim=2,
        complex_type=pers.ComplexType.VietorisRips,
        distance_type=pers.DistanceType.Euclidean,
    )

    h0 = bcs[0]
    h1 = bcs[1]
    h2 = bcs[2]

    expected_h0 = pers.Barcode(np.array([
        [0.0, np.inf], [0.0, 3.0], [0.0, 3.0], [0.0, 4.0],
    ]))
    expected_h1 = pers.Barcode(np.array([[4.0, 5.0]]))
    expected_h2 = pers.Barcode(np.zeros((0, 2)))

    assert expected_h0.is_isomorphic_to(h0)
    assert expected_h1.is_isomorphic_to(h1)
    assert expected_h2.is_isomorphic_to(h2)


def test_persistence_ripser_reduced_homology():
    X = _make_rectangle_point_cloud()

    bcs = pers.compute_persistent_homology(
        X,
        max_dim=2,
        reduced=True,
        complex_type=pers.ComplexType.VietorisRips,
        distance_type=pers.DistanceType.Euclidean,
    )

    h0 = bcs[0]
    h1 = bcs[1]
    h2 = bcs[2]

    expected_h0 = pers.Barcode(np.array([[0.0, 3.0], [0.0, 3.0], [0.0, 4.0]]))
    expected_h1 = pers.Barcode(np.array([[4.0, 5.0]]))
    expected_h2 = pers.Barcode(np.zeros((0, 2)))

    assert expected_h0.is_isomorphic_to(h0)
    assert expected_h1.is_isomorphic_to(h1)
    assert expected_h2.is_isomorphic_to(h2)


def test_persistence_ripser_compute_euclidean_barcode_on_tensor():
    X = sb.zeros((3, 4, 5), dtype=sb.pcloud64)

    for i in range(3):
        for j in range(4):
            for k in range(5):
                X[i, j, k] = np.random.randn(10, 5)

    Y = pers.compute_persistent_homology(
        X,
        max_dim=1,
        complex_type=pers.ComplexType.VietorisRips,
        distance_type=pers.DistanceType.Euclidean,
    )

    for i in range(3):
        for j in range(4):
            for k in range(5):
                xbc = pers.compute_persistent_homology(
                    X[i, j, k],
                    max_dim=1,
                    complex_type=pers.ComplexType.VietorisRips,
                    distance_type=pers.DistanceType.Euclidean,
                )

                assert Y[i, j, k, 0].is_isomorphic_to(xbc[0])
                assert Y[i, j, k, 1].is_isomorphic_to(xbc[1])


@pytest.mark.parametrize(
    "pcloud_dtype,np_dtype",
    [(sb.pcloud32, np.float32), (sb.pcloud64, np.float64)],
)
@pytest.mark.parametrize("reduced", [False, True])
def test_persistence_accepts_whole_indexed_tensors_and_outer_views(
    pcloud_dtype, np_dtype, reduced
):
    indexed_tensor = _indexed_rectangle_tensors(pcloud_dtype, np_dtype)

    for indexed in (indexed_tensor, indexed_tensor[1:, 1:]):
        indexed_storage_type = type(indexed._data)
        dense = indexed.to_dense()

        actual = pers.compute_persistent_homology(
            indexed, max_dim=2, reduced=reduced
        )
        expected = pers.compute_persistent_homology(
            dense, max_dim=2, reduced=reduced
        )

        assert type(indexed._data) is indexed_storage_type
        _assert_barcode_tensors_isomorphic(actual, expected)
