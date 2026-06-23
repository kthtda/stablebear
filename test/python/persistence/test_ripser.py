import numpy as np

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
