import numpy as np
import pytest

import stablebear as sb
from stablebear.distance_matrix import DistanceMatrix


def test_pdist_rejects_p_less_than_1():
    X = sb.zeros((2,))
    with pytest.raises(ValueError, match="p must be >= 1"):
        sb.pdist(X, p=0.5)


def test_pdist_rejects_p_zero():
    X = sb.zeros((2,))
    with pytest.raises(ValueError, match="p must be >= 1"):
        sb.pdist(X, p=0)


def test_pdist_rejects_p_negative():
    X = sb.zeros((2,))
    with pytest.raises(ValueError, match="p must be >= 1"):
        sb.pdist(X, p=-1)


def test_pdist_requires_1d_tensor():
    X = sb.zeros((10, 20))

    with pytest.raises(ValueError):
        sb.pdist(X)

    sb.pdist(X[:, 2])


def test_pdist_returns_distance_matrix():
    X = sb.zeros((3,))
    D = sb.pdist(X)
    assert isinstance(D, DistanceMatrix)


def test_pdist_of_empty_gives_empty():
    X = sb.zeros((0,))
    D = sb.pdist(X)

    assert isinstance(D, DistanceMatrix)
    assert D.size == 0


def test_pdist_of_one_gives_zero_1x1():
    X = sb.zeros((1,))
    D = sb.pdist(X)

    assert D.size == 1
    assert D[0, 0] == 0.0


def test_pdist_of_two_gives_correct_output():
    X = sb.zeros((2,), dtype=sb.pcf64)

    X[0] = sb.Pcf(np.array([[0.0, 10.0], [2.0, 5.0], [3.0, 0.0]]))
    X[1] = sb.Pcf(np.array([[0.0, 5.0], [6.0, 0.0]]))

    D = sb.pdist(X)

    assert D.size == 2

    assert D[0, 0] == 0.0
    assert D[0, 1] == pytest.approx(2 * 5 + 3 * 5)
    assert D[1, 0] == D[0, 1]
    assert D[1, 1] == 0.0


def test_pdist_to_dense():
    X = sb.zeros((2,), dtype=sb.pcf64)

    X[0] = sb.Pcf(np.array([[0.0, 10.0], [2.0, 5.0], [3.0, 0.0]]))
    X[1] = sb.Pcf(np.array([[0.0, 5.0], [6.0, 0.0]]))

    D = sb.pdist(X)
    dense = D.to_dense()

    assert isinstance(dense, np.ndarray)
    assert dense.shape == (2, 2)
    assert dense[0, 0] == 0.0
    assert dense[0, 1] == pytest.approx(2 * 5 + 3 * 5)
    assert dense[1, 0] == dense[0, 1]
    assert dense[1, 1] == 0.0


def test_pdist_l3_constant_pcfs():
    X = sb.zeros((2,), dtype=sb.pcf64)

    # f(t) = 4 on [0, 1), g(t) = 1 on [0, 1)
    X[0] = sb.Pcf(np.array([[0.0, 4.0], [1.0, 0.0]]))
    X[1] = sb.Pcf(np.array([[0.0, 1.0], [1.0, 0.0]]))

    D = sb.pdist(X, p=3)

    assert isinstance(D, DistanceMatrix)
    assert D.size == 2
    # ||f - g||_3 = (integral |4 - 1|^3 dt)^(1/3) = (27)^(1/3) = 3
    assert D[0, 1] == pytest.approx(3.0)
    assert D[0, 0] == 0.0
    assert D[1, 1] == 0.0


def test_pdist_lp_returns_distance_matrix():
    X = sb.zeros((3,))
    D = sb.pdist(X, p=3)
    assert isinstance(D, DistanceMatrix)


def test_from_dense_valid():
    dense = np.array([[0.0, 1.0, 2.0],
                       [1.0, 0.0, 3.0],
                       [2.0, 3.0, 0.0]])
    dm = DistanceMatrix.from_dense(dense)
    assert dm.size == 3
    assert dm[0, 1] == 1.0
    assert dm[0, 2] == 2.0
    assert dm[1, 2] == 3.0


def test_from_dense_rejects_nonzero_diagonal():
    dense = np.array([[1.0, 0.0],
                       [0.0, 0.0]])
    with pytest.raises(ValueError, match="Diagonal"):
        DistanceMatrix.from_dense(dense)


def test_from_dense_rejects_negative():
    dense = np.array([[0.0, -1.0],
                       [-1.0, 0.0]])
    with pytest.raises(ValueError, match="nonnegative"):
        DistanceMatrix.from_dense(dense)


def test_from_dense_rejects_asymmetric():
    dense = np.array([[0.0, 1.0],
                       [2.0, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        DistanceMatrix.from_dense(dense)


# --- Tests for PcfContainerLike acceptance (list / single Pcf) ---


def test_pdist_accepts_list_of_pcfs():
    f = sb.Pcf(np.array([[0.0, 10.0], [2.0, 5.0], [3.0, 0.0]]))
    g = sb.Pcf(np.array([[0.0, 5.0], [6.0, 0.0]]))

    D = sb.pdist([f, g], verbose=False)

    assert isinstance(D, DistanceMatrix)
    assert D.size == 2
    assert D[0, 0] == 0.0
    assert D[0, 1] == pytest.approx(2 * 5 + 3 * 5)


def test_cdist_accepts_lists_of_pcfs():
    f = sb.Pcf(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
    g = sb.Pcf(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64))

    D = sb.cdist([f], [g], verbose=False)

    assert D.shape == (1, 1)
    assert float(np.asarray(D).flat[0]) == pytest.approx(1.0)


def test_lp_norm_accepts_list_of_pcfs():
    f = sb.Pcf(np.array([[0.0, 3.0], [1.0, 0.0]], dtype=np.float64))

    norms = sb.lp_norm([f], verbose=False)

    assert norms.shape == (1,)
    assert norms[0] == pytest.approx(3.0)


def test_cdist_rejects_mismatched_dtypes():
    f32 = sb.Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
    f64 = sb.Pcf(np.array([[0.0, 1.0]], dtype=np.float64))

    X = sb.PcfTensor([f32])
    Y = sb.PcfTensor([f64])

    with pytest.raises(TypeError, match="same dtype"):
        sb.cdist(X, Y, verbose=False)


def test_pdist_accepts_empty_list():
    D = sb.pdist([], verbose=False)

    assert isinstance(D, DistanceMatrix)
    assert D.size == 0
