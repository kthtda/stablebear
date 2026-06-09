import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


def test_lp_norm_rejects_p_less_than_1():
    X = sb.zeros((2,))
    with pytest.raises(ValueError, match="p must be >= 1"):
        sb.lp_norm(X, p=0.5)


def test_lp_norm_rejects_p_zero():
    X = sb.zeros((2,))
    with pytest.raises(ValueError, match="p must be >= 1"):
        sb.lp_norm(X, p=0)


def test_lp_norm_rejects_p_negative():
    X = sb.zeros((2,))
    with pytest.raises(ValueError, match="p must be >= 1"):
        sb.lp_norm(X, p=-1)


def get_test_data():
    X = sb.zeros((2, 3), dtype=sb.pcf64)

    X[0, 0] = sb.Pcf(np.array([[0.0, 10.0], [2.0, 5.0], [4.0, 0.0]]))
    X[0, 1] = sb.Pcf(np.array([[0.0, 5.0], [3.0, 6.0], [5.0, 5.0], [8.0, 0.0]]))
    X[0, 2] = sb.Pcf(np.array([[0.0, 12.0], [3.0, 0.0]]))

    X[1, 0] = sb.Pcf(np.array([[0.0, 50.0], [0.5, 0.0]]))
    X[1, 1] = sb.Pcf(np.array([[0.0, 0.0]]))
    X[1, 2] = sb.Pcf(np.array([[0.0, 2.0], [3.0, 0.0]]))

    return X


def test_l1_norm():
    X = get_test_data()

    expected = np.zeros((2, 3))

    expected[0, 0] = 2 * 10 + 2 * 5
    expected[0, 1] = 3 * 5 + 2 * 6 + 3 * 5
    expected[0, 2] = 3 * 12

    expected[1, 0] = 0.5 * 50
    expected[1, 1] = 0
    expected[1, 2] = 3 * 2

    actual = sb.lp_norm(X, p=1)

    assert isinstance(actual, sb.FloatTensor)
    assert actual.shape == (2, 3)

    npt.assert_allclose(np.asarray(actual), expected, rtol=1e-7, atol=0)
