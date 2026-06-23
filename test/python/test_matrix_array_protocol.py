"""Regression tests for issue #75: ``np.asarray`` on a DistanceMatrix /
SymmetricMatrix used to silently return a 0-d object array wrapping the C++
object instead of materializing the dense (n, n) matrix."""

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb
from stablebear.distance_matrix import DistanceMatrix
from stablebear.symmetric_matrix import SymmetricMatrix


def _sample_distance_array():
    return np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )


def _sample_symmetric_array():
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [3.0, 5.0, 6.0],
        ]
    )


@pytest.mark.parametrize("convert", [np.asarray, np.array])
def test_distance_matrix_asarray_is_dense(convert):
    dense = _sample_distance_array()
    dm = DistanceMatrix.from_dense(dense)
    arr = convert(dm)
    assert arr.dtype != object
    assert arr.shape == (3, 3)
    npt.assert_allclose(arr, dense)


@pytest.mark.parametrize("convert", [np.asarray, np.array])
def test_symmetric_matrix_asarray_is_dense(convert):
    dense = _sample_symmetric_array()
    sm = SymmetricMatrix.from_dense(dense)
    arr = convert(sm)
    assert arr.dtype != object
    assert arr.shape == (3, 3)
    npt.assert_allclose(arr, dense)


def test_distance_matrix_to_numpy_alias():
    dm = DistanceMatrix.from_dense(_sample_distance_array())
    npt.assert_allclose(dm.to_numpy(), dm.to_dense())


def test_symmetric_matrix_to_numpy_alias():
    sm = SymmetricMatrix.from_dense(_sample_symmetric_array())
    npt.assert_allclose(sm.to_numpy(), sm.to_dense())


def test_asarray_dtype_argument():
    dm = DistanceMatrix.from_dense(_sample_distance_array())
    arr = np.asarray(dm, dtype=np.float32)
    assert arr.dtype == np.float32
    npt.assert_allclose(arr, _sample_distance_array())


def test_pdist_result_asarray():
    """The motivating scenario: np.asarray on a pdist result is the n×n matrix."""
    fs = sb.random.noisy_sin((4,))
    dm = sb.pdist(fs)
    arr = np.asarray(dm)
    assert arr.dtype != object
    assert arr.shape == (4, 4)
    # Distance matrix is symmetric with a zero diagonal.
    npt.assert_allclose(arr, arr.T)
    npt.assert_allclose(np.diag(arr), np.zeros(4))
