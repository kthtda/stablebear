import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb
from stablebear.distance_matrix import DistanceMatrix


def test_tensor_asarray_views_the_tensor():
    tensor = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))

    array = np.asarray(tensor)
    array[0] = 7.0

    npt.assert_array_equal(np.asarray(tensor), [7.0, 2.0, 3.0])


def test_tensor_array_copies():
    tensor = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))

    array = np.array(tensor)
    array[0] = 7.0

    npt.assert_array_equal(array, [7.0, 2.0, 3.0])
    npt.assert_array_equal(np.asarray(tensor), [1.0, 2.0, 3.0])


def test_tensor_asarray_without_copy_rejects_dtype_conversion():
    tensor = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError):
        np.asarray(tensor, dtype=np.float32, copy=False)


def test_noncontiguous_tensor_asarray_without_copy_raises():
    tensor = sb.FloatTensor(np.array([1.0, 2.0, 3.0, 4.0]))

    # Every other element
    with pytest.raises(ValueError):
        np.asarray(tensor[::2], copy=False)


def test_pcf_array_with_copy_does_not_share_memory():
    pcf = sb.Pcf(np.array([[0.0, 1.0],
                           [1.0, 2.0],
                           [2.0, 0.0]]))

    array = np.array(pcf, copy=True)

    assert not np.shares_memory(array, np.asarray(pcf))
    npt.assert_array_equal(array, [[0.0, 1.0],
                                   [1.0, 2.0],
                                   [2.0, 0.0]])


def test_pcf_asarray_without_copy_rejects_dtype_conversion():
    pcf = sb.Pcf(np.array([[0.0, 1.0],
                           [1.0, 2.0],
                           [2.0, 0.0]]))

    with pytest.raises(ValueError):
        np.asarray(pcf, dtype=np.float32, copy=False)


def test_distance_matrix_asarray_without_copy_raises():
    # The dense matrix is always built anew
    matrix = DistanceMatrix(np.array([[0.0, 1.0],
                                      [1.0, 0.0]]))

    with pytest.raises(ValueError):
        np.asarray(matrix, copy=False)
