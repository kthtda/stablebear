"""Regression tests for issue #62: ``tensor + ndarray`` used to raise a
pybind ``TypeError`` even though ``ndarray + tensor`` worked, an asymmetry
that surprised users. The forward operators should now accept an ndarray
(or list/tuple) RHS and behave like the reflected ones."""

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


_BINOPS = [
    pytest.param(lambda x, y: x + y, id="add"),
    pytest.param(lambda x, y: x - y, id="sub"),
    pytest.param(lambda x, y: x * y, id="mul"),
    pytest.param(lambda x, y: x / y, id="truediv"),
]


@pytest.mark.parametrize("op", _BINOPS)
def test_forward_and_reflected_each_match_numpy(op):
    """Both directions must match numpy: previously only the reflected
    direction (arr + f) worked while the forward (f + arr) crashed. The
    forward op must also preserve the tensor type rather than decay to an
    ndarray."""
    np_a = np.array([1.0, 2.0, 3.0])
    arr = np.array([10.0, 20.0, 30.0])
    f = sb.FloatTensor(np_a)
    forward = op(f, arr)
    assert isinstance(forward, sb.FloatTensor)
    npt.assert_allclose(np.asarray(forward), op(np_a, arr))
    npt.assert_allclose(np.asarray(op(arr, f)), op(arr, np_a))


def test_forward_ndarray_broadcasts():
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr = np.array([10.0, 20.0])
    f = sb.FloatTensor(np_a)
    npt.assert_allclose(np.asarray(f + arr), np_a + arr)


def test_list_rhs():
    np_a = np.array([1.0, 2.0, 3.0])
    f = sb.FloatTensor(np_a)
    npt.assert_allclose(np.asarray(f + [10.0, 20.0, 30.0]), np_a + np.array([10.0, 20.0, 30.0]))


def test_inplace_add_ndarray():
    np_a = np.array([1.0, 2.0, 3.0])
    arr = np.array([10.0, 20.0, 30.0])
    f = sb.FloatTensor(np_a.copy())
    f += arr
    npt.assert_allclose(np.asarray(f), np_a + arr)


def test_scalar_rhs_still_works():
    np_a = np.array([1.0, 2.0, 3.0])
    f = sb.FloatTensor(np_a)
    npt.assert_allclose(np.asarray(f + 5), np_a + 5)


def test_int_tensor_plus_ndarray():
    np_a = np.array([1, 2, 3], dtype=np.int64)
    arr = np.array([10, 20, 30], dtype=np.int64)
    t = sb.IntTensor(np_a)
    npt.assert_array_equal(np.asarray(t + arr), np_a + arr)
