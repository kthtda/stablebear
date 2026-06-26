import numpy as np
import pytest

import stablebear as sb
from stablebear.base_tensor import IntTensor


_INT_DTYPES = [
    pytest.param(np.int32, sb.int32, id="int32"),
    pytest.param(np.int64, sb.int64, id="int64"),
    pytest.param(np.uint32, sb.uint32, id="uint32"),
    pytest.param(np.uint64, sb.uint64, id="uint64"),
]


# -- Construction and dtype inference --

@pytest.mark.parametrize("np_dtype, sb_dtype", _INT_DTYPES)
class TestIntTensorConstruction:
    def test_construct_from_array(self, np_dtype, sb_dtype):
        arr = np.array([1, 2, 3], dtype=np_dtype)
        t = IntTensor(arr)
        assert t.dtype == sb_dtype
        np.testing.assert_array_equal(np.asarray(t), arr)

    def test_numpy_roundtrip(self, np_dtype, sb_dtype):
        arr = np.array([10, 20, 30], dtype=np_dtype)
        t = IntTensor(arr)
        np.testing.assert_array_equal(np.asarray(t), arr)

    def test_zeros(self, np_dtype, sb_dtype):
        t = sb.zeros((3,), dtype=sb_dtype)
        assert isinstance(t, IntTensor)
        assert t.dtype == sb_dtype
        np.testing.assert_array_equal(np.asarray(t), np.zeros(3, dtype=np_dtype))


def test_construct_explicit_dtype():
    arr = np.array([1, 2, 3])
    t = IntTensor(arr, dtype=sb.int32)
    assert t.dtype == sb.int32
    np.testing.assert_array_equal(np.asarray(t), arr)


def test_construct_from_int_tensor():
    arr = np.array([5, 6, 7], dtype=np.int64)
    t1 = IntTensor(arr)
    t2 = IntTensor(t1)
    assert t2.dtype == sb.int64
    np.testing.assert_array_equal(np.asarray(t2), arr)


def test_construct_2d():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    t = IntTensor(arr)
    assert t.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(t), arr)


def test_construct_bad_type():
    with pytest.raises(TypeError):
        IntTensor("not a tensor")


def test_numpy_roundtrip_large_uint64():
    arr = np.array([0, 2**32, 2**40], dtype=np.uint64)
    t = IntTensor(arr)
    np.testing.assert_array_equal(np.asarray(t), arr)


# -- Division (matches NumPy: int / int -> float) --

def test_div_tensor_tensor():
    np_a = np.array([10, 21, 35], dtype=np.int64)
    np_b = np.array([3, 7, 5], dtype=np.int64)
    result = IntTensor(np_a) / IntTensor(np_b)
    assert isinstance(result, sb.FloatTensor)
    np.testing.assert_array_almost_equal(np.asarray(result), np_a / np_b)


def test_div_scalar():
    np_a = np.array([10, 21, 35], dtype=np.int64)
    result = IntTensor(np_a) / 2
    assert isinstance(result, sb.FloatTensor)
    np.testing.assert_array_almost_equal(np.asarray(result), np_a / 2)


def test_rdiv_scalar():
    np_a = np.array([2, 5, 10], dtype=np.int32)
    result = 100 / IntTensor(np_a)
    assert isinstance(result, sb.FloatTensor)
    np.testing.assert_array_almost_equal(np.asarray(result), 100 / np_a)


# -- Floor division (int // int -> int, matching NumPy) --

def test_floordiv_tensor_tensor():
    np_a = np.array([10, 21, -7], dtype=np.int64)
    np_b = np.array([3, 7, 2], dtype=np.int64)
    result = IntTensor(np_a) // IntTensor(np_b)
    assert isinstance(result, IntTensor)
    np.testing.assert_array_equal(np.asarray(result), np_a // np_b)


def test_floordiv_scalar():
    np_a = np.array([10, 21, 35], dtype=np.int32)
    result = IntTensor(np_a) // 4
    assert isinstance(result, IntTensor)
    np.testing.assert_array_equal(np.asarray(result), np_a // 4)


def test_rfloordiv_scalar():
    np_a = np.array([3, 7, 4], dtype=np.int32)
    result = 10 // IntTensor(np_a)
    assert isinstance(result, IntTensor)
    np.testing.assert_array_equal(np.asarray(result), 10 // np_a)


def test_ifloordiv():
    np_a = np.array([10, 21, 35], dtype=np.int64)
    t = IntTensor(np_a.copy())
    t //= 4
    np.testing.assert_array_equal(np.asarray(t), np_a // 4)


# -- Negation --

def test_negation_signed():
    arr = np.array([1, -2, 3], dtype=np.int32)
    t = IntTensor(arr)
    np.testing.assert_array_equal(np.asarray(-t), -arr)


def test_negation_unsigned_raises():
    t = IntTensor(np.array([1, 2, 3], dtype=np.uint32))
    with pytest.raises(TypeError):
        _ = -t

    t64 = IntTensor(np.array([1, 2, 3], dtype=np.uint64))
    with pytest.raises(TypeError):
        _ = -t64


# -- repr/str --

def test_repr():
    arr = np.array([1, 2, 3], dtype=np.int32)
    t = IntTensor(arr)
    r = repr(t)
    assert "1" in r and "2" in r and "3" in r


# ---------------------------------------------------------------------------
# Bug #7: a Python-float scalar / slice assignment into an IntTensor raised a
# raw pybind TypeError. It now truncates toward zero, as in NumPy.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("np_dtype, sb_dtype",
                         [pytest.param(np.int32, sb.int32, id="int32"),
                          pytest.param(np.int64, sb.int64, id="int64")])
class TestIntTensorFloatAssignTruncates:
    def test_scalar_assignment_truncates(self, np_dtype, sb_dtype):
        t = sb.IntTensor(np.array([1, 2, 3], dtype=np_dtype))
        ref = np.array([1, 2, 3], dtype=np_dtype)
        t[0] = 7.9
        ref[0] = 7.9
        np.testing.assert_array_equal(np.asarray(t), ref)

    def test_slice_assignment_truncates(self, np_dtype, sb_dtype):
        t = sb.IntTensor(np.array([1, 2, 3, 4], dtype=np_dtype))
        t[0:2] = 7.9
        np.testing.assert_array_equal(np.asarray(t)[:2], np.array([7, 7]))

    def test_negative_float_truncates_toward_zero(self, np_dtype, sb_dtype):
        if sb_dtype in (sb.int32, sb.int64):
            t = sb.IntTensor(np.array([1, 2, 3], dtype=np_dtype))
            t[0] = -7.9
            assert int(np.asarray(t)[0]) == -7

    def test_np_float64_scalar_truncates(self, np_dtype, sb_dtype):
        t = sb.IntTensor(np.array([1, 2, 3], dtype=np_dtype))
        t[0] = np.float64(5.7)
        assert int(np.asarray(t)[0]) == 5


# ---------------------------------------------------------------------------
# Bug #65: a fractional exponent on an IntTensor raised a raw pybind error, and
# a negative integer exponent silently truncated to 0. Fractional exponents now
# promote to float64 and a negative integer exponent raises ValueError, both as
# in NumPy.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("np_dtype, sb_dtype", _INT_DTYPES)
class TestIntTensorPow:
    def test_fractional_exponent_promotes_to_float(self, np_dtype, sb_dtype):
        t = sb.IntTensor(np.array([4, 9, 16], dtype=np_dtype))
        result = t ** 0.5
        assert isinstance(result, sb.FloatTensor)
        assert result.dtype == sb.float64
        np.testing.assert_allclose(
            np.asarray(result), np.array([4, 9, 16], dtype=np_dtype) ** 0.5)

    def test_whole_float_exponent_promotes_to_float(self, np_dtype, sb_dtype):
        t = sb.IntTensor(np.array([2, 3], dtype=np_dtype))
        result = t ** 2.0
        assert isinstance(result, sb.FloatTensor)
        np.testing.assert_allclose(np.asarray(result), np.array([4.0, 9.0]))

    def test_nonnegative_int_exponent_stays_int(self, np_dtype, sb_dtype):
        t = sb.IntTensor(np.array([2, 3], dtype=np_dtype))
        result = t ** 2
        assert isinstance(result, sb.IntTensor)
        np.testing.assert_array_equal(np.asarray(result), np.array([4, 9]))

    def test_inplace_fractional_exponent_raises(self, np_dtype, sb_dtype):
        t = sb.IntTensor(np.array([4, 9], dtype=np_dtype))
        with pytest.raises(TypeError):
            t **= 0.5

    def test_inplace_int_exponent_works(self, np_dtype, sb_dtype):
        t = sb.IntTensor(np.array([2, 3], dtype=np_dtype))
        t **= 3
        np.testing.assert_array_equal(np.asarray(t), np.array([8, 27]))


@pytest.mark.parametrize("np_dtype, sb_dtype",
                         [pytest.param(np.int32, sb.int32, id="int32"),
                          pytest.param(np.int64, sb.int64, id="int64")])
def test_negative_integer_exponent_raises(np_dtype, sb_dtype):
    t = sb.IntTensor(np.array([2, 3], dtype=np_dtype))
    with pytest.raises(ValueError, match="negative integer powers"):
        t ** -1
