# Copyright 2024-2026 Bjorn Wehlin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

import stablebear as sb
from stablebear.tensor import IntTensor


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
