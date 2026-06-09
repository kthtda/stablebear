import numpy as np
import pytest

import stablebear as sb


_NUMERIC_TYPES = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
    pytest.param(sb.IntTensor, np.int32, id="int32"),
    pytest.param(sb.IntTensor, np.int64, id="int64"),
]


def _assert_iter(np_arr, TensorType):
    """Assert that iterating a stablebear tensor matches iterating a NumPy array."""
    t = TensorType(np_arr)
    for sb_item, np_item in zip(t, np_arr):
        np.testing.assert_array_equal(np.asarray(sb_item), np_item)


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestIter:
    def test_1d(self, TensorType, np_dtype):
        _assert_iter(np.arange(5, dtype=np_dtype), TensorType)

    def test_2d(self, TensorType, np_dtype):
        _assert_iter(np.arange(12, dtype=np_dtype).reshape(3, 4), TensorType)

    def test_3d(self, TensorType, np_dtype):
        _assert_iter(np.arange(24, dtype=np_dtype).reshape(2, 3, 4), TensorType)

    def test_length_matches(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype).reshape(3, 4)
        t = TensorType(np_arr)
        assert len(list(t)) == len(np_arr)

    def test_list_conversion(self, TensorType, np_dtype):
        np_arr = np.arange(6, dtype=np_dtype).reshape(2, 3)
        t = TensorType(np_arr)
        items = list(t)
        assert len(items) == 2
        for sb_item, np_item in zip(items, np_arr):
            np.testing.assert_array_equal(np.asarray(sb_item), np_item)

    def test_unpacking(self, TensorType, np_dtype):
        np_arr = np.arange(6, dtype=np_dtype).reshape(2, 3)
        t = TensorType(np_arr)
        a, b = t
        np.testing.assert_array_equal(np.asarray(a), np_arr[0])
        np.testing.assert_array_equal(np.asarray(b), np_arr[1])

    def test_nested_iter(self, TensorType, np_dtype):
        np_arr = np.arange(24, dtype=np_dtype).reshape(2, 3, 4)
        t = TensorType(np_arr)
        for sb_row, np_row in zip(t, np_arr):
            for sb_item, np_item in zip(sb_row, np_row):
                np.testing.assert_array_equal(np.asarray(sb_item), np_item)

    def test_empty(self, TensorType, np_dtype):
        np_arr = np.zeros((0, 3), dtype=np_dtype)
        t = TensorType(np_arr)
        assert list(t) == []

    def test_transposed(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype).reshape(3, 4)
        t = TensorType(np_arr)
        _assert_iter(np_arr.T, type(t.T))
        for sb_item, np_item in zip(t.T, np_arr.T):
            np.testing.assert_array_equal(np.asarray(sb_item), np_item)
