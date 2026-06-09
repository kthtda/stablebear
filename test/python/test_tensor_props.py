import numpy as np
import pytest

import stablebear as sb


_NUMERIC_TYPES = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
    pytest.param(sb.IntTensor, np.int32, id="int32"),
    pytest.param(sb.IntTensor, np.int64, id="int64"),
]


def _assert_ndim(np_arr, TensorType):
    t = TensorType(np_arr)
    assert t.ndim == np_arr.ndim


def _assert_size(np_arr, TensorType):
    t = TensorType(np_arr)
    assert t.size == np_arr.size


def _assert_len(np_arr, TensorType):
    t = TensorType(np_arr)
    assert len(t) == len(np_arr)


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestNdim:
    def test_1d(self, TensorType, np_dtype):
        _assert_ndim(np.arange(5, dtype=np_dtype), TensorType)

    def test_2d(self, TensorType, np_dtype):
        _assert_ndim(np.arange(12, dtype=np_dtype).reshape(3, 4), TensorType)

    def test_3d(self, TensorType, np_dtype):
        _assert_ndim(np.arange(24, dtype=np_dtype).reshape(2, 3, 4), TensorType)

    def test_after_squeeze(self, TensorType, np_dtype):
        np_arr = np.arange(6, dtype=np_dtype).reshape(1, 6, 1)
        t = TensorType(np_arr)
        assert t.squeeze().ndim == np_arr.squeeze().ndim

    def test_after_expand_dims(self, TensorType, np_dtype):
        np_arr = np.arange(6, dtype=np_dtype)
        t = TensorType(np_arr)
        assert t.expand_dims(0).ndim == np.expand_dims(np_arr, 0).ndim


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestSize:
    def test_1d(self, TensorType, np_dtype):
        _assert_size(np.arange(5, dtype=np_dtype), TensorType)

    def test_2d(self, TensorType, np_dtype):
        _assert_size(np.arange(12, dtype=np_dtype).reshape(3, 4), TensorType)

    def test_3d(self, TensorType, np_dtype):
        _assert_size(np.arange(24, dtype=np_dtype).reshape(2, 3, 4), TensorType)

    def test_empty(self, TensorType, np_dtype):
        np_arr = np.zeros((0, 3), dtype=np_dtype)
        t = TensorType(np_arr)
        assert t.size == 0

    def test_after_slice(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype).reshape(3, 4)
        t = TensorType(np_arr)
        assert t[1:, :2].size == np_arr[1:, :2].size


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestLen:
    def test_1d(self, TensorType, np_dtype):
        _assert_len(np.arange(5, dtype=np_dtype), TensorType)

    def test_2d(self, TensorType, np_dtype):
        _assert_len(np.arange(12, dtype=np_dtype).reshape(3, 4), TensorType)

    def test_3d(self, TensorType, np_dtype):
        _assert_len(np.arange(24, dtype=np_dtype).reshape(2, 3, 4), TensorType)

    def test_empty(self, TensorType, np_dtype):
        np_arr = np.zeros((0, 3), dtype=np_dtype)
        t = TensorType(np_arr)
        assert len(t) == 0
