import numpy as np
import pytest

import stablebear as sb


_NUMERIC_TYPES = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
    pytest.param(sb.IntTensor, np.int32, id="int32"),
    pytest.param(sb.IntTensor, np.int64, id="int64"),
]


def _assert_concatenate(arrays, axis, TensorType, np_dtype):
    np_arrays = tuple(np.asarray(a, dtype=np_dtype) for a in arrays)
    tensors = tuple(TensorType(a) for a in np_arrays)
    result = np.asarray(sb.concatenate(tensors, axis=axis))
    expected = np.concatenate(np_arrays, axis=axis)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


def _assert_stack(arrays, axis, TensorType, np_dtype):
    np_arrays = tuple(np.asarray(a, dtype=np_dtype) for a in arrays)
    tensors = tuple(TensorType(a) for a in np_arrays)
    result = np.asarray(sb.stack(tensors, axis=axis))
    expected = np.stack(np_arrays, axis=axis)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


# --- concatenate ---


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestConcatenate:
    def test_1d(self, TensorType, np_dtype):
        _assert_concatenate(([1, 2, 3], [4, 5]), 0, TensorType, np_dtype)

    def test_2d_axis0(self, TensorType, np_dtype):
        _assert_concatenate(
            ([[1, 2], [3, 4]], [[5, 6]]), 0, TensorType, np_dtype)

    def test_2d_axis1(self, TensorType, np_dtype):
        a = np.arange(6, dtype=np_dtype).reshape(2, 3)
        b = np.arange(4, dtype=np_dtype).reshape(2, 2)
        _assert_concatenate((a, b), 1, TensorType, np_dtype)

    def test_three_tensors(self, TensorType, np_dtype):
        _assert_concatenate(([1, 2], [3, 4], [5, 6]), 0, TensorType, np_dtype)

    def test_mismatched_shape_raises(self, TensorType, np_dtype):
        a = TensorType(np.zeros((2, 3), dtype=np_dtype))
        b = TensorType(np.zeros((2, 4), dtype=np_dtype))
        with pytest.raises((ValueError, RuntimeError)):
            sb.concatenate((a, b), axis=0)


# --- stack ---


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestStack:
    def test_1d_axis0(self, TensorType, np_dtype):
        _assert_stack(([1, 2, 3], [4, 5, 6]), 0, TensorType, np_dtype)

    def test_1d_axis1(self, TensorType, np_dtype):
        _assert_stack(([1, 2, 3], [4, 5, 6]), 1, TensorType, np_dtype)

    def test_2d_axis0(self, TensorType, np_dtype):
        a = np.arange(6, dtype=np_dtype).reshape(2, 3)
        b = np.arange(6, 12, dtype=np_dtype).reshape(2, 3)
        _assert_stack((a, b), 0, TensorType, np_dtype)

    def test_2d_axis_neg1(self, TensorType, np_dtype):
        a = np.arange(6, dtype=np_dtype).reshape(2, 3)
        b = np.arange(6, 12, dtype=np_dtype).reshape(2, 3)
        _assert_stack((a, b), -1, TensorType, np_dtype)

    def test_three_tensors(self, TensorType, np_dtype):
        _assert_stack(([1, 2], [3, 4], [5, 6]), 0, TensorType, np_dtype)

    def test_mismatched_shape_raises(self, TensorType, np_dtype):
        a = TensorType(np.zeros((2, 3), dtype=np_dtype))
        b = TensorType(np.zeros((2, 4), dtype=np_dtype))
        with pytest.raises((ValueError, RuntimeError)):
            sb.stack((a, b), axis=0)
