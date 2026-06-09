import copy

import numpy as np
import pytest
from utils import np_strides_in_items

import stablebear as sb


_NUMERIC_TYPES = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
    pytest.param(sb.IntTensor, np.int32, id="int32"),
    pytest.param(sb.IntTensor, np.int64, id="int64"),
]


def _populate(np_arr):
    """Fill an array with unique values based on index: 100*i + 10*j + k ..."""
    it = np.nditer(np_arr, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        idx = it.multi_index
        val = sum(v * 10 ** (len(idx) - 1 - i) for i, v in enumerate(idx))
        it[0] = val
        it.iternext()
    return np_arr


def _assert_extract(np_arr, index, TensorType=sb.FloatTensor):
    """Assert that stablebear slicing matches numpy slicing."""
    t = TensorType(np_arr)
    result = np.asarray(t[index]) if not isinstance(t[index], (int, float)) else t[index]
    expected = np_arr[index]
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
        assert result.shape == expected.shape
    else:
        assert result == expected


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestScalarIndexing:
    def test_extract_element(self, TensorType, np_dtype):
        np_arr = np.array([[0, 0, 0], [0, 0, 0]], dtype=np_dtype)
        np_arr[0, 1] = 2
        t = TensorType(np_arr)
        assert t[1, 0] == np_arr[1, 0]
        assert t[0, 1] == np_arr[0, 1]

    def test_extract_element1d(self, TensorType, np_dtype):
        np_arr = np.array([0, 2], dtype=np_dtype)
        t = TensorType(np_arr)
        assert t[0] == np_arr[0]
        assert t[1] == np_arr[1]

    def test_slice_getitem(self, TensorType, np_dtype):
        np_arr = np.array([10, 20, 30, 40], dtype=np_dtype)
        t = TensorType(np_arr)
        s = t[1:3]
        assert isinstance(s, TensorType)
        np.testing.assert_array_equal(np.asarray(s), np_arr[1:3])

    def test_2d_indexing(self, TensorType, np_dtype):
        np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype)
        t = TensorType(np_arr)
        assert t[0, 1] == np_arr[0, 1]
        assert t[1, 2] == np_arr[1, 2]


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestCopy:
    def test_copy(self, TensorType, np_dtype):
        arr = np.array([1, 2, 3], dtype=np_dtype)
        t = TensorType(arr)
        t2 = t.copy()
        assert isinstance(t2, TensorType)
        np.testing.assert_array_equal(np.asarray(t2), arr)
        # Mutating the copy must not affect the original
        t2[0] = 99
        assert t[0] == 1

    def test_deepcopy(self, TensorType, np_dtype):
        arr = np.array([1, 2, 3], dtype=np_dtype)
        t = TensorType(arr)
        t2 = copy.deepcopy(t)
        assert isinstance(t2, TensorType)
        np.testing.assert_array_equal(np.asarray(t2), arr)
        # Mutating the deep copy must not affect the original
        t2[0] = 99
        assert t[0] == 1


# --- Float-specific stride and multi-dim extract tests ---


def test_extract_subtensor():
    np_arr = _populate(np.zeros((3, 4, 5), dtype=np.float64))
    t = sb.FloatTensor(np_arr)

    np_sub = np_arr[:, 1:3, 2:4]
    sub = t[:, 1:3, 2:4]

    assert sub.shape == np_sub.shape

    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            for k in range(sub.shape[2]):
                assert sub[i, j, k] == np_sub[i, j, k]


def test_extract1d_with_step():
    np_arr = np.arange(6, dtype=np.float64)
    t = sb.FloatTensor(np_arr)

    np_view = np_arr[0:5:2]
    view = t[0:5:2]

    assert view.shape == np_view.shape
    np.testing.assert_array_equal(np.asarray(view), np_view)


def test_extract_with_step():
    np_arr = _populate(np.zeros((3, 9, 2), dtype=np.float64))
    t = sb.FloatTensor(np_arr)

    np_view = np_arr[:, 0:7:2, :]
    view = t[:, 0:7:2, :]

    assert view.shape == np_view.shape
    assert view.strides == np_strides_in_items(np_view)

    for i in range(view.shape[0]):
        for j in range(view.shape[1]):
            for k in range(view.shape[2]):
                assert view[i, j, k] == np_view[i, j, k]


def test_extract_with_offsets():
    np_arr = _populate(np.zeros((7, 9, 5), dtype=np.float64))
    t = sb.FloatTensor(np_arr)

    np_view = np_arr[1::3, 3:7:2, 2:5]
    view = t[1::3, 3:7:2, 2:5]

    assert view.shape == np_view.shape
    assert view.strides == np_strides_in_items(np_view)

    for i in range(view.shape[0]):
        for j in range(view.shape[1]):
            for k in range(view.shape[2]):
                assert view[i, j, k] == np_view[i, j, k]


def test_recursive_extract():
    np_arr = _populate(np.zeros((9, 8, 7, 6), dtype=np.float64))
    t = sb.FloatTensor(np_arr)

    np_v0 = np_arr[4:9:2, ::3, 2:5, 1:3:2]
    v0 = t[4:9:2, ::3, 2:5, 1:3:2]

    assert v0.shape == np_v0.shape
    assert v0.strides == np_strides_in_items(np_v0)

    for i in range(v0.shape[0]):
        for j in range(v0.shape[1]):
            for k in range(v0.shape[2]):
                for l in range(v0.shape[3]):
                    assert v0[i, j, k, l] == np_v0[i, j, k, l]

    np_v1 = np_v0[1:3, :, :2, 0]
    v1 = v0[1:3, :, :2, 0]

    assert v1.shape == np_v1.shape
    assert v1.strides == np_strides_in_items(np_v1)

    for i in range(v1.shape[0]):
        for j in range(v1.shape[1]):
            for k in range(v1.shape[2]):
                assert v1[i, j, k] == np_v1[i, j, k]


# --- Negative strides ---


def test_reverse_1d():
    np_arr = np.arange(5, dtype=np.float64)
    t = sb.FloatTensor(np_arr)
    np.testing.assert_array_equal(np.asarray(t[::-1]), np_arr[::-1])


def test_reverse_1d_step2():
    np_arr = np.arange(6, dtype=np.float64)
    t = sb.FloatTensor(np_arr)
    np.testing.assert_array_equal(np.asarray(t[::-2]), np_arr[::-2])


def test_negative_step_with_bounds():
    np_arr = np.arange(10, dtype=np.float64)
    t = sb.FloatTensor(np_arr)
    np.testing.assert_array_equal(np.asarray(t[7:2:-1]), np_arr[7:2:-1])
    np.testing.assert_array_equal(np.asarray(t[8:1:-3]), np_arr[8:1:-3])


def test_negative_step_2d():
    np_arr = _populate(np.zeros((4, 5), dtype=np.float64))
    t = sb.FloatTensor(np_arr)
    np.testing.assert_array_equal(np.asarray(t[::-1, :]), np_arr[::-1, :])
    np.testing.assert_array_equal(np.asarray(t[:, ::-1]), np_arr[:, ::-1])
    np.testing.assert_array_equal(np.asarray(t[::-1, ::-1]), np_arr[::-1, ::-1])


def test_negative_step_2d_with_bounds():
    np_arr = _populate(np.zeros((5, 6), dtype=np.float64))
    t = sb.FloatTensor(np_arr)
    np.testing.assert_array_equal(np.asarray(t[3:0:-1, 5:1:-2]), np_arr[3:0:-1, 5:1:-2])


def test_negative_step_empty_result():
    np_arr = np.arange(5, dtype=np.float64)
    t = sb.FloatTensor(np_arr)
    # start < stop with negative step → empty
    result = np.asarray(t[1:4:-1])
    expected = np_arr[1:4:-1]
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result, expected)


def test_negative_stride_copy():
    np_arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    t = sb.FloatTensor(np_arr)
    rev = t[::-1, ::-1]
    c = rev.copy()
    np.testing.assert_array_equal(np.asarray(c), np_arr[::-1, ::-1])
    # Mutating the copy must not affect the original
    c[0, 0] = -99.0
    assert t[0, 0] == np_arr[0, 0]


def test_negative_stride_arithmetic():
    np_arr = np.arange(6, dtype=np.float64)
    t = sb.FloatTensor(np_arr)
    rev = t[::-1]
    result = np.asarray(rev + sb.FloatTensor(np.ones(6)))
    np.testing.assert_array_equal(result, np_arr[::-1] + 1.0)


def test_negative_stride_scalar_arithmetic():
    np_arr = np.arange(6, dtype=np.float64)
    t = sb.FloatTensor(np_arr)
    rev = t[::-1]
    np.testing.assert_array_equal(np.asarray(rev * 2.0), np_arr[::-1] * 2.0)
    np.testing.assert_array_equal(np.asarray(rev + 10.0), np_arr[::-1] + 10.0)


def test_negative_stride_broadcast():
    np_arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    t = sb.FloatTensor(np_arr)
    rev = t[::-1, :]
    bias = sb.FloatTensor(np.array([10, 20, 30, 40], dtype=np.float64))
    result = np.asarray(rev + bias)
    np.testing.assert_array_equal(result, np_arr[::-1, :] + np.array([10, 20, 30, 40]))


def test_negative_stride_bool_mask():
    np_arr = np.arange(6, dtype=np.float32)
    t = sb.FloatTensor(np_arr)
    rev = t[::-1]
    mask = sb.BoolTensor(np.array([True, False, True, False, True, False]))
    result = np.asarray(rev[mask])
    expected = np_arr[::-1][np.array([True, False, True, False, True, False])]
    np.testing.assert_array_equal(result, expected)


def test_negative_stride_save_load(tmp_path):
    np_arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    t = sb.FloatTensor(np_arr)
    rev = t[::-1, ::-1]
    path = tmp_path / "rev.sb"
    sb.save(rev, str(path))
    loaded = sb.load(str(path))
    np.testing.assert_array_equal(np.asarray(loaded), np_arr[::-1, ::-1])


def test_negative_stride_then_slice():
    np_arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    t = sb.FloatTensor(np_arr)
    rev = t[::-1, :]
    result = np.asarray(rev[1:3, 2:4])
    np.testing.assert_array_equal(result, np_arr[::-1][1:3, 2:4])


def test_negative_stride_then_positive_step():
    np_arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    t = sb.FloatTensor(np_arr)
    rev = t[::-1, :]
    result = np.asarray(rev[::2, :])
    np.testing.assert_array_equal(result, np_arr[::-1][::2])


def test_double_reverse():
    np_arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    t = sb.FloatTensor(np_arr)
    result = np.asarray(t[::-1, :][::-1, :])
    np.testing.assert_array_equal(result, np_arr)


def test_negative_stride_then_negative_stride():
    np_arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    t = sb.FloatTensor(np_arr)
    result = np.asarray(t[::-1, :][::-1, ::-1])
    np.testing.assert_array_equal(result, np_arr[:, ::-1])


def test_negative_stride_comparison():
    np_arr = np.arange(5, dtype=np.float64)
    t = sb.FloatTensor(np_arr)
    rev = t[::-1]
    threshold = sb.FloatTensor(np.full(5, 2.0))
    result = np.asarray(rev > threshold)
    expected = np_arr[::-1] > 2.0
    np.testing.assert_array_equal(result, expected)
