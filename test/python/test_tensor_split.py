import numpy as np
import pytest

import stablebear as sb


_NUMERIC_TYPES = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
    pytest.param(sb.IntTensor, np.int32, id="int32"),
    pytest.param(sb.IntTensor, np.int64, id="int64"),
]


def _assert_split(np_arr, indices_or_sections, axis, TensorType):
    """Assert that sb.split matches np.split."""
    t = TensorType(np_arr)
    sb_parts = sb.split(t, indices_or_sections, axis=axis)
    np_parts = np.split(np_arr, indices_or_sections, axis=axis)
    assert len(sb_parts) == len(np_parts)
    for sb_part, np_part in zip(sb_parts, np_parts):
        np.testing.assert_array_equal(np.asarray(sb_part), np_part)
        assert sb_part.shape == tuple(np_part.shape)


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestSplitEqual:
    def test_1d_into_3(self, TensorType, np_dtype):
        _assert_split(np.arange(9, dtype=np_dtype), 3, 0, TensorType)

    def test_2d_axis0(self, TensorType, np_dtype):
        _assert_split(np.arange(12, dtype=np_dtype).reshape(4, 3), 2, 0, TensorType)

    def test_2d_axis1(self, TensorType, np_dtype):
        _assert_split(np.arange(12, dtype=np_dtype).reshape(3, 4), 2, 1, TensorType)

    def test_3d(self, TensorType, np_dtype):
        _assert_split(np.arange(24, dtype=np_dtype).reshape(2, 6, 2), 3, 1, TensorType)

    def test_uneven_raises(self, TensorType, np_dtype):
        t = TensorType(np.arange(7, dtype=np_dtype))
        with pytest.raises((ValueError, RuntimeError)):
            sb.split(t, 3, axis=0)


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestSplitIndices:
    def test_1d(self, TensorType, np_dtype):
        _assert_split(np.arange(10, dtype=np_dtype), [3, 7], 0, TensorType)

    def test_2d_axis0(self, TensorType, np_dtype):
        _assert_split(np.arange(20, dtype=np_dtype).reshape(5, 4), [1, 3], 0, TensorType)

    def test_2d_axis1(self, TensorType, np_dtype):
        _assert_split(np.arange(12, dtype=np_dtype).reshape(3, 4), [1, 3], 1, TensorType)

    def test_empty_indices(self, TensorType, np_dtype):
        """Empty indices list returns the whole tensor as a single part."""
        _assert_split(np.arange(6, dtype=np_dtype), [], 0, TensorType)

    def test_is_view(self, TensorType, np_dtype):
        np_arr = np.arange(6, dtype=np_dtype)
        t = TensorType(np_arr)
        parts = sb.split(t, [3], axis=0)
        parts[0][0] = 99
        assert t[0] == 99


def _assert_array_split(np_arr, n_sections, axis, TensorType):
    """Assert that sb.array_split matches np.array_split."""
    t = TensorType(np_arr)
    sb_parts = sb.array_split(t, n_sections, axis=axis)
    np_parts = np.array_split(np_arr, n_sections, axis=axis)
    assert len(sb_parts) == len(np_parts)
    for sb_part, np_part in zip(sb_parts, np_parts):
        np.testing.assert_array_equal(np.asarray(sb_part), np_part)
        assert sb_part.shape == tuple(np_part.shape)


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestArraySplit:
    def test_even(self, TensorType, np_dtype):
        _assert_array_split(np.arange(12, dtype=np_dtype), 3, 0, TensorType)

    def test_uneven(self, TensorType, np_dtype):
        _assert_array_split(np.arange(9, dtype=np_dtype), 4, 0, TensorType)

    def test_more_sections_than_elements(self, TensorType, np_dtype):
        _assert_array_split(np.arange(3, dtype=np_dtype), 5, 0, TensorType)

    def test_2d_axis0(self, TensorType, np_dtype):
        _assert_array_split(np.arange(15, dtype=np_dtype).reshape(5, 3), 3, 0, TensorType)

    def test_2d_axis1(self, TensorType, np_dtype):
        _assert_array_split(np.arange(15, dtype=np_dtype).reshape(3, 5), 3, 1, TensorType)

    def test_single_section(self, TensorType, np_dtype):
        _assert_array_split(np.arange(7, dtype=np_dtype), 1, 0, TensorType)
