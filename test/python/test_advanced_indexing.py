import numpy as np
import pytest

import stablebear as sb
from stablebear.base_tensor import FloatTensor, IntTensor


def _sb(arr):
    """Convert a float32 numpy array to a FloatTensor."""
    return FloatTensor(np.asarray(arr, dtype=np.float32))


def _assert_advanced_select(arr, index):
    """Assert that stablebear advanced indexing matches NumPy."""
    arr = np.asarray(arr, dtype=np.float32)
    result = np.asarray(_sb(arr)[index])
    expected = arr[index]
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


def _assert_advanced_fill(arr, index, fill_value):
    """Assert that stablebear advanced scalar fill matches NumPy."""
    arr = np.asarray(arr, dtype=np.float32)
    np_arr = arr.copy()
    np_arr[index] = fill_value
    t = _sb(arr.copy())
    t[index] = fill_value
    np.testing.assert_array_equal(np.asarray(t), np_arr)
    assert t.shape == np_arr.shape


def _assert_advanced_assign(arr, index, values):
    """Assert that stablebear advanced tensor assign matches NumPy."""
    arr = np.asarray(arr, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    np_arr = arr.copy()
    np_arr[index] = values
    t = _sb(arr.copy())
    t[index] = _sb(values)
    np.testing.assert_array_equal(np.asarray(t), np_arr)
    assert t.shape == np_arr.shape


# =============================================================================
# __getitem__ with integer index arrays
# =============================================================================


class TestAdvancedGetitem:
    def test_1d_gather(self):
        _assert_advanced_select([10, 20, 30, 40, 50], np.array([2, 0, 4]))

    def test_1d_gather_duplicates(self):
        _assert_advanced_select([10, 20, 30], np.array([1, 1, 2, 0]))

    def test_2d_row_gather(self):
        _assert_advanced_select(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            np.array([2, 0]),
        )

    def test_2d_column_gather(self):
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        _assert_advanced_select(arr, (slice(None), np.array([1, 3])))

    def test_negative_indices(self):
        _assert_advanced_select([10, 20, 30, 40, 50], np.array([-1, -2]))

    def test_negative_indices_2d(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        _assert_advanced_select(arr, (slice(None), np.array([-1])))

    def test_single_index(self):
        _assert_advanced_select([10, 20, 30], np.array([1]))

    def test_all_indices(self):
        _assert_advanced_select([10, 20, 30], np.array([0, 1, 2]))

    def test_reversed_order(self):
        _assert_advanced_select([10, 20, 30, 40], np.array([3, 2, 1, 0]))

    def test_out_of_bounds_positive(self):
        t = _sb([10, 20, 30])
        with pytest.raises(IndexError):
            _ = t[np.array([3])]

    def test_out_of_bounds_negative(self):
        t = _sb([10, 20, 30])
        with pytest.raises(IndexError):
            _ = t[np.array([-4])]


# =============================================================================
# __setitem__ with integer index arrays — scalar fill
# =============================================================================


class TestAdvancedSetitemScalar:
    def test_1d_fill(self):
        _assert_advanced_fill([10, 20, 30, 40, 50], np.array([1, 3]), 0.0)

    def test_2d_row_fill(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_advanced_fill(arr, np.array([0, 2]), -1.0)

    def test_2d_col_fill(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        _assert_advanced_fill(arr, (slice(None), np.array([0, 2])), 0.0)


# =============================================================================
# __setitem__ with integer index arrays — tensor assign
# =============================================================================


class TestAdvancedSetitemTensor:
    def test_1d_assign(self):
        _assert_advanced_assign(
            [10, 20, 30, 40, 50],
            np.array([0, 2]),
            [99, 88],
        )

    def test_2d_row_assign(self):
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        _assert_advanced_assign(arr, np.array([0, 2]), [[10, 20], [50, 60]])


# =============================================================================
# Mixed integer index + slice indexing
# =============================================================================


class TestMixedIntSliceGetitem:
    def test_slice_then_int_index(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        idx = (slice(1, 3), np.array([0, 2]))
        result = np.asarray(_sb(arr)[idx])
        expected = arr[1:3][:, [0, 2]]
        np.testing.assert_array_equal(result, expected)
        assert result.shape == expected.shape

    def test_int_index_after_scalar(self):
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        idx = (0, np.array([0, 3]))
        result = np.asarray(_sb(arr)[idx])
        expected = arr[0, [0, 3]]
        np.testing.assert_array_equal(result, expected)
        assert result.shape == expected.shape


# =============================================================================
# IntTensor as index
# =============================================================================


class TestIntTensorAsIndex:
    def test_int_tensor_selects(self):
        t = _sb([10, 20, 30, 40, 50])
        idx = IntTensor(np.array([4, 1, 0]))
        result = np.asarray(t[idx])
        expected = np.array([50, 20, 10], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_advanced_index_on_int_tensor(self):
        arr = np.array([100, 200, 300, 400], dtype=np.int64)
        t = IntTensor(arr)
        idx = np.array([3, 1])
        result = np.asarray(t[idx])
        expected = arr[idx]
        np.testing.assert_array_equal(result, expected)
        assert isinstance(t[idx], IntTensor)


# =============================================================================
# Error cases
# =============================================================================


def _assert_outer_index(arr, index):
    """Assert that stablebear multi-index with outer semantics matches np.ix_."""
    arr = np.asarray(arr, dtype=np.float32)
    t = _sb(arr)
    result = np.asarray(t[index])
    # Build np.ix_ equivalent: convert slices to full ranges, pass bool/int arrays through
    ix_args = []
    for i, idx in enumerate(index if isinstance(index, tuple) else (index,)):
        if isinstance(idx, slice):
            ix_args.append(np.arange(*idx.indices(arr.shape[i])))
        elif hasattr(idx, 'dtype') and idx.dtype == np.bool_:
            ix_args.append(idx)
        else:
            ix_args.append(np.asarray(idx))
    expected = arr[np.ix_(*ix_args)]
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


class TestMultipleIntIndices:
    def test_2d_outer(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_outer_index(arr, (np.array([0, 2]), np.array([1, 3])))

    def test_3d_outer(self):
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        _assert_outer_index(arr, (np.array([0]), slice(None), np.array([1, 3])))

    def test_2d_with_slice(self):
        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        _assert_outer_index(arr, (np.array([1, 3]), slice(1, 4)))


class TestMixedBoolIntIndices:
    def test_bool_rows_int_cols(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_outer_index(arr, (np.array([True, False, True]), np.array([0, 3])))

    def test_int_rows_bool_cols(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_outer_index(arr, (np.array([0, 2]), np.array([False, True, True, False])))

    def test_3d_mixed(self):
        arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        _assert_outer_index(arr, (np.array([True, False, True]), slice(None), np.array([0, 2, 4])))


# =============================================================================
# Setitem with multiple / mixed indices
# =============================================================================


def _assert_outer_setitem_scalar(arr, index, fill_value):
    """Assert that stablebear outer setitem with scalar matches np.ix_."""
    arr = np.asarray(arr, dtype=np.float32)
    t = _sb(arr.copy())
    np_arr = arr.copy()
    t[index] = fill_value
    ix_args = []
    for i, idx in enumerate(index if isinstance(index, tuple) else (index,)):
        if isinstance(idx, slice):
            ix_args.append(np.arange(*idx.indices(arr.shape[i])))
        elif hasattr(idx, 'dtype') and idx.dtype == np.bool_:
            ix_args.append(idx)
        else:
            ix_args.append(np.asarray(idx))
    np_arr[np.ix_(*ix_args)] = fill_value
    np.testing.assert_array_equal(np.asarray(t), np_arr)


class TestOuterSetitem:
    def test_multi_int_scalar_fill(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_outer_setitem_scalar(arr, (np.array([0, 2]), np.array([1, 3])), -1.0)

    def test_mixed_bool_int_scalar_fill(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_outer_setitem_scalar(
            arr, (np.array([True, False, True]), np.array([0, 3])), -1.0)

    def test_3d_mixed_scalar_fill(self):
        arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        _assert_outer_setitem_scalar(
            arr, (np.array([True, False, True]), slice(None), np.array([0, 2, 4])), 0.0)
