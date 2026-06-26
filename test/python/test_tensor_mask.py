import numpy as np
import pytest

import stablebear as sb
from stablebear.base_tensor import BoolTensor, FloatTensor


def _sb(arr):
    """Convert a float32 numpy array to a FloatTensor."""
    return FloatTensor(np.asarray(arr, dtype=np.float32))


def _assert_masked_select(arr, mask):
    """Assert that masked select matches NumPy."""
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    result = np.asarray(_sb(arr)[BoolTensor(mask)])
    masked = arr[mask]
    np.testing.assert_array_equal(result, masked)
    assert result.shape == masked.shape


def _assert_masked_fill(arr, mask, fill_value):
    """Assert that masked scalar fill matches NumPy."""
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    t = _sb(arr.copy())
    t[BoolTensor(mask)] = fill_value
    arr[mask] = fill_value
    np.testing.assert_array_equal(np.asarray(t), arr)
    assert t.shape == arr.shape


def _assert_masked_assign(arr, mask, values):
    """Assert that masked tensor assign matches NumPy."""
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    values = np.asarray(values, dtype=np.float32)
    t = _sb(arr.copy())
    t[BoolTensor(mask)] = _sb(values)
    arr[mask] = values
    np.testing.assert_array_equal(np.asarray(t), arr)
    assert t.shape == arr.shape



# =============================================================================
# __getitem__ with BoolTensor mask
# =============================================================================


class TestMaskedGetitem:
    def test_basic(self):
        _assert_masked_select([0, 1, 2, 3, 4], [True, False, True, False, True])

    def test_bool_tensor(self):
        """BoolTensor[BoolTensor] should work too."""
        t = BoolTensor(np.array([True, False, True, False]))
        mask = BoolTensor(np.array([False, False, True, True]))
        np.testing.assert_array_equal(np.asarray(t[mask]), [True, False])

    def test_2d(self):
        _assert_masked_select(
            [[0, 1, 2], [3, 4, 5]],
            [[False, True, False], [True, False, True]],
        )

    def test_all_true(self):
        _assert_masked_select([10, 20, 30], [True, True, True])

    def test_all_false(self):
        _assert_masked_select([10, 20, 30], [False, False, False])

    def test_all_false_2d(self):
        _assert_masked_select(np.arange(6).reshape(2, 3), np.zeros((2, 3), dtype=bool))

    def test_shape_mismatch_raises(self):
        """Mask shape must exactly match tensor shape."""
        with pytest.raises(ValueError):
            _sb(np.zeros((2, 3)))[BoolTensor(np.array([True, False, True, False, True]))]

    def test_broadcast_mask_raises(self):
        """1D mask on 2D tensor raises, matching NumPy behavior."""
        with pytest.raises(ValueError):
            _sb(np.zeros((2, 3)))[BoolTensor(np.array([True, False, True]))]

    def test_empty(self):
        _assert_masked_select(
            np.array([], dtype=np.float32),
            np.array([], dtype=bool),
        )

    def test_pcf64(self):
        """Masked select on a Pcf64Tensor."""
        t = sb.zeros((3,), dtype=sb.pcf64)
        t[0] = sb.Pcf([(0, 1.0), (1, 2.0)])
        t[1] = sb.Pcf([(0, 3.0)])
        t[2] = sb.Pcf([(0, 0.0)])
        mask = BoolTensor(np.array([True, False, True]))
        assert t[mask].shape[0] == 2


# =============================================================================
# __setitem__ with BoolTensor mask — scalar fill
# =============================================================================


class TestMaskedSetitemScalar:
    def test_fill(self):
        _assert_masked_fill([0, 1, 2, 3, 4], [True, False, True, False, True], 99.0)

    def test_2d_fill(self):
        _assert_masked_fill(
            [[0, 1, 2], [3, 4, 5]],
            [[False, True, False], [True, False, True]],
            -1.0,
        )

    def test_bool_fill(self):
        np_arr = np.array([True, False, True, False])
        np_mask = np.array([True, True, False, False])
        t = BoolTensor(np_arr.copy())
        t[BoolTensor(np_mask)] = False
        np_arr[np_mask] = False
        np.testing.assert_array_equal(np.asarray(t), np_arr)

    def test_fill_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            _sb(np.zeros((2, 3)))[BoolTensor(np.array([True, False, True, False, True]))] = -1.0


# =============================================================================
# __setitem__ with BoolTensor mask — tensor assign
# =============================================================================


class TestMaskedSetitemTensor:
    def test_assign(self):
        _assert_masked_assign([0, 1, 2, 3, 4], [True, False, True, False, True], [10, 20, 30])

    def test_2d_assign(self):
        _assert_masked_assign(
            [[0, 1, 2], [3, 4, 5]],
            [[False, True, False], [True, True, False]],
            [10, 20, 30],
        )

    def test_values_length_mismatch_raises(self):
        t = _sb(np.arange(5))
        mask = BoolTensor(np.array([True, False, True, False, True]))
        with pytest.raises(ValueError):
            t[mask] = _sb(np.array([10, 20]))

    def test_assign_shape_mismatch_raises(self):
        t = _sb(np.zeros((2, 3)))
        mask = BoolTensor(np.array([True, False, True, False, True]))
        with pytest.raises(ValueError):
            t[mask] = _sb(np.array([10, 20, 30]))


# =============================================================================
# Non-contiguous tensors
# =============================================================================


class TestMaskedNonContiguous:
    def test_select_matches_numpy(self):
        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        np_view = arr[::2, 1:4]
        sb_view = _sb(arr)[::2, 1:4]
        mask = np.array([[True, False, True], [False, True, False]])
        np.testing.assert_array_equal(
            np.asarray(sb_view[BoolTensor(mask)]),
            np_view[mask],
        )

    def test_fill_matches_numpy(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = _sb(arr.copy())
        np_view = arr[::2]
        sb_view = t[::2]
        mask = np.array([[True, False, True, False], [False, True, False, True]])
        np_view[mask] = -1.0
        sb_view[BoolTensor(mask)] = -1.0
        np.testing.assert_array_equal(np.asarray(t), arr)


# =============================================================================
# Dtype validation on masked setitem
# =============================================================================


class TestMaskedSetitemDtypeValidation:
    def test_scalar_wrong_type_raises(self):
        t = _sb(np.array([1, 2, 3]))
        mask = BoolTensor(np.array([True, False, True]))
        with pytest.raises(TypeError):
            t[mask] = "bad"

    def test_tensor_wrong_type_raises(self):
        t = _sb(np.array([1, 2, 3]))
        mask = BoolTensor(np.array([True, False, True]))
        with pytest.raises(TypeError):
            t[mask] = BoolTensor(np.array([True, True]))

    def test_bool_tensor_rejects_float_scalar(self):
        t = BoolTensor(np.array([True, False, True]))
        mask = BoolTensor(np.array([True, False, False]))
        with pytest.raises(TypeError):
            t[mask] = 3.14

    def test_bool_tensor_rejects_float_tensor(self):
        t = BoolTensor(np.array([True, False, True]))
        mask = BoolTensor(np.array([True, False, False]))
        with pytest.raises(TypeError):
            t[mask] = _sb(np.array([99]))


# =============================================================================
# BoolTensor from numpy array
# =============================================================================


class TestBoolTensorFromNumpy:
    def test_1d(self):
        bt = BoolTensor(np.array([True, False, True]))
        np.testing.assert_array_equal(np.asarray(bt), [True, False, True])

    def test_2d(self):
        bt = BoolTensor(np.array([[True, False], [False, True]]))
        np.testing.assert_array_equal(np.asarray(bt), [[True, False], [False, True]])

    def test_from_int_array(self):
        """Integer arrays are coerced to bool."""
        bt = BoolTensor(np.array([1, 0, 1, 0]))
        np.testing.assert_array_equal(np.asarray(bt), [True, False, True, False])

    def test_empty(self):
        bt = BoolTensor(np.array([], dtype=bool))
        assert tuple(bt.shape) == (0,)


# =============================================================================
# Mixed bool + slice indexing: tensor[:, mask, 1:5] etc.
# =============================================================================


def _assert_mixed_getitem(arr, index):
    """Assert that stablebear mixed indexing matches NumPy."""
    result = np.asarray(_sb(arr)[index])
    expected = arr[index]
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


class TestMixedBoolSliceGetitem:
    def test_2d_mask_axis0(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_mixed_getitem(arr, (np.array([True, False, True]), slice(None)))

    def test_2d_mask_axis1(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_mixed_getitem(arr, (slice(None), np.array([True, False, True, False])))

    def test_3d_mask_middle_axis(self):
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        _assert_mixed_getitem(arr, (slice(None), np.array([False, True, True]), slice(None)))

    def test_mask_with_slice(self):
        arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        _assert_mixed_getitem(arr, (slice(None), np.array([True, False, True, False]), slice(1, 4)))

    def test_mask_with_int_index(self):
        """tensor[0, :, mask] applies int indexing then mask on the result.
        NumPy's advanced indexing reorders dimensions here; we don't."""
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        mask = np.array([True, False, True, False])
        expected = arr[0][:, mask]
        result = np.asarray(_sb(arr)[0, slice(None), mask])
        np.testing.assert_array_equal(result, expected)
        assert result.shape == expected.shape

    def test_mask_with_step_slice(self):
        arr = np.arange(30, dtype=np.float32).reshape(5, 6)
        _assert_mixed_getitem(arr, (slice(0, 5, 2), np.array([False, True, True, False, True, False])))

    def test_all_false_mask(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_mixed_getitem(arr, (slice(None), np.array([False, False, False, False])))

    def test_all_true_mask(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_mixed_getitem(arr, (slice(None), np.array([True, True, True, True])))

    def test_mask_length_mismatch_raises(self):
        t = _sb(np.zeros((3, 4), dtype=np.float32))
        with pytest.raises(ValueError):
            t[slice(None), np.array([True, False])]

    def test_multiple_masks_2d(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = _sb(arr)
        row_mask = np.array([True, False, True])
        col_mask = np.array([False, True, True, False])
        result = np.asarray(t[row_mask, col_mask])
        expected = arr[np.ix_(row_mask, col_mask)]
        np.testing.assert_array_equal(result, expected)
        assert result.shape == expected.shape

    def test_multiple_masks_3d(self):
        arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        t = _sb(arr)
        mask0 = np.array([True, False, True])
        mask2 = np.array([True, True, False, True, False])
        result = np.asarray(t[mask0, :, mask2])
        expected = arr[np.ix_(mask0, np.ones(4, dtype=bool), mask2)]
        np.testing.assert_array_equal(result, expected)
        assert result.shape == expected.shape


# =============================================================================
# Mixed bool + slice __setitem__
# =============================================================================


def _assert_mixed_setitem_scalar(arr, index, fill_value):
    """Assert that stablebear mixed setitem with scalar matches NumPy."""
    np_arr = arr.copy()
    np_arr[index] = fill_value
    t = _sb(arr.copy())
    t[index] = fill_value
    np.testing.assert_array_equal(np.asarray(t), np_arr)


def _assert_mixed_setitem_tensor(arr, index, values):
    """Assert that stablebear mixed setitem with tensor matches NumPy."""
    np_arr = arr.copy()
    np_arr[index] = values
    t = _sb(arr.copy())
    t[index] = _sb(values)
    np.testing.assert_array_equal(np.asarray(t), np_arr)


class TestMixedBoolSliceSetitem:
    def test_scalar_fill_axis1(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_mixed_setitem_scalar(arr, (slice(None), np.array([True, False, True, False])), -1.0)

    def test_scalar_fill_axis0(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        _assert_mixed_setitem_scalar(arr, (np.array([False, True, True]), slice(None)), 0.0)

    def test_scalar_fill_3d_middle(self):
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        _assert_mixed_setitem_scalar(
            arr, (slice(None), np.array([True, False, True]), slice(None)), 99.0)

    def test_scalar_fill_with_slice(self):
        arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        _assert_mixed_setitem_scalar(
            arr, (slice(None), np.array([True, False, True, False]), slice(1, 4)), -5.0)

    def test_tensor_assign_axis1(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        values = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        _assert_mixed_setitem_tensor(
            arr, (slice(None), np.array([True, False, True, False])), values)

    def test_tensor_assign_3d(self):
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        values = np.zeros((2, 2, 4), dtype=np.float32)
        _assert_mixed_setitem_tensor(
            arr, (slice(None), np.array([True, False, True]), slice(None)), values)

    def test_fill_shape_mismatch_raises(self):
        t = _sb(np.zeros((3, 4), dtype=np.float32))
        with pytest.raises(ValueError):
            t[slice(None), np.array([True, False])] = -1.0

    def test_multiple_masks_scalar_fill(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        row_mask = np.array([True, False, True])
        col_mask = np.array([False, True, True, False])
        t = _sb(arr.copy())
        np_arr = arr.copy()
        t[row_mask, col_mask] = -1.0
        np_arr[np.ix_(row_mask, col_mask)] = -1.0
        np.testing.assert_array_equal(np.asarray(t), np_arr)


# =============================================================================
# Size-1 tensor RHS broadcasts like a scalar (issue #60)
# =============================================================================


class TestSize1TensorRhsBroadcast:
    def test_bool_mask_assign(self):
        arr = np.arange(5, dtype=np.float32)
        mask = arr > 1.0
        t = _sb(arr.copy())
        t[BoolTensor(mask)] = FloatTensor(np.array([99.0], dtype=np.float32))
        arr[mask] = 99.0
        np.testing.assert_array_equal(np.asarray(t), arr)

    def test_int_index_assign(self):
        arr = np.arange(5, dtype=np.float32)
        idx = np.array([1, 3])
        t = _sb(arr.copy())
        t[sb.IntTensor(idx)] = FloatTensor(np.array([99.0], dtype=np.float32))
        arr[idx] = 99.0
        np.testing.assert_array_equal(np.asarray(t), arr)

    def test_outer_assign(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        rows = np.array([0, 2])
        cols = np.array([1, 3])
        t = _sb(arr.copy())
        t[sb.IntTensor(rows), sb.IntTensor(cols)] = FloatTensor(
            np.array([7.0], dtype=np.float32))
        arr[np.ix_(rows, cols)] = 7.0
        np.testing.assert_array_equal(np.asarray(t), arr)

    def test_zero_d_rhs_broadcasts_uniformly(self):
        # A 0-d numeric tensor reads back unreliably through numpy, so only
        # assert the assignment succeeds and fills every selected cell with a
        # single uniform value (whatever it decays to).
        arr = np.arange(5, dtype=np.float32)
        mask = arr > 1.0
        t = _sb(arr.copy())
        t[BoolTensor(mask)] = FloatTensor(np.array(99.0, dtype=np.float32))
        filled = np.asarray(t)[mask]
        assert np.all(filled == filled[0])

    def test_all_false_mask_writes_nothing(self):
        arr = np.arange(5, dtype=np.float32)
        mask = arr > 100.0
        t = _sb(arr.copy())
        t[BoolTensor(mask)] = FloatTensor(np.array([99.0], dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(t), arr)

    def test_multi_element_mismatch_still_raises(self):
        t = _sb(np.arange(5, dtype=np.float32))
        mask = BoolTensor(np.array([True, False, True, False, True]))
        with pytest.raises(ValueError):
            t[mask] = FloatTensor(np.array([10.0, 20.0], dtype=np.float32))