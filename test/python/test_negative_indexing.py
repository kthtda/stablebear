"""NumPy-parity tests for negative indices and negative slice bounds.

Covers review findings: neg-int-multidim-squeeze, neg-int-1d-getelement,
neg-int-in-int-tuple, slicerange-neg-bounds-posstep, slicerange-neg-bounds-negstep,
slicerange-step-zero, setitem-negative-int, setitem-single-int-multidim,
setitem-negative-slice-bounds.

These assert the correct NumPy behavior. They currently FAIL (negative basic
indices are not resolved against the axis size; negative slice bounds are
clamped to 0 rather than resolved) and will pass once that is fixed.

The negative/out-of-bounds *int combined with a slice* cases are memory-unsafe
(they build a view at an invalid data offset) and live in
test_indexing_memory_safety.py instead.
"""

import numpy as np
import pytest

from stablebear.base_tensor import FloatTensor
from _indexing_support import assert_getitem_matches, assert_setitem_matches, ref_array


# =============================================================================
# getitem: single negative int
# =============================================================================


class TestNegativeIntGetitem:
    def test_last_row_2d(self):
        assert_getitem_matches(ref_array(), -1)

    def test_second_to_last_row_2d(self):
        assert_getitem_matches(ref_array(), -2)

    def test_first_via_negative_2d(self):
        assert_getitem_matches(ref_array(), -4)

    def test_last_element_1d(self):
        assert_getitem_matches(np.arange(6.0), -1)

    def test_third_to_last_element_1d(self):
        assert_getitem_matches(np.arange(6.0), -3)

    def test_first_via_negative_1d(self):
        assert_getitem_matches(np.arange(6.0), -6)

    def test_last_row_3d(self):
        assert_getitem_matches(np.arange(24.0).reshape(2, 3, 4), -1)


# =============================================================================
# getitem: negative int inside an all-int tuple
# =============================================================================


class TestNegativeIntInTuple:
    def test_pos_row_neg_col(self):
        assert_getitem_matches(ref_array(), (1, -1))

    def test_neg_row_neg_col(self):
        assert_getitem_matches(ref_array(), (-1, -1))

    def test_neg_row_pos_col(self):
        assert_getitem_matches(ref_array(), (-1, 0))

    def test_3d_all_negative(self):
        assert_getitem_matches(np.arange(24.0).reshape(2, 3, 4), (-1, -1, -1))


# =============================================================================
# getitem: negative slice bounds, positive step
# =============================================================================


class TestNegativeSliceBoundsPositiveStep:
    def test_both_negative(self):
        assert_getitem_matches(ref_array(), slice(-3, -1))

    def test_open_start_negative(self):
        assert_getitem_matches(ref_array(), slice(-2, None))

    def test_open_stop_negative(self):
        assert_getitem_matches(ref_array(), slice(None, -1))

    def test_pos_start_neg_stop(self):
        assert_getitem_matches(ref_array(), slice(1, -1))

    def test_negative_on_column_axis(self):
        assert_getitem_matches(ref_array(), (slice(None), slice(-2, None)))

    def test_negative_both_on_column_axis(self):
        assert_getitem_matches(ref_array(), (slice(None), slice(-3, -1)))

    def test_negative_with_step(self):
        assert_getitem_matches(np.arange(12.0), slice(-9, -1, 2))


# =============================================================================
# getitem: negative slice bounds, negative step
# =============================================================================


class TestNegativeSliceBoundsNegativeStep:
    def test_neg_start_neg_stop(self):
        assert_getitem_matches(ref_array(), slice(-1, -4, -1))

    def test_neg_start_pos_stop(self):
        assert_getitem_matches(ref_array(), slice(-1, 0, -1))

    def test_1d_neg_bounds(self):
        assert_getitem_matches(np.arange(6.0), slice(-1, -4, -1))


# =============================================================================
# getitem: step == 0 must raise (NumPy: ValueError 'slice step cannot be zero')
# =============================================================================


class TestStepZero:
    def test_step_zero_raises(self):
        t = FloatTensor(ref_array())
        with pytest.raises(ValueError):
            _ = t[::0]


# =============================================================================
# getitem: out-of-bounds single int must raise IndexError (not ValueError)
# =============================================================================


class TestOutOfBoundsInt:
    """NumPy raises IndexError for an out-of-range single int on any axis.

    Covers review finding oob-single-int-wrong-exception: stablebear currently
    raises ValueError (from squeeze) for an N-D out-of-bounds row and TypeError
    for an out-of-range negative on a 1-D tensor.
    """

    def test_oob_row_2d_raises(self):
        t = FloatTensor(ref_array())
        with pytest.raises(IndexError):
            _ = t[4]

    def test_large_oob_row_2d_raises(self):
        t = FloatTensor(ref_array())
        with pytest.raises(IndexError):
            _ = t[10]

    def test_too_negative_row_2d_raises(self):
        t = FloatTensor(ref_array())
        with pytest.raises(IndexError):
            _ = t[-5]

    def test_oob_1d_raises(self):
        t = FloatTensor(np.arange(6.0))
        with pytest.raises(IndexError):
            _ = t[6]

    def test_too_negative_1d_raises(self):
        t = FloatTensor(np.arange(6.0))
        with pytest.raises(IndexError):
            _ = t[-7]


# =============================================================================
# setitem: single negative int
# =============================================================================


class TestNegativeIntSetitem:
    def test_set_last_row_scalar(self):
        assert_setitem_matches(ref_array(), -1, 7.0)

    def test_set_second_to_last_row_scalar(self):
        assert_setitem_matches(ref_array(), -2, -3.0)

    def test_set_last_element_1d(self):
        assert_setitem_matches(np.arange(6.0), -1, 99.0)

    def test_set_last_row_from_array(self):
        assert_setitem_matches(ref_array(), -1, np.arange(100.0, 106.0))


# =============================================================================
# setitem: negative int inside an all-int tuple
# =============================================================================


class TestNegativeIntInTupleSetitem:
    def test_pos_row_neg_col(self):
        assert_setitem_matches(ref_array(), (1, -1), 5.0)

    def test_neg_row_neg_col(self):
        assert_setitem_matches(ref_array(), (-1, -1), 5.0)


# =============================================================================
# setitem: negative slice bounds
# =============================================================================


class TestNegativeSliceBoundsSetitem:
    def test_set_both_negative_block(self):
        assert_setitem_matches(ref_array(), slice(-3, -1), np.zeros((2, 6)))

    def test_set_open_start_negative_block(self):
        assert_setitem_matches(ref_array(), slice(-2, None), np.full((2, 6), 8.0))
