"""Remaining indexing-parity tests: leading-axis boolean masks, len() of a
0-d tensor, scalar-result behavior, and cross-type uniformity.

Covers review findings: fancy-1d-rowmask-misrouted, len-0d-wrong-exception,
scalar-result-python-float, crosstype-uniform-shared-path.
"""

import numpy as np
import pytest

import stablebear as sb
from stablebear.base_tensor import BoolTensor, FloatTensor, IntTensor
from _indexing_support import assert_getitem_matches, assert_setitem_matches, ref_array



# =============================================================================
# Boolean mask whose shape covers the leading axes (sub-shape mask)
# =============================================================================


class TestLeadingAxisBoolMask:
    def test_1d_mask_selects_rows(self):
        # a[mask] with a length-4 mask on a (4, 6) array -> rows where True.
        assert_getitem_matches(ref_array(), np.array([True, False, True, False]))

    def test_1d_mask_all_false(self):
        assert_getitem_matches(ref_array(), np.array([False, False, False, False]))

    def test_1d_mask_on_3d_leading_axis(self):
        # b[mask] with a length-2 mask on a (2, 3, 4) array -> (n_true, 3, 4).
        B = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)
        assert_getitem_matches(B, np.array([True, False]))

    def test_2d_mask_on_leading_two_axes(self):
        # b[mask] with a (2, 3) mask on a (2, 3, 4) array -> (n_true, 4).
        mask = np.array([[True, False, True], [False, True, False]])
        B = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)
        assert_getitem_matches(B, mask)


# =============================================================================
# len() / iteration of a 0-d tensor
# =============================================================================


class TestZeroDimLen:
    def test_len_of_scalar_tensor_raises_typeerror(self):
        # NumPy: len(np.array(5.0)) -> TypeError 'len() of unsized object'.
        t = FloatTensor(np.array(5.0))
        with pytest.raises(TypeError):
            len(t)


# =============================================================================
# Scalar result of full integer indexing
# =============================================================================


class TestScalarResult:
    def test_scalar_value_matches_numpy(self):
        # The element value must match NumPy. (The exact result *type* -- a bare
        # Python float vs a 0-d numpy scalar -- is a separate, undecided policy
        # question and is intentionally not asserted here.)
        a = ref_array()
        t = FloatTensor(a)
        assert float(t[1, 2]) == float(a[1, 2])


# =============================================================================
# Cross-type uniformity: the shared __getitem__ behaves identically per type
# =============================================================================

_CROSS_TYPES = [
    pytest.param(FloatTensor, np.float64, id="float64"),
    pytest.param(IntTensor, np.int64, id="int64"),
    pytest.param(BoolTensor, np.bool_, id="bool"),
]


def _assert_type_getitem(TensorType, np_dtype, arr, index):
    a = np.asarray(arr).astype(np_dtype)
    expected = a[index]
    got = np.asarray(TensorType(a.copy())[index])
    assert got.shape == expected.shape, f"shape {got.shape} != numpy {expected.shape}"
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("TensorType, np_dtype", _CROSS_TYPES)
class TestCrossTypeNegativeIndex:
    def test_negative_int_last_row(self, TensorType, np_dtype):
        base = (np.arange(24).reshape(4, 6) % 2) if np_dtype is np.bool_ else np.arange(24).reshape(4, 6)
        _assert_type_getitem(TensorType, np_dtype, base, -1)

    def test_negative_slice_bounds(self, TensorType, np_dtype):
        base = (np.arange(24).reshape(4, 6) % 2) if np_dtype is np.bool_ else np.arange(24).reshape(4, 6)
        _assert_type_getitem(TensorType, np_dtype, base, slice(-3, -1))


# =============================================================================
# DistanceMatrix / SymmetricMatrix scalar (i, j) indexing
# =============================================================================


class TestMatrixWrapperNegativeIndexing:
    """The compressed matrix wrappers have their own ``__getitem__(i, j)``.

    Covers review finding distmat-symmat-scalar-getitem. They already raise
    IndexError on out-of-range indices (asserted as a contract guard), but
    negative indices are not resolved (they currently raise TypeError).
    """

    @pytest.mark.parametrize("Ctor", [sb.DistanceMatrix, sb.SymmetricMatrix],
                             ids=["distance", "symmetric"])
    def test_negative_row_resolves(self, Ctor):
        m = Ctor(4)
        m[3, 0] = 5.0
        assert m[-1, 0] == 5.0

    @pytest.mark.parametrize("Ctor", [sb.DistanceMatrix, sb.SymmetricMatrix],
                             ids=["distance", "symmetric"])
    def test_negative_col_resolves(self, Ctor):
        m = Ctor(4)
        m[3, 0] = 5.0
        assert m[0, -1] == 5.0

    @pytest.mark.parametrize("Ctor", [sb.DistanceMatrix, sb.SymmetricMatrix],
                             ids=["distance", "symmetric"])
    def test_out_of_bounds_raises_indexerror(self, Ctor):
        m = Ctor(4)
        with pytest.raises(IndexError):
            _ = m[4, 0]


# =============================================================================
# setitem right-hand-side handling: single-int row, scalar broadcast, dtype cast
# =============================================================================


class TestSetitemRhsHandling:
    def test_single_int_row_scalar(self):
        # t[1] = 7.0 broadcasts a scalar across row 1 (NumPy).
        assert_setitem_matches(ref_array(), 1, 7.0)

    def test_single_int_row_array(self):
        # t[1] = <row> assigns a whole row (NumPy).
        assert_setitem_matches(ref_array(), 1, np.arange(10.0, 16.0))

    def test_scalar_into_slice(self):
        # t[1:3] = 5.0 broadcasts a scalar across the slice (NumPy).
        assert_setitem_matches(ref_array(), slice(1, 3), 5.0)

    def test_cross_dtype_tensor_rhs_casts(self):
        # NumPy casts an integer RHS to the float destination dtype.
        a = ref_array()
        expected = a.copy()
        expected[1:3] = np.ones((2, 6), dtype=np.int64)
        t = FloatTensor(a)
        t[1:3] = IntTensor(np.ones((2, 6), dtype=np.int64))
        np.testing.assert_array_equal(np.asarray(t), expected)
