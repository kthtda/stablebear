#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Memory-safety tests for indexing paths that once dereferenced an invalid offset.

Covers review findings: neg-sliceindex-oob-write-segv, sliceindex-positive-oob-no-check,
outer-assign-no-shape-validation, multi-axis-assign-no-shape-validation.

A negative or out-of-bounds integer combined with a slice (e.g. ``t[-1, 2:4]``,
``t[:, -1]``, ``t[5, 2:4]``) used to build a tensor view at an invalid data
offset because ``extract()`` resolved neither negative indices nor bounds;
reading returned garbage and writing corrupted memory or raised SIGSEGV. The
multi-axis assign kernels likewise omitted a values-shape check. Those bugs are
fixed, so these cases now run safely in-process and assert NumPy parity.
"""

import numpy as np
import pytest

from masspcf.tensor import FloatTensor, BoolTensor
from _indexing_support import assert_getitem_matches, assert_setitem_matches, ref_array


# =============================================================================
# Reads through a (formerly invalid) offset must return the correct elements
# =============================================================================


class TestUnsafeReads:
    def test_neg_row_then_col_slice(self):
        # t[-1, 2:4] -> last row, columns 2:4 == [20., 21.]
        assert_getitem_matches(ref_array(), (-1, slice(2, 4)))

    def test_neg_row_full(self):
        # t[-1, :] -> last row (mixed int+slice path, offset went negative)
        assert_getitem_matches(ref_array(), (-1, slice(None)))

    def test_last_column(self):
        # t[:, -1] -> last column [5., 11., 17., 23.]
        assert_getitem_matches(ref_array(), (slice(None), -1))

    def test_row_slice_then_neg_col(self):
        # t[2:4, -1] -> [17., 23.]
        assert_getitem_matches(ref_array(), (slice(2, 4), -1))

    def test_oob_row_then_col_slice_raises(self):
        # t[5, 2:4] on a 4-row tensor -> NumPy raises IndexError (no OOB read).
        t = FloatTensor(ref_array())
        with pytest.raises(IndexError):
            _ = t[5, 2:4]


# =============================================================================
# Writes through a (formerly invalid) offset must mutate the right cells (or raise)
# =============================================================================


class TestUnsafeWrites:
    def test_set_neg_row_col_slice(self):
        # t[-1, 2:4] = [100, 200]  (the confirmed SIGSEGV-on-write case)
        assert_setitem_matches(ref_array(), (-1, slice(2, 4)), np.array([100., 200.]))

    def test_set_last_column(self):
        # t[:, -1] = [10, 20, 30, 40]
        assert_setitem_matches(
            ref_array(), (slice(None), -1), np.array([10., 20., 30., 40.])
        )

    def test_set_oob_row_col_slice_raises(self):
        # t[5, 2:4] = ...  -> NumPy raises IndexError (no OOB write).
        t = FloatTensor(ref_array())
        with pytest.raises(IndexError):
            t[5, 2:4] = FloatTensor(np.array([0., 0.]))


# =============================================================================
# Multi-axis assignment must validate the values shape (no OOB write)
# =============================================================================


class TestMultiAxisAssignShapeValidation:
    def test_outer_assign_wrong_shape_raises(self):
        # t[[0,2],[1,3]] selects a (2,2) block (outer / np.ix_ semantics);
        # assigning a (3,3) tensor must raise, not write out of bounds.
        t = FloatTensor(ref_array())
        with pytest.raises((ValueError, RuntimeError)):
            t[np.array([0, 2]), np.array([1, 3])] = FloatTensor(np.full((3, 3), 9.0))

    def test_bool_mask_multi_axis_assign_wrong_shape_raises(self):
        # t[rowmask, colmask] selects a (2, 2) block (outer / np.ix_ semantics);
        # assigning a (3, 3) tensor must raise rather than write out of bounds.
        t = FloatTensor(ref_array())
        rows = BoolTensor(np.array([True, False, True, False]))               # 2 rows
        cols = BoolTensor(np.array([True, False, True, False, False, False]))  # 2 cols
        with pytest.raises((ValueError, RuntimeError)):
            t[rows, cols] = FloatTensor(np.full((3, 3), -9.0))


# =============================================================================
# Out-of-bounds multi-axis fancy index must raise (no OOB read)
# =============================================================================


class TestMultiAxisFancyBounds:
    def test_oob_multi_axis_fancy_index_raises(self):
        # t[rows, cols] with an out-of-range row index must raise IndexError
        # rather than read out of bounds (whatever kernel ends up backing it).
        t = FloatTensor(ref_array())
        with pytest.raises(IndexError):
            _ = t[np.array([0, 9]), np.array([0, 1])]   # 9 is OOB on axis 0 (size 4)
