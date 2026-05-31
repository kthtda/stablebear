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

"""Memory-safety tests for indexing paths that dereference an invalid offset.

Covers review findings: neg-sliceindex-oob-write-segv, sliceindex-positive-oob-no-check,
outer-assign-no-shape-validation, multi-axis-assign-no-shape-validation.

A negative or out-of-bounds integer combined with a slice (e.g. ``t[-1, 2:4]``,
``t[:, -1]``, ``t[5, 2:4]``) builds a tensor view at an invalid data offset
because ``extract()`` resolves neither negative indices nor bounds. Reading it
returns garbage; writing through it corrupts memory or raises SIGSEGV. The
multi-axis assign kernels likewise omit a values-shape check and can write out
of bounds.

Because such operations can crash the interpreter (and the crash is
nondeterministic — sometimes SIGSEGV, sometimes silent corruption), each test
runs in an isolated subprocess that performs the correctness check internally;
the parent only asserts the child exited cleanly. The snippets assert the
correct NumPy behavior, so they fail until the underlying bugs are fixed.

See _indexing_support.assert_isolated_ok for the isolation mechanism.
"""

from _indexing_support import assert_isolated_ok


# =============================================================================
# Reads through an invalid offset must return the correct elements
# =============================================================================


class TestUnsafeReads:
    def test_neg_row_then_col_slice(self):
        # t[-1, 2:4] -> last row, columns 2:4 == [20., 21.]
        assert_isolated_ok(
            """
            got = np.asarray(F()[-1, 2:4])
            assert got.tolist() == a[-1, 2:4].tolist(), got
            """
        )

    def test_neg_row_full(self):
        # t[-1, :] -> last row (mixed int+slice path, offset goes negative)
        assert_isolated_ok(
            """
            got = np.asarray(F()[-1, :])
            assert got.tolist() == a[-1, :].tolist(), got
            """
        )

    def test_last_column(self):
        # t[:, -1] -> last column [5., 11., 17., 23.]
        assert_isolated_ok(
            """
            got = np.asarray(F()[:, -1])
            assert got.tolist() == a[:, -1].tolist(), got
            """
        )

    def test_row_slice_then_neg_col(self):
        # t[2:4, -1] -> [17., 23.]
        assert_isolated_ok(
            """
            got = np.asarray(F()[2:4, -1])
            assert got.tolist() == a[2:4, -1].tolist(), got
            """
        )

    def test_oob_row_then_col_slice_raises(self):
        # t[5, 2:4] on a 4-row tensor -> NumPy raises IndexError (no OOB read).
        assert_isolated_ok(
            """
            try:
                _ = np.asarray(F()[5, 2:4])
            except IndexError:
                sys.exit(0)
            sys.exit("expected IndexError for out-of-bounds row")
            """
        )


# =============================================================================
# Writes through an invalid offset must mutate the right cells (or raise)
# =============================================================================


class TestUnsafeWrites:
    def test_set_neg_row_col_slice(self):
        # t[-1, 2:4] = [100, 200]  (the confirmed SIGSEGV-on-write case)
        assert_isolated_ok(
            """
            d = F()
            d[-1, 2:4] = FloatTensor(np.array([100., 200.]))
            expected = a.copy(); expected[-1, 2:4] = [100., 200.]
            assert np.asarray(d).tolist() == expected.tolist(), np.asarray(d)[-1].tolist()
            """
        )

    def test_set_last_column(self):
        # t[:, -1] = [10, 20, 30, 40]
        assert_isolated_ok(
            """
            d = F()
            d[:, -1] = FloatTensor(np.array([10., 20., 30., 40.]))
            expected = a.copy(); expected[:, -1] = [10., 20., 30., 40.]
            assert np.asarray(d).tolist() == expected.tolist(), np.asarray(d).tolist()
            """
        )

    def test_set_oob_row_col_slice_raises(self):
        # t[5, 2:4] = ...  -> NumPy raises IndexError (no OOB write).
        assert_isolated_ok(
            """
            d = F()
            try:
                d[5, 2:4] = FloatTensor(np.array([0., 0.]))
            except IndexError:
                sys.exit(0)
            sys.exit("expected IndexError for out-of-bounds row")
            """
        )


# =============================================================================
# Multi-axis assignment must validate the values shape (no OOB write)
# =============================================================================


class TestMultiAxisAssignShapeValidation:
    def test_outer_assign_wrong_shape_raises(self):
        # t[[0,2],[1,3]] selects a (2,2) block (outer / np.ix_ semantics);
        # assigning a (3,3) tensor must raise, not write out of bounds.
        assert_isolated_ok(
            """
            d = F()
            try:
                d[np.array([0, 2]), np.array([1, 3])] = FloatTensor(np.full((3, 3), 9.0))
            except (ValueError, RuntimeError):
                sys.exit(0)
            sys.exit("expected shape-mismatch error from outer assignment")
            """
        )

    def test_bool_mask_multi_axis_assign_wrong_shape_raises(self):
        # t[rowmask, colmask] selects a (2, 2) block (outer / np.ix_ semantics);
        # assigning a (3, 3) tensor must raise rather than write out of bounds.
        assert_isolated_ok(
            """
            d = F()
            rows = BoolTensor(np.array([True, False, True, False]))               # 2 rows
            cols = BoolTensor(np.array([True, False, True, False, False, False]))  # 2 cols
            try:
                d[rows, cols] = FloatTensor(np.full((3, 3), -9.0))
            except (ValueError, RuntimeError):
                sys.exit(0)
            sys.exit("expected shape-mismatch error from multi-axis mask assignment")
            """
        )


# =============================================================================
# Out-of-bounds multi-axis fancy index must raise (no OOB read)
# =============================================================================


class TestMultiAxisFancyBounds:
    def test_oob_multi_axis_fancy_index_raises(self):
        # t[rows, cols] with an out-of-range row index must raise IndexError
        # rather than read out of bounds (whatever kernel ends up backing it).
        assert_isolated_ok(
            """
            try:
                _ = F()[np.array([0, 9]), np.array([0, 1])]   # 9 is OOB on axis 0 (size 4)
            except IndexError:
                sys.exit(0)
            sys.exit("expected IndexError for out-of-bounds multi-axis fancy index")
            """
        )
