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

"""NumPy-parity tests for the various index objects NumPy accepts.

Covers review findings: fancy-python-list-index, numpy-scalar-int-unhandled,
bool-scalar-index, fancy-float-index-wrong-error, float-scalar-index-wrong-error,
empty-tuple-index, partial-multiint-tuple, too-many-indices-wrong-message,
fancy-multidim-int-index (single N-D / 0-d integer index array).

Each asserts the correct NumPy behavior. Most currently FAIL (``_pyslice_to_slice``
and ``_coerce_index_arrays`` reject lists / numpy scalars / Ellipsis-like objects,
and integer index arrays must be 1-D); a couple already pass and serve as
contract/regression guards.
"""

import numpy as np
import pytest

from stablebear.tensor import FloatTensor
from _indexing_support import assert_getitem_matches, assert_setitem_matches, ref_array




# =============================================================================
# Python list as a fancy index (NumPy treats it like an integer array)
# =============================================================================


class TestPythonListIndex:
    def test_list_rows(self):
        assert_getitem_matches(ref_array(), [0, 2])

    def test_list_rows_reordered(self):
        assert_getitem_matches(ref_array(), [2, 0, 1])

    def test_list_rows_negative(self):
        assert_getitem_matches(ref_array(), [0, -1])

    def test_list_on_column_axis(self):
        assert_getitem_matches(ref_array(), (slice(None), [1, 3]))

    def test_list_duplicates(self):
        assert_getitem_matches(ref_array(), [1, 1, 2])

    def test_list_setitem_rows(self):
        assert_setitem_matches(ref_array(), [0, 2], np.full((2, 6), -1.0))

    def test_list_setitem_scalar(self):
        assert_setitem_matches(ref_array(), [0, 2], 0.0)


# =============================================================================
# NumPy scalar integers (np.int64, np.intp) behave like Python ints
# =============================================================================


class TestNumpyScalarIntIndex:
    def test_np_int64(self):
        assert_getitem_matches(ref_array(), np.int64(1))

    def test_np_intp(self):
        assert_getitem_matches(ref_array(), np.intp(2))

    def test_np_int32_in_tuple(self):
        assert_getitem_matches(ref_array(), (np.int64(1), np.int32(2)))


# =============================================================================
# Scalar booleans: NumPy adds a leading length-1 (True) or length-0 (False) axis
# =============================================================================


class TestScalarBoolIndex:
    def test_true_adds_leading_axis(self):
        assert_getitem_matches(ref_array(), True)

    def test_false_adds_empty_axis(self):
        assert_getitem_matches(ref_array(), False)


# =============================================================================
# Float indices: NumPy raises IndexError (only ints/bools are valid)
# =============================================================================


class TestFloatIndexRaises:
    def test_float_scalar_raises_indexerror(self):
        t = FloatTensor(ref_array())
        with pytest.raises(IndexError):
            _ = t[1.0]

    def test_float_array_raises_indexerror(self):
        t = FloatTensor(ref_array())
        with pytest.raises(IndexError):
            _ = t[np.array([0.0, 1.0])]


# =============================================================================
# Tuple arity: empty tuple, partial tuple, too many indices
# =============================================================================


class TestTupleArity:
    def test_empty_tuple_returns_full(self):
        # NumPy: a[()] is the whole array.
        assert_getitem_matches(ref_array(), ())

    def test_partial_tuple_returns_subtensor(self):
        # NumPy: 3-D indexed by 2 ints -> remaining-axis vector.
        B = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)
        assert_getitem_matches(B, (1, 2))

    def test_too_many_indices_raises(self):
        t = FloatTensor(ref_array())
        with pytest.raises(IndexError):
            _ = t[0, 1, 2]


# =============================================================================
# Single N-D / 0-d integer index array (independent of multi-axis semantics)
# =============================================================================


class TestNdIndexArray:
    def test_2d_index_array_adopts_shape(self):
        # NumPy: a[idx] has shape idx.shape + a.shape[1:] -> (2, 2, 6).
        assert_getitem_matches(ref_array(), np.array([[0, 1], [2, 3]]))

    def test_0d_index_array_drops_axis(self):
        # NumPy: a[np.array(2)] is row 2, shape (6,).
        assert_getitem_matches(ref_array(), np.array(2))
