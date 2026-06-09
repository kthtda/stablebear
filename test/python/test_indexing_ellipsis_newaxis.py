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

"""NumPy-parity tests for Ellipsis (``...``) and newaxis (``None``) indexing.

Covers review findings: ellipsis-unhandled, newaxis-none-unhandled.

Currently both raise ``TypeError('Unhandled slice type')`` because
``_pyslice_to_slice`` only understands ``int`` and ``slice``. These assert the
correct NumPy behavior and will pass once an Ellipsis/newaxis normalization
pass is added.
"""

import numpy as np
import pytest

from stablebear.tensor import FloatTensor
from _indexing_support import assert_getitem_matches, assert_setitem_matches, ref_array

_B = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)


# =============================================================================
# Ellipsis
# =============================================================================


class TestEllipsisGetitem:
    def test_bare_ellipsis(self):
        assert_getitem_matches(ref_array(), Ellipsis)

    def test_ellipsis_then_int(self):
        assert_getitem_matches(ref_array(), (Ellipsis, 0))

    def test_int_then_ellipsis(self):
        assert_getitem_matches(ref_array(), (0, Ellipsis))

    def test_ellipsis_in_middle_3d(self):
        assert_getitem_matches(_B, (0, Ellipsis, 1))

    def test_ellipsis_with_slice(self):
        assert_getitem_matches(_B, (Ellipsis, slice(1, 3)))

    def test_ellipsis_absorbs_zero_axes(self):
        # With ndim explicit indices already present, ... matches zero axes.
        assert_getitem_matches(ref_array(), (0, Ellipsis, 1))

    def test_multiple_ellipsis_raises(self):
        # NumPy: an index can only have a single ellipsis (IndexError).
        t = FloatTensor(ref_array())
        with pytest.raises(IndexError):
            _ = t[Ellipsis, 0, Ellipsis]


class TestEllipsisSetitem:
    def test_ellipsis_then_int_scalar(self):
        assert_setitem_matches(ref_array(), (Ellipsis, 0), 7.0)

    def test_ellipsis_then_int_array(self):
        assert_setitem_matches(ref_array(), (Ellipsis, 0), np.arange(50.0, 54.0))


# =============================================================================
# newaxis / None
# =============================================================================


class TestNewaxisGetitem:
    def test_leading_none(self):
        assert_getitem_matches(ref_array(), None)

    def test_trailing_none(self):
        assert_getitem_matches(ref_array(), (Ellipsis, None))

    def test_middle_none(self):
        assert_getitem_matches(ref_array(), (slice(None), None))

    def test_none_before_axis(self):
        assert_getitem_matches(ref_array(), (None, slice(None)))

    def test_none_with_int(self):
        assert_getitem_matches(ref_array(), (None, 1))

    def test_two_none(self):
        assert_getitem_matches(ref_array(), (None, slice(None), None))

    def test_np_newaxis_alias(self):
        assert_getitem_matches(ref_array(), (slice(None), np.newaxis))


class TestNewaxisSetitem:
    def test_leading_newaxis_scalar(self):
        # a[None] has shape (1, 4, 6); assigning a scalar broadcasts.
        assert_setitem_matches(ref_array(), None, 3.0)
