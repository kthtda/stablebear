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

"""Red-until-fixed regression tests for KNOWN, unfixed bugs in tensor
``__getitem__`` index normalization (bug-scan area ``getitem``).

These tests assert the CORRECT / intended behavior, so each FAILS today and
will pass once the underlying defect is fixed. Do NOT weaken a test to make it
green -- fix the bug instead. See ``bug-scan-findings.md`` at the repo root for
the catalogue and root-cause hints.

This area's single confirmed defect raises a clean Python ``ValueError`` (it
does not hard-crash the interpreter), so the test runs in-process.
"""

import numpy as np

from masspcf.tensor import FloatTensor, IntTensor


def test_0d_inttensor_index_drops_axis_like_numpy():
    """A 0-d IntTensor index must act as a scalar (axis-dropping) index."""
    # BUG: 0-d IntTensor index raises 'indices must be 1D' while equivalent
    #      0-d numpy int array / Python int are treated as a scalar
    #      (axis-dropping) index.
    # Expected: FloatTensor(a)[IntTensor(np.array(2))] returns row 2 with
    #           shape (6,), identical to a[np.array(2)] and a[2] (NumPy parity);
    #           the __getitem__ docstring lists 'integer arrays
    #           (numpy.ndarray, a Python list, or an IntTensor)' as equivalent.
    # Observed today: ValueError 'indices must be 1D' from the C++
    #           index_select path (validate_axis_indices rejects the 0-d
    #           selector that _normalize_index routed to the advanced index).
    a = np.arange(24.0).reshape(4, 6)

    # Independent NumPy reference for the intended result.
    expected = a[np.array(2)]
    assert expected.shape == (6,)

    # Sanity guard: the two index forms the docstring lists as equivalent to a
    # 0-d IntTensor already behave correctly (these must keep passing).
    assert np.asarray(FloatTensor(a.copy())[np.array(2)]).shape == (6,)
    assert np.asarray(FloatTensor(a.copy())[2]).shape == (6,)

    # The defect under test: a 0-d IntTensor must behave the same way.
    idx = IntTensor(np.array(2))
    assert idx.ndim == 0
    result = FloatTensor(a.copy())[idx]

    got = np.asarray(result)
    assert got.shape == (6,), (
        f"0-d IntTensor index should drop the leading axis (shape (6,)), "
        f"got {got.shape}")
    np.testing.assert_array_equal(got, expected)
