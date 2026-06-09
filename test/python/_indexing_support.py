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

"""Helpers for the tensor-indexing parity tests.

NumPy is the oracle: the stablebear result must match what NumPy produces for the
same index expression on the same array.
"""


def ref_array():
    """A reusable (4, 6) float64 reference array (a fresh copy each call)."""
    import numpy as np
    return np.arange(24.0, dtype=np.float64).reshape(4, 6)


def assert_getitem_matches(arr, index):
    """``np.asarray(FloatTensor(arr)[index])`` must equal ``arr[index]``.

    Compares shape and values against NumPy. ``index`` may be anything that is
    a legal NumPy index (int, slice, tuple, Ellipsis, None, list, ndarray, ...).
    """
    import numpy as np
    from stablebear.tensor import FloatTensor

    a = np.asarray(arr, dtype=np.float64)
    expected = a[index]
    got = np.asarray(FloatTensor(a.copy())[index])
    assert got.shape == expected.shape, f"shape {got.shape} != numpy {expected.shape}"
    np.testing.assert_array_equal(got, expected)


def assert_setitem_matches(arr, index, value):
    """``FloatTensor(arr)[index] = value`` must mutate like ``arr[index] = value``.

    ``value`` may be a scalar or an ndarray; an ndarray is wrapped in a
    ``FloatTensor`` for the stablebear side (its RHS contract).
    """
    import numpy as np
    from stablebear.tensor import FloatTensor

    a = np.asarray(arr, dtype=np.float64)
    expected = a.copy()
    expected[index] = value

    t = FloatTensor(a.copy())
    rhs = FloatTensor(np.asarray(value, dtype=np.float64)) if isinstance(value, np.ndarray) else value
    t[index] = rhs
    np.testing.assert_array_equal(np.asarray(t), expected)
