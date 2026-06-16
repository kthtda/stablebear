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

"""Regression tests for issue #57: comparison operators with a scalar (or
numpy-array) right-hand side used to crash with ``AttributeError: 'int' object
has no attribute '_data'`` instead of returning a boolean mask."""

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


_NUMERIC_TYPES = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
    pytest.param(sb.IntTensor, np.int32, id="int32"),
    pytest.param(sb.IntTensor, np.int64, id="int64"),
]

_ORDER_OPS = [
    pytest.param(lambda x, y: x < y, id="lt"),
    pytest.param(lambda x, y: x <= y, id="le"),
    pytest.param(lambda x, y: x > y, id="gt"),
    pytest.param(lambda x, y: x >= y, id="ge"),
]


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
@pytest.mark.parametrize("op", _ORDER_OPS)
def test_scalar_rhs_matches_numpy(TensorType, np_dtype, op):
    np_a = np.array([1, 5, 3, 2, 4], dtype=np_dtype)
    t = TensorType(np_a)
    result = op(t, 3)
    expected = op(np_a, 3)
    assert isinstance(result, sb.BoolTensor)
    npt.assert_array_equal(np.asarray(result), expected)


def test_mask_select_idiom():
    """The canonical ``t[t > scalar]`` masking idiom should work directly."""
    np_a = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    t = sb.FloatTensor(np_a)
    selected = t[t > 3]
    npt.assert_array_equal(np.asarray(selected), np_a[np_a > 3])


def test_mask_select_idiom_2d():
    np_a = np.arange(12, dtype=np.float64).reshape(3, 4)
    t = sb.FloatTensor(np_a)
    selected = t[t >= 5]
    npt.assert_array_equal(np.asarray(selected), np_a[np_a >= 5])


@pytest.mark.parametrize("op", _ORDER_OPS)
def test_ndarray_rhs_broadcasts(op):
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    np_b = np.array([2.0, 3.0])
    t = sb.FloatTensor(np_a)
    result = op(t, np_b)
    expected = op(np_a, np_b)
    assert isinstance(result, sb.BoolTensor)
    assert result.shape == expected.shape
    npt.assert_array_equal(np.asarray(result), expected)


def test_tensor_rhs_still_works():
    np_a = np.array([1.0, 5.0, 3.0])
    np_b = np.array([2.0, 4.0, 3.0])
    result = sb.FloatTensor(np_a) > sb.FloatTensor(np_b)
    assert isinstance(result, sb.BoolTensor)
    npt.assert_array_equal(np.asarray(result), np_a > np_b)


def test_non_numeric_tensor_scalar_compare_raises_typeerror():
    """A PCF tensor compared to a scalar should raise a clear ``TypeError``
    (not an ``AttributeError`` leaking the ``_data`` internals)."""
    pcfs = sb.random.noisy_sin((3,))
    with pytest.raises(TypeError):
        _ = pcfs > 3
