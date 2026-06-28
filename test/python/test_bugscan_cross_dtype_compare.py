import operator

import numpy as np
import pytest

import stablebear as sb


# ---------------------------------------------------------------------------
# Bug #49: cross-dtype comparisons (==, !=, <, <=, >, >=) and array_equal
# between two numeric tensors leaked the raw pybind overload error, because the
# C++ bindings only accept a same-typed RHS. They now promote via NumPy
# (result_type rules), matching NumPy's by-value semantics.
# ---------------------------------------------------------------------------


def _ftensor(dtype):
    return sb.FloatTensor(np.array([1.0, 2.0, 3.0], dtype=dtype))


def _itensor(dtype):
    return sb.IntTensor(np.array([1, 2, 3], dtype=dtype))


_OPS = [
    (operator.eq, np.equal),
    (operator.ne, np.not_equal),
    (operator.lt, np.less),
    (operator.le, np.less_equal),
    (operator.gt, np.greater),
    (operator.ge, np.greater_equal),
]


@pytest.mark.parametrize("op, np_op", _OPS)
@pytest.mark.parametrize(
    "lhs, rhs",
    [
        (_ftensor(np.float32), _ftensor(np.float64)),
        (_itensor(np.int32), _itensor(np.int64)),
        (_itensor(np.int64), _ftensor(np.float64)),
        (_itensor(np.uint64), _itensor(np.int64)),
    ],
)
def test_cross_dtype_comparison_matches_numpy(op, np_op, lhs, rhs):
    result = op(lhs, rhs)
    assert isinstance(result, sb.BoolTensor)
    ref = np_op(np.asarray(lhs), np.asarray(rhs))
    np.testing.assert_array_equal(np.asarray(result), ref)


def test_cross_dtype_comparison_is_by_value_not_cast():
    # int 1 vs float 1.5 must be unequal -- NumPy promotes both to float, it does
    # NOT cast 1.5 down to int 1 (which would wrongly report equality).
    result = sb.IntTensor([1]) == sb.FloatTensor([1.5])
    np.testing.assert_array_equal(np.asarray(result), np.array([False]))


@pytest.mark.parametrize(
    "lhs, rhs, expected",
    [
        (_itensor(np.int32), _itensor(np.int64), True),
        (_ftensor(np.float32), _ftensor(np.float64), True),
        (_itensor(np.int64), sb.FloatTensor([1.0, 2.0, 3.5]), False),
    ],
)
def test_cross_dtype_array_equal_matches_numpy(lhs, rhs, expected):
    assert lhs.array_equal(rhs) is expected
    assert lhs.array_equal(rhs) == bool(
        np.array_equal(np.asarray(lhs), np.asarray(rhs)))


def test_no_pybind_overload_leak():
    f = sb.FloatTensor([1.0, 2.0, 3.0])
    i = sb.IntTensor([1, 2, 3])
    # Previously these raised TypeError with the raw pybind overload table.
    for call in (lambda: f == i, lambda: f < i, lambda: f.array_equal(i)):
        call()  # must not raise


def test_same_dtype_comparison_unaffected():
    a = sb.FloatTensor([1.0, 2.0, 3.0])
    b = sb.FloatTensor([1.0, 9.0, 3.0])
    np.testing.assert_array_equal(np.asarray(a == b), np.array([True, False, True]))


def test_cross_dtype_mask_select_idiom():
    f = sb.FloatTensor([1.0, 2.0, 3.0])
    i = sb.IntTensor([1, 5, 3])
    selected = f[f == i]
    np.testing.assert_array_equal(np.asarray(selected), np.array([1.0, 3.0]))
