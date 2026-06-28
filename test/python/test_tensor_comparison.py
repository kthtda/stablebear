import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


# --- BoolTensor basics ---


def test_bool_tensor_zeros():
    bt = sb.zeros((3,), dtype=sb.boolean)
    expected = np.zeros((3,), dtype=bool)
    assert isinstance(bt, sb.BoolTensor)
    assert bt.shape == expected.shape
    npt.assert_array_equal(np.asarray(bt), expected)


def test_bool_tensor_bool_single_element():
    np_a = np.array([1.0])
    np_b = np.array([1.0])
    np_c = np.array([2.0])

    a = sb.FloatTensor(np_a)
    b = sb.FloatTensor(np_b)
    c = sb.FloatTensor(np_c)

    assert a.shape == np_a.shape

    assert bool(a == b) is bool(np_a == np_b)
    assert bool(a == c) is bool(np_a == np_c)


def test_bool_tensor_bool_multi_element_raises():
    np_a = np.array([1.0, 2.0])
    np_b = np.array([1.0, 2.0])

    # NumPy raises on bool() of multi-element array; stablebear should too
    with pytest.raises(ValueError):
        bool(np_a == np_b)

    a = sb.FloatTensor(np_a)
    b = sb.FloatTensor(np_b)
    with pytest.raises(ValueError, match="more than one element"):
        bool(a == b)


# --- Elementwise comparisons (parameterized across float and int) ---


_NUMERIC_TYPES = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
    pytest.param(sb.IntTensor, np.int32, id="int32"),
    pytest.param(sb.IntTensor, np.int64, id="int64"),
]


def _assert_comparison(np_a, np_b, op, TensorType=sb.FloatTensor):
    """Assert that a comparison op on tensors matches numpy."""
    result = op(TensorType(np_a), TensorType(np_b))
    expected = op(np_a, np_b)
    assert isinstance(result, sb.BoolTensor)
    npt.assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestElementwiseComparison:
    def test_eq(self, TensorType, np_dtype):
        np_a = np.array([1, 2, 3], dtype=np_dtype)
        np_b = np.array([1, 9, 3], dtype=np_dtype)
        _assert_comparison(np_a, np_b, lambda x, y: x == y, TensorType)

    def test_ne(self, TensorType, np_dtype):
        np_a = np.array([1, 2, 3], dtype=np_dtype)
        np_b = np.array([1, 9, 3], dtype=np_dtype)
        _assert_comparison(np_a, np_b, lambda x, y: x != y, TensorType)

    def test_lt(self, TensorType, np_dtype):
        np_a = np.array([1, 5, 3], dtype=np_dtype)
        np_b = np.array([2, 4, 3], dtype=np_dtype)
        _assert_comparison(np_a, np_b, lambda x, y: x < y, TensorType)

    def test_le(self, TensorType, np_dtype):
        np_a = np.array([1, 5, 3], dtype=np_dtype)
        np_b = np.array([2, 4, 3], dtype=np_dtype)
        _assert_comparison(np_a, np_b, lambda x, y: x <= y, TensorType)

    def test_gt(self, TensorType, np_dtype):
        np_a = np.array([1, 5, 3], dtype=np_dtype)
        np_b = np.array([2, 4, 3], dtype=np_dtype)
        _assert_comparison(np_a, np_b, lambda x, y: x > y, TensorType)

    def test_ge(self, TensorType, np_dtype):
        np_a = np.array([1, 5, 3], dtype=np_dtype)
        np_b = np.array([2, 4, 3], dtype=np_dtype)
        _assert_comparison(np_a, np_b, lambda x, y: x >= y, TensorType)


# --- Broadcasting comparisons ---


def _assert_broadcast_comparison(np_a, np_b, op):
    """Assert that a broadcasting comparison on FloatTensors matches numpy."""
    result = op(sb.FloatTensor(np_a), sb.FloatTensor(np_b))
    expected = op(np_a, np_b)
    assert isinstance(result, sb.BoolTensor)
    assert result.shape == expected.shape
    npt.assert_array_equal(np.asarray(result), expected)


def test_eq_broadcast_row():
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    np_b = np.array([1.0, 4.0])
    _assert_broadcast_comparison(np_a, np_b, lambda x, y: x == y)


def test_lt_broadcast_col():
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    np_b = np.array([[2.0], [3.0]])
    _assert_broadcast_comparison(np_a, np_b, lambda x, y: x < y)


def test_ge_broadcast_scalar_tensor():
    np_a = np.array([1.0, 2.0, 3.0])
    np_b = np.array([2.0])
    _assert_broadcast_comparison(np_a, np_b, lambda x, y: x >= y)


# --- array_equal ---


def test_array_equal_true():
    np_a = np.array([1.0, 2.0, 3.0])
    a = sb.FloatTensor(np_a)
    assert a.array_equal(a) == np.array_equal(np_a, np_a)
    assert a.array_equal(a.copy()) == np.array_equal(np_a, np_a.copy())


def test_array_equal_false():
    np_a = np.array([1.0, 2.0, 3.0])
    np_b = np.array([1.0, 9.0, 3.0])
    a = sb.FloatTensor(np_a)
    b = sb.FloatTensor(np_b)
    assert a.array_equal(b) == np.array_equal(np_a, np_b)


def test_array_equal_different_shape():
    np_a = np.array([1.0, 2.0])
    np_b = np.array([1.0, 2.0, 3.0])
    a = sb.FloatTensor(np_a)
    b = sb.FloatTensor(np_b)
    assert a.array_equal(b) == np.array_equal(np_a, np_b)


def test_array_equal_numpy():
    np_a = np.array([1.0, 2.0, 3.0])
    np_b = np.array([1.0, 9.0, 3.0])
    a = sb.FloatTensor(np_a)
    assert a.array_equal(np_a) == np.array_equal(np_a, np_a)
    assert a.array_equal(np_b) == np.array_equal(np_a, np_b)


# --- PCF tensor elementwise eq ---


def test_pcf_tensor_eq():
    a = sb.random.noisy_sin((3,))
    b = a.copy()
    result = a == b
    assert isinstance(result, sb.BoolTensor)
    npt.assert_array_equal(np.asarray(result), np.array([True, True, True]))


def test_pcf_tensor_array_equal():
    a = sb.random.noisy_sin((3,))
    b = a.copy()
    assert a.array_equal(b)


# ---------------------------------------------------------------------------
# Bug #96: BoolTensor had no __repr__/__str__, so it printed as
# "<...BoolTensor object at 0x...>" in the REPL / notebooks. It now mirrors the
# numeric tensors and shows the underlying boolean array.
# ---------------------------------------------------------------------------


def test_bool_tensor_repr_shows_values():
    t = sb.BoolTensor(np.array([True, False, True]))
    assert "True" in repr(t) and "False" in repr(t)
    assert "object at 0x" not in repr(t)
    assert repr(t) == repr(np.asarray(t))


def test_bool_tensor_str_shows_values():
    t = sb.BoolTensor(np.array([True, False]))
    assert str(t) == str(np.asarray(t))


def test_bool_tensor_from_comparison_repr():
    t = sb.FloatTensor([1.0, 2.0, 3.0]) > 2.0
    assert isinstance(t, sb.BoolTensor)
    assert "True" in repr(t)
