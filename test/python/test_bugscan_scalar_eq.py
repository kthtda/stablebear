import operator

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


# ---------------------------------------------------------------------------
# Bug #47: == / != against a Python/NumPy scalar returned a plain Python bool
# (via Python's identity fallback) instead of an elementwise BoolTensor,
# silently breaking the documented masking idiom t[t == scalar]. (#57 fixed
# the ordering operators < <= > >=; == and != were still broken.)
# ---------------------------------------------------------------------------

_NUMERIC = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
    pytest.param(sb.IntTensor, np.int64, id="int64"),
    pytest.param(sb.IntTensor, np.int32, id="int32"),
]


def _assert_compare(op, np_a, np_rhs, *, wrap_rhs=False, TensorType=sb.FloatTensor):
    """Run a comparison op in stablebear and numpy and assert they agree.

    The op is applied to ``TensorType(np_a)`` against ``np_rhs`` (optionally
    wrapped in a tensor) and to the raw numpy operands; the result must be a
    ``BoolTensor`` matching numpy's elementwise result.
    """
    sb_rhs = TensorType(np_rhs) if wrap_rhs else np_rhs
    result = op(TensorType(np_a), sb_rhs)
    assert isinstance(result, sb.BoolTensor)
    npt.assert_array_equal(np.asarray(result), op(np_a, np_rhs))


@pytest.mark.parametrize("TensorType, dt", _NUMERIC)
def test_eq_scalar_returns_bool_tensor(TensorType, dt):
    _assert_compare(operator.eq, np.array([1, 2, 3, 2], dtype=dt), 2, TensorType=TensorType)


@pytest.mark.parametrize("TensorType, dt", _NUMERIC)
def test_ne_scalar_returns_bool_tensor(TensorType, dt):
    _assert_compare(operator.ne, np.array([1, 2, 3, 2], dtype=dt), 2, TensorType=TensorType)


def test_eq_numpy_scalar_and_0d_array():
    for rhs in (np.float64(2.0), np.array(2.0)):
        _assert_compare(operator.eq, np.array([1.0, 2.0, 3.0]), rhs)


def test_eq_ndarray_broadcasts():
    _assert_compare(operator.eq, np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 4.0]))


def test_eq_tensor_rhs_still_works():
    _assert_compare(
        operator.eq, np.array([1.0, 2.0, 3.0]), np.array([1.0, 9.0, 3.0]), wrap_rhs=True
    )


def test_eq_2d_scalar_broadcast():
    _assert_compare(operator.eq, np.arange(6, dtype=np.float64).reshape(2, 3), 3.0)


def test_eq_scalar_mask_select_idiom():
    """The headline idiom: t[t == scalar] selects the matching elements."""
    np_a = np.array([1.0, 2.0, 3.0, 2.0])
    t = sb.FloatTensor(np_a)
    npt.assert_array_equal(np.asarray(t[t == 2.0]), np_a[np_a == 2.0])
