import numpy as np
import pytest

import stablebear as sb


# ---------------------------------------------------------------------------
# Bug #8: arithmetic whose broadcast result is rank-0 (shape ()) silently
# returned 0, and in-place ops were silent no-ops, because walk_impl skipped
# the single rank-0 output element. A rank-0 result has exactly one element
# and must be computed (NumPy: np.array(5.0) + 1.0 == 6.0).
# ---------------------------------------------------------------------------


def _scalar0d(value, dtype=sb.float64):
    z = sb.zeros((), dtype=dtype)
    z[()] = value
    return z


def test_rank0_binary_scalar_ops():
    z = _scalar0d(5.0)
    assert np.asarray(z + 1.0).item() == 6.0
    assert np.asarray(z - 1.0).item() == 4.0
    assert np.asarray(z * 2.0).item() == 10.0
    assert np.asarray(z / 2.0).item() == 2.5
    assert np.asarray(z ** 2).item() == 25.0
    # the result stays rank-0
    assert tuple((z + 1.0).shape) == ()


def test_rank0_tensor_tensor_op():
    a = _scalar0d(3.0)
    b = _scalar0d(4.0)
    assert np.asarray(a + b).item() == 7.0


def test_rank0_inplace_ops_mutate():
    z = _scalar0d(5.0)
    z += 1.0
    assert float(z) == 6.0
    z -= 2.0
    assert float(z) == 4.0
    z *= 3.0
    assert float(z) == 12.0
    z /= 4.0
    assert float(z) == 3.0


def test_rank0_int_via_squeeze():
    si = sb.IntTensor(np.array([5], dtype=np.int64)).squeeze()
    assert tuple(si.shape) == ()
    assert np.asarray(si + 3).item() == 8


def test_rank0_broadcast_up_to_1d_still_works():
    """Regression guard: a 0-d operand broadcasting UP to a 1-D result always
    worked; keep it working."""
    z = _scalar0d(5.0)
    one_d = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))
    assert np.asarray(z + one_d).tolist() == [6.0, 7.0, 8.0]


def test_zero_size_arithmetic_stays_empty():
    """Contrast: arithmetic on a zero-size extent yields an empty result (no
    elements visited), unaffected by the rank-0 fix."""
    e = sb.FloatTensor(np.zeros((0,)))
    assert (e + 1.0).size == 0
