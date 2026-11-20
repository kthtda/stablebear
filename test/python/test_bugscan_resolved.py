"""Regression tests for bug-scan issues that were already fixed before this
sweep but lacked an issue-tagged test. Grouped here for traceability.

Covers #36, #48, #59, #78, #81.
"""
import numpy as np
import pytest

import stablebear as sb
import stablebear.point_process as pp
from stablebear.reductions import mean, max_time


# ---------------------------------------------------------------------------
# #36: sample_poisson with a negative rate passed a negative mean to
# std::poisson_distribution (UB). It now validates rate and raises ValueError.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [-3.0, -0.5, -1e9])
def test_sample_poisson_negative_rate_raises(rate):
    with pytest.raises(ValueError):
        pp.sample_poisson((3,), dim=2, rate=rate, generator=sb.random.Generator(1))


def test_sample_poisson_zero_rate_is_valid():
    # rate == 0 is a valid (degenerate) input: empty clouds, no error.
    out = pp.sample_poisson((2,), dim=2, rate=0.0, generator=sb.random.Generator(1))
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# #48: ordering comparisons (<, <=, >, >=) against a scalar or ndarray used to
# leak an AttributeError about the internal _data. They now return a BoolTensor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op, ref", [
    (lambda a: a < 2.0, [True, False, False]),
    (lambda a: a <= 2.0, [True, True, False]),
    (lambda a: a > 2.0, [False, False, True]),
    (lambda a: a >= 2.0, [False, True, True]),
])
def test_ordering_against_scalar_returns_bool_tensor(op, ref):
    a = sb.FloatTensor([1.0, 2.0, 3.0])
    result = op(a)
    assert isinstance(result, sb.BoolTensor)
    np.testing.assert_array_equal(np.asarray(result), np.array(ref))


def test_ordering_reflected_and_ndarray():
    a = sb.FloatTensor([1.0, 2.0, 3.0])
    assert isinstance(2.0 < a, sb.BoolTensor)
    np.testing.assert_array_equal(np.asarray(2.0 < a), np.array([False, False, True]))
    res = a < np.array([2.0, 2.0, 2.0])
    np.testing.assert_array_equal(np.asarray(res), np.array([True, False, False]))


# ---------------------------------------------------------------------------
# #59: writing through a broadcast view silently scattered to multiple elements
# (no read-only guard). Writing through a broadcast view now raises ValueError.
# ---------------------------------------------------------------------------


def test_column_broadcast_view_is_read_only():
    src = sb.FloatTensor(np.arange(4.0).reshape(4, 1))
    view = src.broadcast_to((4, 3))
    with pytest.raises(ValueError):
        view[0, 0] = 99.0
    with pytest.raises(ValueError):
        view += 1.0
    # the view is still readable, and .copy() yields a writeable tensor
    np.testing.assert_array_equal(np.asarray(view)[:, 0], np.arange(4.0))
    writeable = view.copy()
    writeable[0, 0] = 99.0
    assert np.asarray(writeable)[0, 0] == 99.0


# ---------------------------------------------------------------------------
# #78: np.asarray(SymmetricMatrix) produced a 0-d object array. It now returns
# the dense (n, n) matrix -- in particular for an l2_kernel result.
# ---------------------------------------------------------------------------


def test_l2_kernel_asarray_is_dense():
    X = sb.PcfTensor([
        sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0]])),
        sb.Pcf(np.array([[0.0, 2.0], [1.0, 3.0]])),
        sb.Pcf(np.array([[0.0, 0.0], [1.0, 1.0]])),
    ])
    K = sb.l2_kernel(X)
    arr = np.asarray(K)
    assert arr.dtype != object
    assert arr.shape == (3, 3)
    # symmetric kernel
    np.testing.assert_allclose(arr, arr.T)


# ---------------------------------------------------------------------------
# #81: reductions (mean / max_time) accept a negative dim (NumPy axis-from-end).
# ---------------------------------------------------------------------------


def test_mean_negative_dim_matches_positive():
    t = sb.zeros((2, 3, 4))
    assert tuple(mean(t, dim=-1).shape) == tuple(mean(t, dim=2).shape) == (2, 3)
    assert tuple(mean(t, dim=-2).shape) == tuple(mean(t, dim=1).shape) == (2, 4)


def test_max_time_negative_dim_matches_positive():
    t = sb.zeros((2, 3, 4))
    assert tuple(max_time(t, dim=-1).shape) == tuple(max_time(t, dim=2).shape)


@pytest.mark.parametrize("bad_dim", [3, -4, 100])
def test_reduction_out_of_range_dim_raises(bad_dim):
    t = sb.zeros((2, 3, 4))
    with pytest.raises(IndexError):   # numpy.AxisError subclasses IndexError
        mean(t, dim=bad_dim)
