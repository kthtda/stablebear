"""Extended tests for lp_norm: L2/Lp/Linf norms, edge cases, and dtype coverage."""

import math

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


def _pcf(points, dtype):
    """Create a Pcf with the correct numpy dtype for the given stablebear dtype."""
    np_dtype = np.float32 if dtype == sb.pcf32 else np.float64
    return sb.Pcf(np.array(points, dtype=np_dtype), dtype=dtype)


# --- L2 norm ---



def test_l2_norm_constant_function(pcf_dtype):
    """L2 norm of f=c on [0,T) is c*sqrt(T)."""
    X = sb.zeros((1,), dtype=pcf_dtype)
    X[0] = _pcf([[0.0, 3.0], [4.0, 0.0]], pcf_dtype)
    result = sb.lp_norm(X, p=2)
    # L2 = sqrt(9 * 4) = 6
    npt.assert_allclose(result[0], 6.0, rtol=1e-5)



def test_l2_norm_two_step_function(pcf_dtype):
    """L2 norm of a piecewise constant function with two steps."""
    X = sb.zeros((1,), dtype=pcf_dtype)
    X[0] = _pcf([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]], pcf_dtype)
    result = sb.lp_norm(X, p=2)
    # L2 = sqrt(1*1 + 1*4) = sqrt(5)
    npt.assert_allclose(result[0], np.sqrt(5.0), rtol=1e-5)



def test_l2_norm_differs_from_l1():
    """L2 norm should differ from L1 norm for non-trivial functions."""
    X = sb.zeros((1,), dtype=sb.pcf64)
    # f = 1 on [0,1), 2 on [1,2)
    X[0] = sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]]))
    l1 = sb.lp_norm(X, p=1)
    l2 = sb.lp_norm(X, p=2)
    # L1 = 3, L2 = sqrt(5) ~ 2.236 -- they must differ
    assert l1[0] != pytest.approx(l2[0], abs=0.1)


# --- Lp norm for general p ---



def test_lp_norm_p3(pcf_dtype):
    """L3 norm of f with two steps where L3 != L1."""
    X = sb.zeros((1,), dtype=pcf_dtype)
    # f = 1 on [0,1), 3 on [1,2) => L1 = 4, L3 = (1 + 27)^(1/3) = 28^(1/3) ~ 3.037
    X[0] = _pcf([[0.0, 1.0], [1.0, 3.0], [2.0, 0.0]], pcf_dtype)
    result = sb.lp_norm(X, p=3)
    expected = (1.0 + 27.0) ** (1.0 / 3.0)
    npt.assert_allclose(result[0], expected, rtol=1e-4)



def test_lp_norm_large_p_approaches_linfinity(pcf_dtype):
    """For large p, Lp norm should approach max|f|."""
    X = sb.zeros((1,), dtype=pcf_dtype)
    X[0] = _pcf([[0.0, 1.0], [1.0, 5.0], [2.0, 2.0], [3.0, 0.0]], pcf_dtype)
    result = sb.lp_norm(X, p=50)
    npt.assert_allclose(result[0], 5.0, atol=0.1)


def test_lp_norm_large_p_does_not_overflow(pcf_dtype):
    """5^500 overflows both float32 and float64, but the L500 norm is about 5."""
    X = sb.zeros((1,), dtype=pcf_dtype)
    X[0] = _pcf([[0.0, 1.0], [1.0, 5.0], [2.0, 2.0], [3.0, 0.0]], pcf_dtype)
    result = sb.lp_norm(X, p=500)
    # (1 + 5^500 + 2^500)^(1/500) = 5 * (1 + (1/5)^500 + (2/5)^500)^(1/500)
    npt.assert_allclose(result[0], 5.0, rtol=1e-6)


# --- Linf norm ---


def test_linf_norm_is_largest_absolute_value(pcf_dtype):
    """The largest |f| is in a middle segment and negative, not the first value."""
    X = sb.zeros((1,), dtype=pcf_dtype)
    X[0] = _pcf([[0.0, 2.0], [1.0, -5.0], [2.0, 3.0], [3.0, 0.0]], pcf_dtype)
    result = sb.lp_norm(X, p=math.inf)
    assert result[0] == 5.0


def test_linf_norm_of_each_pcf_in_tensor(pcf_dtype):
    X = sb.zeros((2,), dtype=pcf_dtype)
    X[0] = _pcf([[0.0, 1.0], [1.0, 4.0], [2.0, 0.0]], pcf_dtype)
    X[1] = _pcf([[0.0, -6.0], [1.0, 2.0], [2.0, 0.0]], pcf_dtype)
    result = sb.lp_norm(X, p=math.inf)
    npt.assert_array_equal(np.asarray(result), [4.0, 6.0])


# --- Edge cases ---


@pytest.mark.parametrize("p", [1, 2, math.inf])
def test_lp_norm_nan_value_gives_nan(pcf_dtype, p):
    """NaN in a middle segment, between larger and smaller finite values."""
    X = sb.zeros((1,), dtype=pcf_dtype)
    X[0] = _pcf([[0.0, 4.0], [1.0, np.nan], [2.0, 2.0], [3.0, 0.0]], pcf_dtype)
    result = sb.lp_norm(X, p=p)
    assert np.isnan(result[0])


def test_lp_norm_zero_function(pcf_dtype):
    """Norm of the zero function is 0."""
    X = sb.zeros((1,), dtype=pcf_dtype)
    X[0] = _pcf([[0.0, 0.0]], pcf_dtype)
    result = sb.lp_norm(X, p=1)
    assert result[0] == pytest.approx(0.0)


def test_lp_norm_negative_values(pcf_dtype):
    """Norm uses |f|, so negative values should give same norm as positive."""
    X_pos = sb.zeros((1,), dtype=pcf_dtype)
    X_neg = sb.zeros((1,), dtype=pcf_dtype)
    X_pos[0] = _pcf([[0.0, 3.0], [1.0, 0.0]], pcf_dtype)
    X_neg[0] = _pcf([[0.0, -3.0], [1.0, 0.0]], pcf_dtype)
    for p in [1, 2]:
        npt.assert_allclose(sb.lp_norm(X_pos, p=p), sb.lp_norm(X_neg, p=p), rtol=1e-6)


def test_lp_norm_multidimensional_tensor(pcf_dtype):
    """lp_norm should work on 2D tensors and return matching shape."""
    X = sb.zeros((2, 3), dtype=pcf_dtype)
    for i in range(2):
        for j in range(3):
            X[i, j] = _pcf([[0.0, float(i + j + 1)], [1.0, 0.0]], pcf_dtype)
    result = sb.lp_norm(X, p=1)
    assert result.shape == (2, 3)
    # L1 of f=c on [0,1) is c
    for i in range(2):
        for j in range(3):
            assert result[i, j] == pytest.approx(float(i + j + 1), rel=1e-5)
