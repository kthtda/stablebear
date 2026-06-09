"""Tests for cdist (cross-distance computation)."""

import itertools

import numpy as np
import pytest

import stablebear as sb


def assert_cdist_matches_lp_distance(D, X, Y, p=1):
    """Assert every entry of cdist result D matches lp_distance(X[xi], Y[yi])."""
    x_ranges = [range(s) for s in X.shape]
    y_ranges = [range(s) for s in Y.shape]

    for xi in itertools.product(*x_ranges):
        for yi in itertools.product(*y_ranges):
            idx = xi + yi
            assert D[idx] == pytest.approx(sb.lp_distance(X[xi], Y[yi], p=p)), \
                f"Mismatch at D{list(idx)}"


def _pcf(points, dtype):
    """Create a Pcf from a list of [time, value] pairs with the given dtype."""
    return sb.Pcf(np.array(points), dtype=dtype)


def test_cdist_rejects_p_less_than_1(device):
    X = sb.zeros((2,))
    Y = sb.zeros((3,))
    with pytest.raises(ValueError, match="p must be >= 1"):
        sb.cdist(X, Y, p=0.5)


def test_cdist_1d_tensors(device, pcf_dtype):
    X = sb.zeros((2,), dtype=pcf_dtype)
    Y = sb.zeros((3,), dtype=pcf_dtype)

    X[0] = _pcf([[0.0, 5.0], [1.0, 0.0]], pcf_dtype)
    X[1] = _pcf([[0.0, 1.0], [2.0, 0.0]], pcf_dtype)

    Y[0] = _pcf([[0.0, 0.0]], pcf_dtype)
    Y[1] = _pcf([[0.0, 5.0], [1.0, 0.0]], pcf_dtype)
    Y[2] = _pcf([[0.0, 1.0]], pcf_dtype)

    D = sb.cdist(X, Y)

    assert isinstance(D, sb.FloatTensor)
    assert D.shape == (2, 3)
    assert_cdist_matches_lp_distance(D, X, Y)


def test_cdist_multidim(device, pcf_dtype):
    X = sb.zeros((2, 3), dtype=pcf_dtype)
    Y = sb.zeros((4,), dtype=pcf_dtype)

    for i in range(2):
        for j in range(3):
            X[i, j] = _pcf([[0.0, float(i * 3 + j + 1)], [1.0, 0.0]], pcf_dtype)

    for k in range(4):
        Y[k] = _pcf([[0.0, float(k)], [1.0, 0.0]], pcf_dtype)

    D = sb.cdist(X, Y)

    assert D.shape == (2, 3, 4)
    assert_cdist_matches_lp_distance(D, X, Y)


def test_cdist_matches_pdist_when_same_input(device, pcf_dtype):
    X = sb.zeros((3,), dtype=pcf_dtype)

    X[0] = _pcf([[0.0, 3.0], [1.0, 0.0]], pcf_dtype)
    X[1] = _pcf([[0.0, 1.0], [2.0, 0.0]], pcf_dtype)
    X[2] = _pcf([[0.0, 0.0]], pcf_dtype)

    D_pdist = sb.pdist(X)
    D_cdist = sb.cdist(X, X)

    assert D_cdist.shape == (3, 3)

    for i in range(3):
        for j in range(3):
            assert D_cdist[i, j] == pytest.approx(D_pdist[i, j], abs=1e-5), \
                f"Mismatch at ({i}, {j})"


def test_cdist_lp(device, pcf_dtype):
    X = sb.zeros((2,), dtype=pcf_dtype)
    Y = sb.zeros((1,), dtype=pcf_dtype)

    X[0] = _pcf([[0.0, 4.0], [1.0, 0.0]], pcf_dtype)
    X[1] = _pcf([[0.0, 1.0], [1.0, 0.0]], pcf_dtype)
    Y[0] = _pcf([[0.0, 1.0], [1.0, 0.0]], pcf_dtype)

    D = sb.cdist(X, Y, p=3)

    assert D.shape == (2, 1)
    assert_cdist_matches_lp_distance(D, X, Y, p=3)


def test_cdist_both_multidim(device, pcf_dtype):
    X = sb.zeros((2, 3), dtype=pcf_dtype)
    Y = sb.zeros((4, 2), dtype=pcf_dtype)

    for i in range(2):
        for j in range(3):
            X[i, j] = _pcf([[0.0, float(i * 3 + j + 1)], [1.0, 0.0]], pcf_dtype)

    for i in range(4):
        for j in range(2):
            Y[i, j] = _pcf([[0.0, float(i * 2 + j)], [1.0, 0.0]], pcf_dtype)

    D = sb.cdist(X, Y)

    assert D.shape == (2, 3, 4, 2)
    assert_cdist_matches_lp_distance(D, X, Y)


def test_cdist_sliced_inputs(device, pcf_dtype):
    """cdist with noncontiguous (sliced) tensor views."""
    X_full = sb.zeros((6,), dtype=pcf_dtype)
    Y_full = sb.zeros((8,), dtype=pcf_dtype)

    for i in range(6):
        X_full[i] = _pcf([[0.0, float(i + 1)], [1.0, 0.0]], pcf_dtype)
    for i in range(8):
        Y_full[i] = _pcf([[0.0, float(i)], [1.0, 0.0]], pcf_dtype)

    X = X_full[::2]  # elements 0, 2, 4 => values 1, 3, 5
    Y = Y_full[1::3]  # elements 1, 4, 7 => values 1, 4, 7

    D = sb.cdist(X, Y)

    assert D.shape == (3, 3)
    assert_cdist_matches_lp_distance(D, X, Y)


def test_cdist_sliced_multidim(device, pcf_dtype):
    """cdist with noncontiguous slices of multidimensional tensors."""
    X_full = sb.zeros((4, 6), dtype=pcf_dtype)
    Y_full = sb.zeros((3, 8), dtype=pcf_dtype)

    for i in range(4):
        for j in range(6):
            X_full[i, j] = _pcf([[0.0, float(i * 6 + j + 1)], [1.0, 0.0]], pcf_dtype)
    for i in range(3):
        for j in range(8):
            Y_full[i, j] = _pcf([[0.0, float(i * 8 + j)], [1.0, 0.0]], pcf_dtype)

    X = X_full[::2, 1::2]  # shape (2, 3)
    Y = Y_full[:, ::3]     # shape (3, 3)

    assert X.shape == (2, 3)
    assert Y.shape == (3, 3)

    D = sb.cdist(X, Y)

    assert D.shape == (2, 3, 3, 3)
    assert_cdist_matches_lp_distance(D, X, Y)


def test_cdist_empty(device, pcf_dtype):
    X = sb.zeros((0,), dtype=pcf_dtype)
    Y = sb.zeros((3,), dtype=pcf_dtype)

    D = sb.cdist(X, Y)
    assert D.shape == (0, 3)
