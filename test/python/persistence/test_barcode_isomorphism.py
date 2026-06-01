"""Tests for Barcode.is_isomorphic_to tolerance behavior."""

import numpy as np

from masspcf.persistence.barcode import Barcode


def _bc(pairs, dtype=np.float64):
    """Create a Barcode from a list of (birth, death) pairs."""
    if len(pairs) == 0:
        return Barcode(np.zeros((0, 2), dtype=dtype))
    return Barcode(np.array(pairs, dtype=dtype))


def test_isomorphic_is_order_independent():
    """Isomorphism ignores bar ordering."""
    a = _bc([[0.0, 1.0], [2.0, 3.0]])
    b = _bc([[2.0, 3.0], [0.0, 1.0]])
    assert a.is_isomorphic_to(b)


def test_isomorphic_tolerates_tiny_endpoint_noise():
    """Endpoints differing below display precision still compare isomorphic.

    This mirrors barcodes computed from a point cloud versus a precomputed
    distance matrix, whose endpoints can differ in low-order bits.
    """
    a = _bc([[0.0, 0.678862], [0.0, 0.880222], [0.0, np.inf]])
    b = a.to_numpy()
    b[0, 1] += 3e-13
    b[1, 1] -= 5e-13
    bb = Barcode(b)
    assert a.is_isomorphic_to(bb)


def test_exact_mode_rejects_tiny_noise():
    """With zero tolerance, sub-ULP differences make barcodes non-isomorphic."""
    a = _bc([[0.0, 0.678862], [0.0, 0.880222]])
    b = a.to_numpy()
    b[0, 1] += 3e-13
    bb = Barcode(b)
    assert not a.is_isomorphic_to(bb, atol=0.0, rtol=0.0)


def test_genuinely_different_bars_not_isomorphic():
    """Differences well above tolerance are not isomorphic."""
    a = _bc([[0.0, 1.0], [2.0, 3.0]])
    b = _bc([[0.0, 1.0], [2.0, 4.0]])
    assert not a.is_isomorphic_to(b)


def test_infinite_endpoints_must_match_exactly():
    """An infinite endpoint never matches a finite one, regardless of tolerance."""
    a = _bc([[0.0, np.inf]])
    b = _bc([[0.0, 1e12]])
    assert not a.is_isomorphic_to(b)


def test_different_bar_counts_not_isomorphic():
    a = _bc([[0.0, 1.0]])
    b = _bc([[0.0, 1.0], [0.0, 1.0]])
    assert not a.is_isomorphic_to(b)
