"""Red-until-fixed regression tests for known, unfixed bugs in the ``distance``
area of masspcf (Lp distance/norm and the pairwise/cross distance matrices).

These tests were authored from a broad API bug scan; each asserts the CORRECT
intended behavior so it FAILS today and will pass once the underlying defect is
fixed. Do NOT weaken these tests to make them green -- fix the bug instead. See
``bug-scan-findings.md`` at the repo root for the catalogue and root-cause hints.

Single root defect covered here (global index 21):
    Lp distance/norm with p>1 overflows to ``inf`` for results that are perfectly
    representable in the element dtype, because the per-interval ``pow``
    accumulation is not rescaled/normalized (no max-factoring or log-domain
    accumulation) before being summed and rooted. This corrupts ``lp_distance``,
    ``lp_norm``, ``cdist`` and ``pdist`` (CPU and GPU) at large-but-representable
    magnitudes. The p=1 path is unaffected (handles 1e300 fine), confirming the
    defect is specific to the p>1 power-accumulation path.

The repro does NOT hard-crash (it silently returns ``inf``), so these are
ordinary in-process tests; a failed assert here fails cleanly.
"""
import math

import numpy as np
import pytest

import masspcf as mpcf
from masspcf.distance_matrix import DistanceMatrix


# A constant PCF equal to ``value`` on [0, 1) and 0 afterwards. For such a
# function the Lp norm / distance against zero collapses to ``value`` for every
# p, since (integral_0^1 value^p dt)^(1/p) = (value^p)^(1/p) = value.
def _const_pcf(value, dtype):
    return mpcf.Pcf(np.array([[0.0, value], [1.0, 0.0]], dtype=dtype))


def _zero_pcf(dtype):
    return mpcf.Pcf(np.array([[0.0, 0.0]], dtype=dtype))


def test_lp_distance_p5_float32_returns_representable_value():
    """lp_distance with p>1 must not overflow when the result is representable."""
    # BUG: Lp distance (p>1) overflows to inf for representable results
    # Expected: lp_distance(1e8-on-[0,1), 0, p=5) == 1e8 (float32, 1e8 << 3.4e38)
    # Observed today: inf, because the intermediate (1e8)**5 = 1e40 overflows
    #                 float32 before the final p-th root is taken.
    val = np.float32(1e8)
    f = _const_pcf(val, np.float32)
    g = _zero_pcf(np.float32)

    d = mpcf.lp_distance(f, g, p=5)

    assert math.isfinite(d), f"lp_distance overflowed to {d}; 1e8 is representable"
    assert d == pytest.approx(1e8, rel=1e-5)


def test_lp_distance_p3_float64_returns_representable_value():
    """lp_distance with p>1 in float64 must not overflow a representable result."""
    # BUG: Lp distance (p>1) overflows to inf for representable results
    # Expected: lp_distance(1e120-on-[0,1), 0, p=3) == 1e120 (float64, well below 1.8e308)
    # Observed today: inf, because (1e120)**3 = 1e360 overflows float64.
    f = _const_pcf(1e120, np.float64)
    g = _zero_pcf(np.float64)

    d = mpcf.lp_distance(f, g, p=3)

    assert math.isfinite(d), f"lp_distance overflowed to {d}; 1e120 is representable"
    assert d == pytest.approx(1e120, rel=1e-12)


def test_lp_norm_p5_returns_representable_value():
    """lp_norm with p>1 must not overflow when the result is representable."""
    # BUG: Lp norm (p>1) overflows to inf for representable results
    # Expected: lp_norm(1e8-on-[0,1), p=5) == 1e8 (float32)
    # Observed today: inf (same un-rescaled pow accumulation).
    f = _const_pcf(np.float32(1e8), np.float32)

    norms = np.asarray(mpcf.lp_norm(mpcf.PcfTensor([f]), p=5))

    assert norms.shape == (1,)
    assert math.isfinite(norms[0]), f"lp_norm overflowed to {norms[0]}; 1e8 is representable"
    assert norms[0] == pytest.approx(1e8, rel=1e-5)


def test_cdist_p5_returns_representable_value():
    """cdist with p>1 must not overflow when the result is representable."""
    # BUG: cdist (p>1) overflows to inf for representable results
    # Expected: cdist(1e8-on-[0,1), 0, p=5)[0,0] == 1e8 (float32)
    # Observed today: inf.
    f = _const_pcf(np.float32(1e8), np.float32)
    g = _zero_pcf(np.float32)

    D = np.asarray(mpcf.cdist(mpcf.PcfTensor([f]), mpcf.PcfTensor([g]), p=5, verbose=False))

    assert D.shape == (1, 1)
    val = float(D[0, 0])
    assert math.isfinite(val), f"cdist overflowed to {val}; 1e8 is representable"
    assert val == pytest.approx(1e8, rel=1e-5)


def test_pdist_p5_returns_representable_value(device):
    """pdist with p>1 must not overflow a representable result (CPU and GPU)."""
    # BUG: pdist (p>1) overflows to inf for representable results, both CPU and
    #      GPU kernels (same un-rescaled OperationLpDist code path).
    # Expected: pdist of {1e8-on-[0,1), 0} at p=5 has off-diagonal entry 1e8.
    # Observed today: inf at the off-diagonal on both devices.
    f = _const_pcf(np.float32(1e8), np.float32)
    g = _zero_pcf(np.float32)
    X = mpcf.PcfTensor([f, g])

    D = mpcf.pdist(X, p=5, verbose=False)
    assert isinstance(D, DistanceMatrix)
    dense = D.to_dense()

    off = float(dense[0, 1])
    assert math.isfinite(off), f"pdist overflowed to {off} on {device}; 1e8 is representable"
    assert off == pytest.approx(1e8, rel=1e-5)
    assert dense[1, 0] == pytest.approx(off)
    assert dense[0, 0] == 0.0
    assert dense[1, 1] == 0.0
