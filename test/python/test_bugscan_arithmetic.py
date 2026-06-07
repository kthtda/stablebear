"""Red-until-fixed regression tests for known arithmetic bugs (bug scan).

These tests document *known, currently-unfixed* defects in the arithmetic
surface of masspcf, discovered by a broad API fuzz/scan (see
``bug-scan-findings.md`` at the repo root for the catalogue and root-cause
hints). Each test asserts the CORRECT / intended behavior, so it FAILS today
(red) and will pass once the underlying bug is fixed. Do **not** weaken a test
to make it green -- fix the bug instead.

All three defects in this area exit cleanly (no SIGSEGV/SIGABRT/SIGFPE -- the
``is_crasher`` hint notwithstanding, every repro was confirmed to exit 0 with a
silently wrong value), so they run safely in-process: an uncaught exception or
failed assert here fails cleanly without aborting the suite.
"""

import operator

import numpy as np
import numpy.testing as npt
import pytest

import masspcf as mpcf


def test_rank0_arithmetic_computes_numpy_semantics():
    """Rank-0 (0-d) arithmetic must compute, not silently return 0 / no-op."""
    # BUG: Arithmetic producing a 0-d (rank-0) result silently returns 0
    #      (and in-place ops on a 0-d tensor are silent no-ops).
    # Expected: NumPy 0-d semantics (the single element IS visited/computed).
    # Observed today: every 0-d-output op returns 0.0 regardless of operands,
    #      and in-place ops leave the value unchanged.

    # Scalar ops on a 0-d float tensor.
    z = mpcf.zeros((), dtype=mpcf.float64)
    z[()] = 5.0
    assert float(z) == 5.0  # storage is correct; only the op path is broken
    assert float(np.asarray(z + 1.0)) == 6.0   # observed: 0.0
    assert float(np.asarray(z * 2.0)) == 10.0  # observed: 0.0
    assert float(np.asarray(z ** 2)) == 25.0   # observed: 0.0

    # Tensor-tensor op whose broadcast result is 0-d.
    a = mpcf.zeros((), dtype=mpcf.float64)
    a[()] = 3.0
    b = mpcf.zeros((), dtype=mpcf.float64)
    b[()] = 4.0
    assert float(np.asarray(a + b)) == 7.0  # observed: 0.0

    # In-place op on a 0-d tensor must mutate, not silently no-op.
    z2 = mpcf.zeros((), dtype=mpcf.float64)
    z2[()] = 5.0
    z2 += 1.0
    assert float(z2) == 6.0  # observed: 5.0 (silent no-op)

    # 0-d integer tensor reached via squeeze() of a size-1 tensor.
    si = mpcf.IntTensor(np.array([5], dtype=np.int64)).squeeze()
    assert int(np.asarray(si + 3)) == 8  # observed: 0


def test_inplace_through_broadcast_view_does_not_corrupt_source():
    """In-place arithmetic through a stride-0 broadcast_to view must not corrupt."""
    # BUG: In-place arithmetic through a broadcast_to view double-writes shared
    #      memory (silent source corruption).
    # Expected: NumPy makes broadcast views non-writeable and raises ValueError
    #      ('output array is read-only') for in-place arithmetic. If a write were
    #      allowed at all, the only coherent result for `bv += 1.0` on [1,2,3] is
    #      [2,3,4] (each physical cell incremented once), never [3,4,5].
    # Observed today: source silently corrupted to [3,4,5] / [31,32,33] (each
    #      physical cell written once per LOGICAL element), no error raised.

    # NumPy reference: in-place arithmetic on a broadcast view is rejected.
    nbv = np.broadcast_to(np.array([1.0, 2.0, 3.0]), (2, 3))
    with pytest.raises(ValueError):
        nbv += 1.0

    # Scalar in-place case.
    b = mpcf.FloatTensor(np.array([1.0, 2.0, 3.0]))
    bv = b.broadcast_to((2, 3))  # axis-0 stride 0: both rows alias the same cells
    try:
        bv += 1.0
    except ValueError:
        # Matching NumPy by refusing to write a non-writeable broadcast view is
        # an acceptable fix -- and crucially leaves the source intact.
        npt.assert_array_equal(np.asarray(b), [1.0, 2.0, 3.0])
    else:
        # If the write was allowed, each physical cell must be incremented
        # exactly once -- never twice. Observed today: [3., 4., 5.] (corrupt).
        npt.assert_array_equal(np.asarray(b), [2.0, 3.0, 4.0])

    # Tensor-tensor in-place case: both RHS rows must not accumulate into the
    # same 3 cells. Observed today: [31., 32., 33.] (= 1 + 10 + 20).
    b2 = mpcf.FloatTensor(np.array([1.0, 2.0, 3.0]))
    bv2 = b2.broadcast_to((2, 3))
    rhs = mpcf.FloatTensor(np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]))
    try:
        bv2 += rhs
    except ValueError:
        npt.assert_array_equal(np.asarray(b2), [1.0, 2.0, 3.0])
    else:
        # A single coherent write per physical cell cannot add two RHS rows into
        # one cell, so the source must NOT be [31, 32, 33].
        assert not np.array_equal(np.asarray(b2), [31.0, 32.0, 33.0]), (
            "in-place add through broadcast view corrupted the source tensor"
        )


@pytest.mark.parametrize(
    "np_dtype, base, exp",
    [
        pytest.param(np.int32, 100, 5, id="int32"),
        pytest.param(np.int64, 100000, 5, id="int64"),
    ],
)
def test_integer_pow_overflow_matches_numpy_wrap(np_dtype, base, exp):
    """Integer ** overflow must match NumPy's modular wrap, not saturate to INT_MIN."""
    # BUG: Integer tensor ** overflow saturates to INT_MIN (via float64 pow +
    #      UB double->int cast) instead of NumPy's modular two's-complement wrap.
    # Expected: NumPy integer pow semantics (modular wraparound).
    # Observed today: int32 100**5 -> -2147483648, int64 100000**5 -> INT64_MIN,
    #      silently (no RuntimeWarning / exception).
    t = mpcf.IntTensor(np.array([base], dtype=np_dtype))
    result = int(np.asarray(t ** exp)[0])
    expected = int((np.array([base], dtype=np_dtype) ** exp)[0])
    assert result == expected, (
        f"{base}**{exp} ({np_dtype.__name__}): got {result}, "
        f"numpy modular wrap is {expected}"
    )
