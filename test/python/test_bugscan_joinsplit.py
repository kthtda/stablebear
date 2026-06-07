"""Red-until-fixed regression tests for join/split bugs from the bug scan.

These tests document *known, currently-unfixed* defects in the join/split family
of public tensor ops (``mpcf.split`` / ``mpcf.array_split`` / ``mpcf.concatenate``).
Each test asserts the CORRECT / intended behavior, so it FAILS today (red) and
will pass once the underlying bug is fixed. Do NOT weaken a test to make it
green -- fix the bug instead. See ``bug-scan-findings.md`` at the repo root for
the catalogue and root-cause hints.

One defect (``split(t, 0)``) hard-crashes the interpreter with SIGFPE
(modulo-by-zero), so it is exercised only through the subprocess crash-isolation
helpers in ``_bugscan_support`` -- never called directly in-process. The other
two defects raise a clean (but wrong/low-level) exception today, so they run as
normal in-process tests.
"""

import numpy as np
import pytest

import masspcf as mpcf

from _bugscan_support import assert_clean_raises


# ---------------------------------------------------------------------------
# Bug 18: split(t, 0) SIGFPEs (modulo-by-zero) and aborts the interpreter.
# ---------------------------------------------------------------------------
def test_split_zero_sections_raises_clean_exception():
    """split(t, 0) must raise a clean Python error, not SIGFPE/abort the process."""
    # BUG: split(tensor, 0) SIGFPEs (modulo-by-zero) and aborts the interpreter;
    #      array_split guards n==0 but split does not.
    # Expected: a clean, catchable ValueError/ZeroDivisionError (array_split
    #           already raises "number of sections must be > 0"; numpy raises
    #           ZeroDivisionError) -- a library must never crash the whole process.
    # Observed today: hard crash, SIGFPE (subprocess returncode -8, shell exit 136),
    #           no Python traceback, no catchable exception.
    # Run in a subprocess: calling split(t, 0) in-process would abort all of pytest.
    assert_clean_raises(
        """
        t = mpcf.FloatTensor(np.arange(6, dtype=np.float64))
        mpcf.split(t, 0)
        """,
        ("ValueError", "ZeroDivisionError"),
    )


# ---------------------------------------------------------------------------
# Bug 19: concatenate/split/array_split reject negative axis with a pybind
#         "incompatible function arguments" TypeError, while stack accepts it.
# ---------------------------------------------------------------------------
def test_negative_axis_resolves_like_numpy():
    """concatenate/split/array_split must accept negative axis like stack/numpy do."""
    # BUG: negative axis is rejected by a low-level pybind TypeError on
    #      concatenate/split/array_split, even though stack accepts axis=-1
    #      (tested in test_tensor_join.py:93) and the API mirrors numpy.
    # Expected: negative axis is resolved -- concatenate((2,3),(2,3),axis=-1) -> (2,6);
    #      split((3,4),2,axis=-1) -> two (3,2); array_split((3,4),3,axis=-1) ->
    #      (3,2),(3,1),(3,1).  (Weakest acceptable fallback would be a clean
    #      ValueError naming 'axis', but the numpy-like contract is the full result.)
    # Observed today: TypeError "concatenate()/split_sections()/array_split():
    #      incompatible function arguments ...".
    FT = mpcf.FloatTensor

    # concatenate along the last axis.
    a = FT(np.arange(6, dtype=np.float64).reshape(2, 3))
    b = FT(np.arange(6, 12, dtype=np.float64).reshape(2, 3))
    got = np.asarray(mpcf.concatenate((a, b), axis=-1))
    expected = np.concatenate(
        (np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)), axis=-1)
    np.testing.assert_array_equal(got, expected)
    assert got.shape == (2, 6)

    # split along the last axis.
    t = FT(np.arange(12, dtype=np.float64).reshape(3, 4))
    parts = mpcf.split(t, 2, axis=-1)
    np_parts = np.split(np.arange(12).reshape(3, 4), 2, axis=-1)
    assert len(parts) == len(np_parts)
    for part, np_part in zip(parts, np_parts):
        np.testing.assert_array_equal(np.asarray(part), np_part)
        assert np.asarray(part).shape == np_part.shape

    # array_split along the last axis (uneven).
    aparts = mpcf.array_split(t, 3, axis=-1)
    np_aparts = np.array_split(np.arange(12).reshape(3, 4), 3, axis=-1)
    assert len(aparts) == len(np_aparts)
    for part, np_part in zip(aparts, np_aparts):
        np.testing.assert_array_equal(np.asarray(part), np_part)
        assert np.asarray(part).shape == np_part.shape


# ---------------------------------------------------------------------------
# Bug 20: split/array_split with a negative index entry raise a low-level pybind
#         TypeError instead of numpy-style from-the-end offset.
# ---------------------------------------------------------------------------
def test_negative_split_index_offsets_like_numpy():
    """split/array_split with a negative index must split numpy-style (axis_size+idx)."""
    # BUG: a negative entry in the indices list (e.g. [-2]) raises
    #      "split_indices(): incompatible function arguments" because the binding
    #      takes std::vector<size_t>; every other index edge case already matches numpy.
    # Expected: numpy semantics -- np.split(arange(6), [-2]) -> [[0,1,2,3],[4,5]]
    #      (split at axis_size + idx).
    # Observed today: low-level pybind TypeError leaking the C++ overload signature.
    arr = np.arange(6, dtype=np.float64)
    np_parts = np.split(arr, [-2])

    t = mpcf.FloatTensor(arr)
    for op in (mpcf.split, mpcf.array_split):
        parts = op(t, [-2])
        assert len(parts) == len(np_parts)
        for part, np_part in zip(parts, np_parts):
            np.testing.assert_array_equal(np.asarray(part), np_part)
            assert np.asarray(part).shape == np_part.shape
