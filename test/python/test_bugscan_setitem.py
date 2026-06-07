"""Red-until-fixed regression tests for known __setitem__ defects.

These tests document *known, currently-unfixed* bugs found by the API bug scan
(area key: ``setitem``). Each test asserts the CORRECT / intended behavior, so it
FAILS today and will pass once the underlying bug is fixed. Do NOT weaken these
to make them green -- fix the bug instead. See ``bug-scan-findings.md`` at the
repo root for the catalogue and root-cause hints.

Neither defect below hard-crashes the interpreter (both exit cleanly: one
silently writes wrong values, the other raises a raw pybind11 TypeError), so they
are written as ordinary in-process tests.
"""

import numpy as np
import pytest

import masspcf as mpcf
from masspcf.tensor import FloatTensor, IntTensor


def test_aliased_basic_slice_assignment_matches_numpy():
    """Self-aliasing basic-slice assignment must match NumPy (no in-place corruption)."""
    # BUG: Self-aliasing basic-slice assignment (a[:] = a[::-1], a[1:] = a[:-1])
    #      silently corrupts data because the RHS view shares storage with the
    #      destination and is read back mid-write.
    # Expected: NumPy-equivalent results (NumPy materializes an overlapping RHS
    #           into a temporary before assigning, so the result is well-defined).
    # Observed today: in-place read-back of overlapping storage writes wrong
    #                 values silently (no exception raised).

    # Case 1: in-place reversal.
    a = FloatTensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    a[:] = a[::-1]
    np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np_a[:] = np_a[::-1]
    assert np.asarray(a).tolist() == np_a.tolist()  # expected [4, 3, 2, 1]

    # Case 2: 2D row shift-down with backward overlap.
    b = FloatTensor(np.arange(12, dtype=np.float32).reshape(4, 3))
    b[1:] = b[:-1]
    np_b = np.arange(12, dtype=np.float32).reshape(4, 3)
    np_b[1:] = np_b[:-1]
    assert np.array_equal(np.asarray(b), np_b)  # expected rows [0,1,2],[0,1,2],[3,4,5],[6,7,8]

    # Case 3: 1D forward overlap.
    c = FloatTensor(np.arange(6, dtype=np.float32))
    c[1:] = c[:-1]
    np_c = np.arange(6, dtype=np.float32)
    np_c[1:] = np_c[:-1]
    assert np.asarray(c).tolist() == np_c.tolist()  # expected [0,0,1,2,3,4]


def test_float_scalar_into_inttensor_no_raw_pybind_error():
    """Python-float scalar/slice assignment into IntTensor must not emit a raw pybind dump."""
    # BUG: Python-float scalar/slice assignment into IntTensor raises a raw
    #      pybind11 TypeError dump instead of truncating (NumPy parity, matching
    #      the cross-dtype tensor-RHS path) or raising a clean library error.
    # Expected: either truncate to int (NumPy parity -- np.array([1,2,3]);
    #           a[0]=7.9 -> [7,2,3]) like the working tensor-RHS path, OR raise
    #           the library's own clean Pythonic TypeError from
    #           _validate_setitem_dtype. The raw pybind11 overload/constructor
    #           "incompatible ... arguments" dump is never acceptable, and float
    #           is explicitly declared a valid setitem dtype for IntTensor.
    # Observed today: b[0] = 7.9 -> "TypeError: _set_element(): incompatible
    #                 function arguments ..."; c[0:2] = 7.9 -> "TypeError:
    #                 __init__(): incompatible constructor arguments ...".

    # Sanity: the equivalent cross-dtype tensor-RHS path already truncates.
    ref = IntTensor(np.array([0, 0, 0], dtype=np.int64))
    ref[:] = FloatTensor(np.array([7.9, 8.9, 9.9], dtype=np.float32))
    assert np.asarray(ref).tolist() == [7, 8, 9]

    # Sanity: float is declared a valid setitem dtype for IntTensor.
    probe = IntTensor(np.array([1, 2, 3], dtype=np.int64))
    assert float in probe._get_valid_setitem_dtypes()

    # Scalar element assignment: must not leak a raw pybind11 signature dump.
    b = IntTensor(np.array([1, 2, 3], dtype=np.int64))
    try:
        b[0] = 7.9
    except TypeError as exc:
        # Acceptable only if it's a clean library error, not a pybind dump.
        msg = str(exc)
        assert "incompatible function arguments" not in msg
        assert "incompatible constructor arguments" not in msg
    else:
        # If it succeeded, it must have truncated to NumPy parity.
        assert np.asarray(b).tolist() == [7, 2, 3]

    # Slice broadcast-fill assignment: same contract.
    c = IntTensor(np.array([1, 2, 3, 4], dtype=np.int64))
    try:
        c[0:2] = 7.9
    except TypeError as exc:
        msg = str(exc)
        assert "incompatible function arguments" not in msg
        assert "incompatible constructor arguments" not in msg
    else:
        assert np.asarray(c).tolist() == [7, 7, 3, 4]
