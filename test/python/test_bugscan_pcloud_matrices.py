"""Red-until-fixed regression tests for the ``pcloud_matrices`` bug-scan area.

These tests document KNOWN, currently-unfixed defects in the installed
``masspcf`` package that involve PointCloud / DistanceMatrix / SymmetricMatrix
element tensors (assignment, validation, and read-back semantics). See the
bug-scan catalogue ``bug-scan-findings.md`` at the repo root for the full list
and root-cause hints.

Each test asserts the CORRECT / intended behavior, so it FAILS today (red) and
will pass once the underlying bug is fixed. Do NOT weaken a test to make it
green -- fix the bug instead.

Every repro in this area exits cleanly (no SIGSEGV / SIGABRT / SIGFPE), so the
tests run in-process: an uncaught exception or failed assert fails this test
cleanly without aborting the rest of the pytest session.
"""
import numpy as np
import pytest

import masspcf as mpcf
from masspcf.distance_matrix import DistanceMatrix
from masspcf.symmetric_matrix import SymmetricMatrix


# ---------------------------------------------------------------------------
# Bug 38: element assignment aliases the source's storage instead of copying.
# ---------------------------------------------------------------------------
def test_element_assignment_copies_not_aliases():
    """Assignment into matrix/pcloud element tensors must copy, not alias."""
    # BUG: Element assignment into PointCloud/DistanceMatrix/SymmetricMatrix
    #      tensors aliases the source storage instead of copying.
    # Expected: numpy object-array .copy() semantics -- after t[0]=m, mutating m
    #           must not change t[0]; the same source into two cells leaves the
    #           cells independent; for pcloud neither the source FloatTensor nor
    #           sibling cells are affected by a later write to one cell.
    # Observed today: cells share the source's shared_ptr buffer, so a later
    #                 write corrupts the cell / sibling cells / the source.

    # (1) mutate source AFTER assignment must not change the stored cell.
    t = mpcf.zeros((2,), dtype=mpcf.distmat64)
    m = DistanceMatrix(3)
    m[0, 1] = 2.0
    t[0] = m
    m[0, 2] = 8.0
    assert t[0][0, 2] == 0.0, f"aliased: cell changed to {t[0][0, 2]}"

    # (2) same source into two cells -> cells must be independent (distmat + symmat).
    for dt, make in [(mpcf.distmat64, lambda: DistanceMatrix(3)),
                     (mpcf.symmat64, lambda: SymmetricMatrix(3))]:
        tt = mpcf.zeros((2,), dtype=dt)
        shared = make()
        tt[0] = shared
        tt[1] = shared
        tt[0][0, 1] = 99.0
        assert tt[1][0, 1] == 0.0, f"{dt.__name__} aliased sibling: {tt[1][0, 1]}"

    # (3) pcloud: neither sibling cell nor the source FloatTensor may be aliased.
    pt = mpcf.zeros((2,), dtype=mpcf.pcloud64)
    src = mpcf.FloatTensor(np.zeros((3, 2)))
    pt[0] = src
    pt[1] = src
    pt[0][0, 0] = 99.0
    assert float(pt[1][0, 0]) == 0.0, "pcloud aliased sibling cell"
    assert float(src[0, 0]) == 0.0, "pcloud aliased source FloatTensor"

    # (4) slice assignment must propagate copies, not aliases.
    ts = mpcf.zeros((3,), dtype=mpcf.distmat64)
    srct = mpcf.zeros((2,), dtype=mpcf.distmat64)
    sm = DistanceMatrix(3)
    sm[0, 1] = 1.0
    srct[0] = sm
    srct[1] = sm
    ts[0:2] = srct
    ts[0][0, 1] = 77.0
    assert ts[1][0, 1] == 1.0, f"slice-assign aliased sibling: {ts[1][0, 1]}"


# ---------------------------------------------------------------------------
# Bug 39: pcloud slice/Ellipsis/partial-int ndarray assignment stores the WHOLE
#         array as one cloud in every selected cell (silent data corruption).
# ---------------------------------------------------------------------------
def test_pcloud_slice_array_assignment_distributes_per_cell():
    """ndarray assignment into a pcloud slice must distribute one cloud per cell."""
    # BUG: PointCloudTensor slice/Ellipsis/partial-int assignment of a numpy
    #      array stores the whole array as one cloud broadcast into every cell.
    # Expected: per the __setitem__ docstring and NumericTensor behavior, an
    #           ndarray RHS is distributed element-wise across selected cells, so
    #           t[:] = arr with arr=(2,4,2) gives t[0]==arr[0] and t[1]==arr[1].
    # Observed today: t[0] and t[1] both equal the whole (2,4,2) array.

    arr = np.arange(2 * 4 * 2, dtype=np.float64).reshape(2, 4, 2)
    t = mpcf.zeros((2,), dtype=mpcf.pcloud64)
    t[:] = arr
    assert tuple(t[0].shape) == (4, 2), f"got cell shape {tuple(t[0].shape)}"
    assert t[0].array_equal(arr[0]), "first cell did not receive arr[0]"
    assert t[1].array_equal(arr[1]), "second cell did not receive arr[1]"

    # Ellipsis on a 2D tensor of clouds: each cell gets its own (4,2) sub-cloud.
    arr2 = np.arange(3 * 5 * 4 * 2, dtype=np.float64).reshape(3, 5, 4, 2)
    t2 = mpcf.zeros((3, 5), dtype=mpcf.pcloud64)
    t2[...] = arr2
    assert tuple(t2[2, 4].shape) == (4, 2), f"got cell shape {tuple(t2[2, 4].shape)}"
    assert t2[2, 4].array_equal(arr2[2, 4]), "Ellipsis cell did not receive arr2[2,4]"

    # Partial-int target: t3[0] = arr (a (2,4,2) array) fills row 0's cells.
    t3 = mpcf.zeros((2, 2), dtype=mpcf.pcloud64)
    t3[0] = arr
    assert tuple(t3[0, 0].shape) == (4, 2), f"got cell shape {tuple(t3[0, 0].shape)}"
    assert t3[0, 0].array_equal(arr[0]), "partial-int cell did not receive arr[0]"
    assert t3[0, 1].array_equal(arr[1]), "partial-int cell did not receive arr[1]"


# ---------------------------------------------------------------------------
# Bug 40: pcloud advertises float/int as valid setitem dtypes but scalar
#         assignment slips past validation and raises a misleading
#         "Cannot create FloatTensor from <class float>".
# ---------------------------------------------------------------------------
def test_pcloud_scalar_assignment_raises_clean_consistent_error():
    """Advertised pcloud setitem dtypes must match what assignment accepts."""
    # BUG: PointCloudTensor._get_valid_setitem_dtypes() advertises float/int,
    #      so scalar assignment passes validation, then _decay_value calls
    #      FloatTensor(scalar) which raises a misleading constructor error.
    # Expected: internal consistency -- a scalar RHS either assigns cleanly or is
    #           rejected by validation with a Python-level message that does NOT
    #           reference the unrelated FloatTensor constructor.
    # Observed today: TypeError "Cannot create FloatTensor from <class 'float'>"
    #                 (resp. int / numpy.float64) -- a leaked internal message.

    t = mpcf.zeros((2,), dtype=mpcf.pcloud64)
    valid = t._get_valid_setitem_dtypes()

    for v in (3.14, 7, np.float64(3.0)):
        if float in valid or int in valid:
            # The validator advertises scalars as valid, so assignment must
            # actually succeed (consistency between advertisement and behavior).
            tt = mpcf.zeros((2,), dtype=mpcf.pcloud64)
            tt[0] = v  # must not raise
        else:
            # If scalars are NOT advertised, validation must reject them with a
            # clean message that does not leak the FloatTensor constructor error.
            with pytest.raises(TypeError) as excinfo:
                t[0] = v
            assert "Cannot create FloatTensor" not in str(excinfo.value), (
                f"leaked internal constructor error: {excinfo.value}")


# ---------------------------------------------------------------------------
# Bug 41: matrix-element tensor assignment with mismatched dtype leaks a raw
#         pybind "incompatible function arguments" diagnostic.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tensor_dt,make_src", [
    (mpcf.distmat64, lambda: DistanceMatrix(3, dtype=mpcf.float32)),
    (mpcf.symmat64, lambda: SymmetricMatrix(3, dtype=mpcf.float32)),
])
def test_matrix_element_dtype_mismatch_raises_clean_error(tensor_dt, make_src):
    """Cross-precision matrix-element assignment must raise a clean Python error."""
    # BUG: DistanceMatrix/SymmetricMatrix element assignment with mismatched
    #      dtype leaks a raw pybind11 "incompatible function arguments" message
    #      that exposes internal C++ type names.
    # Expected: a clear Python-level TypeError naming the dtype mismatch, not the
    #           raw binding diagnostic.
    # Observed today: TypeError whose message is the raw pybind dump
    #                 "_set_element(): incompatible function arguments. ...".

    t = mpcf.zeros((2,), dtype=tensor_dt)
    src = make_src()
    src[0, 1] = 3.0

    with pytest.raises(TypeError) as excinfo:
        t[0] = src
    msg = str(excinfo.value)
    assert "incompatible function arguments" not in msg, (
        f"leaked raw pybind diagnostic: {msg.splitlines()[0]}")
    assert "_set_element" not in msg, (
        f"leaked internal binding name: {msg.splitlines()[0]}")


# ---------------------------------------------------------------------------
# Bug 42: DistanceMatrix nonnegativity check (value < 0) lets NaN bypass the
#         documented "nonnegative" guarantee.
# ---------------------------------------------------------------------------
def test_distance_matrix_rejects_nan():
    """DistanceMatrix must reject NaN like it rejects negatives."""
    # BUG: the nonnegativity guard uses `value < 0`, which is False for NaN, so
    #      NaN slips through the documented "nonnegative entries" contract.
    # Expected: assigning NaN raises ValueError (NaN is not a valid nonnegative
    #           distance), the same way negatives and -inf are rejected.
    # Observed today: dm[0,1] = np.nan is accepted silently and reads back nan.

    dm = DistanceMatrix(3)
    # sanity: negatives already raise (pins the contract we extend to NaN).
    with pytest.raises(ValueError, match="nonnegative"):
        dm[0, 2] = -1.0

    with pytest.raises(ValueError, match="nonnegative"):
        dm[0, 1] = np.nan


# ---------------------------------------------------------------------------
# Bug 43: zeros(dtype=pcloud*) fresh cell reads back as a 0-d scalar 0.0 instead
#         of the documented "empty point cloud".
# ---------------------------------------------------------------------------
def test_pcloud_fresh_cell_is_empty_cloud_not_scalar():
    """A freshly-zeroed pcloud cell must read back as an empty 2-D cloud."""
    # BUG: zeros(dtype=pcloud*) cells default to a 0-d scalar 0.0 rather than the
    #      documented empty point cloud.
    # Expected: per the zeros() docstring ("an empty point cloud") and the sibling
    #           collection dtypes (symmat/distmat 0x0, barcode (0,2)), a fresh
    #           pcloud cell reads back as a 2-D empty array, matching what an
    #           explicitly-assigned empty cloud produces.
    # Observed today: shape Shape() (ndim 0), value scalar 0.0.

    t = mpcf.zeros((1,), dtype=mpcf.pcloud64)
    fresh = t[0]

    # An assigned empty (0, dim) cloud is the reference representation.
    t2 = mpcf.zeros((1,), dtype=mpcf.pcloud64)
    t2[0] = np.zeros((0, 2))

    assert fresh.ndim == 2, f"fresh pcloud cell has ndim {fresh.ndim}, expected 2"
    assert tuple(fresh.shape)[0] == 0, (
        f"fresh pcloud cell is not empty: shape {tuple(fresh.shape)}")
    assert fresh.array_equal(t2[0]), (
        "fresh pcloud cell does not match an explicitly-assigned empty cloud")
