"""Red-until-fixed regression tests for KNOWN, unfixed interop/tensor-protocol bugs.

These tests come from the broad API bug scan (see ``bug-scan-findings.md`` at the
repo root for the catalogue, root-cause hints, and severities). Each test asserts
the CORRECT / intended behavior, so it FAILS today and will turn green only once
the underlying defect is fixed. Do NOT weaken a test to make it pass -- fix the
bug instead.

One defect in this area hard-crashes the interpreter (SIGSEGV on reducing/plotting
an empty PcfTensor). pytest runs the whole suite in a single process, so that
case is exercised ONLY through the subprocess crash-isolation helpers in
``_bugscan_support`` -- never called directly in a test body. The remaining
defects raise clean exceptions or return wrong values, so they run in-process.
"""
import numpy as np
import pytest

import masspcf as mpcf

from _bugscan_support import assert_no_hard_crash


# --------------------------------------------------------------------------
# Bug 45: bool() on a numeric tensor reports length, not value truthiness.
# --------------------------------------------------------------------------
def test_bool_numeric_tensor_matches_numpy_truthiness():
    """bool() on FloatTensor/IntTensor must use value truthiness, not __len__."""
    # BUG: bool() on FloatTensor/IntTensor falls back to __len__ (length, not value)
    # Expected: NumPy semantics -- size-1 returns element truthiness; multi-element
    #           raises ValueError("...more than one element is ambiguous"); a 0-d
    #           tensor evaluates its single element instead of raising on len().
    # Observed today: every non-empty numeric tensor is truthy (length>0); a 0-d
    #           tensor raises TypeError("len() of unsized object").

    # size-1 zero -> False (currently True, because len==1 is truthy)
    assert bool(mpcf.IntTensor(np.array([0]))) is False
    assert bool(mpcf.FloatTensor(np.array([0.0]))) is False
    # size-1 nonzero -> True
    assert bool(mpcf.FloatTensor(np.array([5.0]))) is True

    # multi-element -> ValueError like numpy (currently returns True silently)
    with pytest.raises(ValueError, match="more than one element"):
        bool(mpcf.FloatTensor(np.array([0.0, 0.0])))

    # 0-d tensor -> evaluate the single element (currently raises
    # TypeError: len() of unsized object)
    assert bool(mpcf.FloatTensor(np.array(0.0))) is False
    assert bool(mpcf.FloatTensor(np.array(7.0))) is True


# --------------------------------------------------------------------------
# Bug 46: max_time() / plotting.plot() segfault on an empty PcfTensor.
#   HARD CRASH (SIGSEGV, exit 139) -- must run via subprocess helper.
# --------------------------------------------------------------------------
def test_max_time_empty_pcf_tensor_no_segfault():
    """reductions.max_time on an empty PcfTensor must not segfault the process."""
    # BUG: max_time() segfaults on empty PcfTensor (max_element derefs empty iterator)
    # Expected: a clean Python error (e.g. ValueError) or a graceful empty result.
    # Observed today: SIGSEGV (shell exit 139, core dumped), no Python traceback.
    assert_no_hard_crash(
        """
        from masspcf.reductions import max_time
        try:
            max_time(mpcf.PcfTensor([]))
        except Exception as e:
            print("clean exception:", type(e).__name__)
        """
    )


def test_plot_empty_pcf_tensor_no_segfault():
    """plotting.plot on an empty PcfTensor must not segfault the process."""
    # BUG: plotting.plot() segfaults on empty PcfTensor (reaches max_time internally)
    # Expected: a clean Python error or a graceful empty/no-op plot.
    # Observed today: SIGSEGV (shell exit 139, core dumped) via plotting.py:66.
    assert_no_hard_crash(
        """
        import matplotlib; matplotlib.use("Agg")
        import masspcf.plotting as plotting
        try:
            plotting.plot(mpcf.PcfTensor([]))
        except Exception as e:
            print("clean exception:", type(e).__name__)
        """
    )


# --------------------------------------------------------------------------
# Bug 47: == / != against a Python/NumPy scalar silently return a plain bool,
#         corrupting the documented t[t == scalar] masking idiom.
# --------------------------------------------------------------------------
def test_eq_scalar_returns_bool_tensor_and_masks():
    """== against a scalar must return an elementwise BoolTensor, not a Python bool."""
    # BUG: tensor == scalar returns a plain Python bool (identity fallback)
    # Expected: elementwise BoolTensor via scalar broadcasting, so masking works:
    #           FloatTensor([1,2,3]) == 2.0 -> BoolTensor([F,T,F]); a[a==2.0] -> [2.0].
    # Observed today: (a == 2.0) is the Python bool False; a[a==2.0] -> empty tensor.
    a = mpcf.FloatTensor(np.array([1.0, 2.0, 3.0]))

    r = a == 2.0
    assert isinstance(r, mpcf.BoolTensor), (
        f"expected BoolTensor, got {type(r).__name__}={r!r}")
    np.testing.assert_array_equal(
        np.asarray(r), np.array([1.0, 2.0, 3.0]) == 2.0)

    rn = a != 2.0
    assert isinstance(rn, mpcf.BoolTensor), (
        f"expected BoolTensor, got {type(rn).__name__}={rn!r}")

    # Documented masking idiom must select the matching element.
    np.testing.assert_array_equal(np.asarray(a[a == 2.0]), np.array([2.0]))


# --------------------------------------------------------------------------
# Bug 47 + Bug 48: ordered comparisons (<, <=, >, >=) against a scalar leak a
#         raw AttributeError about the internal '_data' attribute.
# --------------------------------------------------------------------------
def test_ordered_comparison_scalar_returns_bool_tensor():
    """< <= > >= against a scalar must broadcast to a BoolTensor, not leak _data."""
    # BUG: ordered ops against a scalar raise AttributeError: 'float' object has
    #      no attribute '_data' (unconditional rhs._data access, no promotion)
    # Expected: elementwise BoolTensor via scalar broadcasting, per docs/arithmetic.rst.
    # Observed today: AttributeError leaking the private '_data' attribute name.
    a = mpcf.FloatTensor(np.array([1.0, 2.0, 3.0]))
    ref = np.array([1.0, 2.0, 3.0])

    for op, npop in [
        (lambda x: x < 2.0, lambda x: x < 2.0),
        (lambda x: x <= 2.0, lambda x: x <= 2.0),
        (lambda x: x > 2.0, lambda x: x > 2.0),
        (lambda x: x >= 2.0, lambda x: x >= 2.0),
    ]:
        r = op(a)
        assert isinstance(r, mpcf.BoolTensor), (
            f"expected BoolTensor, got {type(r).__name__}={r!r}")
        np.testing.assert_array_equal(np.asarray(r), npop(ref))


def test_ordered_comparison_ndarray_no_leaked_attribute_error():
    """< against an ndarray must not leak an internal '_data' AttributeError."""
    # BUG: ordered ops against an ndarray raise AttributeError about '_data'
    # Expected: a usable result (BoolTensor, like ==) or at minimum a clean
    #           TypeError -- never a leaked internal '_data' attribute error.
    # Observed today: AttributeError: 'numpy.ndarray' object has no attribute '_data'.
    a = mpcf.FloatTensor(np.array([1.0, 2.0, 3.0]))
    rhs = np.array([2.0, 2.0, 2.0])
    try:
        result = a < rhs
    except AttributeError as e:
        pytest.fail(f"leaked internal AttributeError on ndarray RHS: {e}")
    except TypeError:
        # A clean TypeError is an acceptable minimum contract.
        return
    # If it produced a result, it should be the correct elementwise BoolTensor.
    assert isinstance(result, mpcf.BoolTensor), (
        f"expected BoolTensor, got {type(result).__name__}={result!r}")
    np.testing.assert_array_equal(
        np.asarray(result), np.array([1.0, 2.0, 3.0]) < rhs)


# --------------------------------------------------------------------------
# Bug 49: cross-dtype comparison / array_equal leak the pybind11 overload table.
# --------------------------------------------------------------------------
def test_cross_dtype_comparison_no_pybind_signature_leak():
    """Cross-dtype == / array_equal must not dump the raw pybind11 overload table."""
    # BUG: float-vs-int (and int32-vs-int64, float32-vs-float64) == / array_equal
    #      raise a TypeError dumping internal C++ pybind11 type names
    # Expected: a clean user-facing result (cast/promote to a common dtype, NumPy
    #           parity: float==int -> [True,True,True]) or a clean error WITHOUT
    #           leaking C++ overload signatures / internal type names.
    # Observed today: TypeError containing "incompatible function arguments" and
    #           internal names like "masspcf._mpcf_cuda12.Float64Tensor".
    f = mpcf.FloatTensor(np.array([1.0, 2.0, 3.0]))
    i = mpcf.IntTensor(np.array([1, 2, 3]))

    try:
        r = f == i
    except TypeError as e:
        msg = str(e)
        assert "incompatible function arguments" not in msg, (
            f"leaked pybind11 overload table: {msg}")
        assert "Tensor" not in msg or "masspcf._mpcf" not in msg, (
            f"leaked internal C++ type names: {msg}")
    else:
        assert isinstance(r, mpcf.BoolTensor), (
            f"expected BoolTensor, got {type(r).__name__}={r!r}")
        np.testing.assert_array_equal(
            np.asarray(r), np.array([1.0, 2.0, 3.0]) == np.array([1, 2, 3]))

    try:
        ae = f.array_equal(i)
    except TypeError as e:
        msg = str(e)
        assert "incompatible function arguments" not in msg, (
            f"leaked pybind11 overload table from array_equal: {msg}")
    else:
        assert ae == np.array_equal(
            np.array([1.0, 2.0, 3.0]), np.array([1, 2, 3]))


# --------------------------------------------------------------------------
# Bug 50: cross-dtype assignment / astype fail for the entire float<->uint quadrant.
# --------------------------------------------------------------------------
def test_float_uint_cross_cast_assignment_and_astype():
    """float<->uint assignment and astype must cast like NumPy (both directions)."""
    # BUG: float[:] = uint and uint[:] = float (and astype both ways) raise
    #      TypeError "Cannot cast from uint32 to float64" (missing C++ casts)
    # Expected: cast like NumPy -- float[:]=uint -> [1.,2.,3.]; uint[:]=float -> [1,2,3];
    #           astype works in both directions and both precisions.
    # Observed today: TypeError "Cannot cast from uint32 to float64" / float64 to uint32.

    # float[:] = uint
    ft = mpcf.FloatTensor(np.zeros(3))
    ut = mpcf.IntTensor(np.array([1, 2, 3], dtype=np.uint32))
    ft[:] = ut
    np.testing.assert_array_equal(np.asarray(ft), np.array([1.0, 2.0, 3.0]))

    # uint[:] = float
    ut2 = mpcf.IntTensor(np.array([0, 0, 0], dtype=np.uint32))
    ft2 = mpcf.FloatTensor(np.array([1.5, 2.5, 3.5]))
    ut2[:] = ft2
    np.testing.assert_array_equal(np.asarray(ut2), np.array([1, 2, 3]))

    # Direct astype both directions
    np.testing.assert_array_equal(
        np.asarray(ut.astype(mpcf.float64)), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(
        np.asarray(ft2.astype(mpcf.uint32)), np.array([1, 2, 3]))


# --------------------------------------------------------------------------
# Bug 51: allclose breaks reflexivity for +inf and gives false positives for
#         opposite-sign infinities.
# --------------------------------------------------------------------------
def test_allclose_infinity_matches_numpy():
    """allclose must treat matching infinities as close and opposite-sign as not."""
    # BUG: allclose(x,x) is False when x has +inf; allclose([inf],[-inf]) is True
    # Expected: NumPy semantics -- matching infinities compare close (reflexivity
    #           holds), opposite-sign infinities compare not-close.
    # Observed today: allclose(x,x)->False (|inf-inf|=nan); allclose([inf],[-inf])->True.
    x = mpcf.FloatTensor(np.array([1.0, np.inf, 3.0]))
    # reflexivity must hold for inf-containing tensors (numpy: True)
    assert bool(mpcf.allclose(x, x)) == bool(
        np.allclose([1.0, np.inf, 3.0], [1.0, np.inf, 3.0]))

    xp = mpcf.FloatTensor(np.array([np.inf]))
    xm = mpcf.FloatTensor(np.array([-np.inf]))
    assert bool(mpcf.allclose(xp, xm)) == bool(
        np.allclose([np.inf], [-np.inf]))  # numpy: False


# --------------------------------------------------------------------------
# Bug 52: allclose silently returns False on shape mismatch instead of
#         broadcasting (when compatible) or raising (when incompatible).
# --------------------------------------------------------------------------
def test_allclose_broadcasts_or_raises_on_shape_mismatch():
    """allclose must broadcast compatible shapes and raise on incompatible ones."""
    # BUG: allclose returns False for ANY shape mismatch (short-circuit before
    #      comparing), inconsistent with the library's own broadcasting in + / ==
    # Expected: broadcast-compatible shapes -> compare elementwise (so (3,) vs
    #           (1,3) of equal values is True); truly incompatible -> ValueError.
    # Observed today: both cases silently return False.
    a = mpcf.FloatTensor(np.array([1.0, 2.0, 3.0]))       # (3,)
    e = mpcf.FloatTensor(np.array([[1.0, 2.0, 3.0]]))     # (1,3) broadcast-equal
    b = mpcf.FloatTensor(np.array([1.0, 2.0]))            # (2,) incompatible

    # broadcast-compatible, equal values -> True
    assert bool(mpcf.allclose(a, e)) is True

    # truly incompatible -> ValueError, not a silent False
    with pytest.raises(ValueError):
        mpcf.allclose(a, b)
