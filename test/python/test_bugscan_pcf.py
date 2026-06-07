#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Red-until-fixed regression tests for KNOWN, currently-unfixed PCF bugs.

Each test below documents a confirmed defect found by the API bug scan (area
``pcf``; see ``bug-scan-findings.md`` at the repo root for the full catalogue and
root-cause hints). Every test asserts the CORRECT / intended behavior, so it
FAILS TODAY and will turn green only once the underlying bug is fixed. Do NOT
weaken these tests to make them pass -- fix the bug instead.

Some defects are flagged as potential hard crashers in the C++ backend. To keep
a single failing test from aborting the whole pytest session, the riskier paths
are exercised through the subprocess helpers in ``_bugscan_support``; the
remainder run in-process where a failed assertion fails cleanly.
"""

import warnings

import numpy as np
import pytest

import masspcf as mpcf
from masspcf.functional import Pcf

from _bugscan_support import assert_clean_raises


# ---------------------------------------------------------------------------
# Bug 10: negative time survives construction (t0 != 0) and bypasses the
#         negative-time evaluation guard.
# ---------------------------------------------------------------------------
def test_negative_breakpoint_pcf_never_evaluates_before_zero():
    """A PCF whose min time is < 0 must be rejected, or at least never eval at t<0.

    The constructor sorts breakpoints *after* validating ``t0`` on the unsorted
    input, so a negative time can be sorted to the front, yielding a PCF with
    ``t0 = -1.0`` that the documented ``t0 == 0`` invariant forbids. The
    evaluation guard then compares against ``front().t`` (the bogus t0) instead
    of 0, so the PCF returns values at negative times.
    """
    # BUG: Pcf construction validates t0 on the unsorted row 0, then std::sort
    #      can move a negative time to the front -> t0 < 0 with no error, and
    #      eval guards against t0 rather than 0.
    # Expected: either construction raises ValueError (min time must be 0), or
    #           at minimum the resulting PCF raises ValueError when evaluated at
    #           a negative time (scalar and array paths alike).
    # Observed today: construction succeeds storing t0=-1.0, and f(-0.5) returns
    #                 99.0 / f([-0.5]) returns [99.0] -- no error at all.
    #
    # Run through a subprocess (flagged crasher) and require a clean ValueError
    # to surface somewhere along construct->scalar-eval->array-eval. Today none
    # of those raise, so the subprocess reports BUGSCAN_NORAISE and this fails.
    assert_clean_raises(
        """
        f = Pcf(np.array([[0.0, 1.0], [2.0, 2.0], [-1.0, 99.0]], dtype=np.float64))
        f(-0.5)
        f(np.array([-0.5]))
        """,
        ("ValueError",),
    )


def test_negative_breakpoint_int_pcf_never_evaluates_before_zero():
    """The same negative-t0 defect must not let an int PCF evaluate at t<0 either.

    The constructor / eval path is shared across dtypes, so the int-typed PCF
    exhibits the identical hole.
    """
    # BUG: same negative-t0 construction/eval hole, int dtype.
    # Expected: ValueError on construction or on negative-time evaluation.
    # Observed today: no error; the malformed int PCF returns a value at t<0.
    assert_clean_raises(
        """
        f = Pcf(np.array([[0, 1], [2, 2], [-1, 99]], dtype=np.int64))
        f(-1)
        """,
        ("ValueError",),
    )


# ---------------------------------------------------------------------------
# Bug 11: __pow__ raises a meaningless SystemError instead of propagating the
#         RuntimeWarning when warnings are configured as errors.
# ---------------------------------------------------------------------------
def test_pcf_pow_under_warnings_as_errors_raises_clean_runtimewarning():
    """Pcf.__pow__ must surface the documented RuntimeWarning cleanly under -W error.

    Under ``warnings.simplefilter('error')`` (== ``python -W error`` /
    ``filterwarnings=error``) a fractional power of a negative-valued PCF should
    raise the documented ``RuntimeWarning`` as a normal exception. Instead the
    binding ignores ``PyErr_WarnEx``'s -1 return and builds a return value with a
    pending exception, which CPython reports as a meaningless ``SystemError``.
    """
    # BUG: py_pcf.cpp __pow__ ignores PyErr_WarnEx's -1 return code; under
    #      warnings-as-errors it returns a value with a pending exception set.
    # Expected: a clean RuntimeWarning (escalated to an exception) is raised.
    # Observed today: SystemError "... returned a result with an exception set".
    f = Pcf(np.array([[0.0, -4.0], [1.0, 9.0]], dtype=np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The documented contract: the RuntimeWarning becomes a clean exception.
        # Today this raises SystemError instead, which pytest.raises(RuntimeWarning)
        # does NOT catch -> the SystemError propagates and fails the test cleanly.
        with pytest.raises(RuntimeWarning):
            f ** 0.5


def test_tensor_pow_under_warnings_as_errors_raises_clean_runtimewarning():
    """The tensor __pow__ path shares the same ignored-return-code defect.

    ``py_tensor.hpp`` contains the identical ``PyErr_WarnEx`` pattern, so a
    negative-base fractional power on a tensor also yields ``SystemError`` under
    warnings-as-errors instead of the documented ``RuntimeWarning``.
    """
    # BUG: py_tensor.hpp __pow__ ignores PyErr_WarnEx's -1 return code.
    # Expected: a clean RuntimeWarning is raised under -W error.
    # Observed today: SystemError "... returned a result with an exception set".
    t = mpcf.FloatTensor(np.array([-4.0, 9.0], dtype=np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning):
            t ** 0.5


# ---------------------------------------------------------------------------
# Bug 12: iterate_rectangles fails on integer PCFs with default arguments.
# ---------------------------------------------------------------------------
def test_iterate_rectangles_int_pcf_default_args_succeeds():
    """iterate_rectangles must work out of the box for int PCFs (advertised dtype).

    int32/int64 are advertised public dtypes routed through ``_BACKEND_MAP``, and
    the docstring imposes no dtype restriction, so the documented default call
    ``iterate_rectangles(f, g)`` should succeed for them. The Python wrapper
    hard-codes float defaults (``a=0.0``, ``b=float('inf')``) that pybind11
    cannot bind to the int-typed C++ overload.
    """
    # BUG: iterate_rectangles hard-codes float defaults a=0.0/b=inf; int PCFs
    #      route to an int-typed C++ overload that rejects the float defaults.
    # Expected: default-arg call on int PCFs returns the correct int rectangles,
    #           matching the explicit-int-bounds result.
    # Observed today: TypeError "incompatible function arguments" on the default
    #                 call (and on b-only / b=inf), only explicit int a,b works.
    fi = Pcf(np.array([[0, 3], [1, 2]], dtype=np.int32))
    gi = Pcf(np.array([[0, 2], [1, 4]], dtype=np.int32))

    # Reference: the documented/working result with explicit int bounds.
    ref = mpcf.iterate_rectangles(fi, gi, a=0, b=10)
    ref_tuples = [(x.left, x.right, x.f_value, x.g_value) for x in ref]
    assert ref_tuples == [(0, 1, 3, 2), (1, 10, 2, 4)]

    # The default-arg call must SUCCEED (today it raises TypeError before
    # producing anything). The exact int representation of "to infinity" for
    # the trailing rectangle's `right` is implementation-defined, so assert only
    # the structurally-determined parts: two rectangles, matching f/g step
    # values and left edges, with the last one extending past its left edge.
    r = mpcf.iterate_rectangles(fi, gi)
    got = [(x.left, x.right, x.f_value, x.g_value) for x in r]
    assert len(got) == 2
    assert got[0] == (0, 1, 3, 2)
    left, right, fv, gv = got[1]
    assert (left, fv, gv) == (1, 2, 4)
    assert right > left


# ---------------------------------------------------------------------------
# Bug 13: scalar vs array Pcf.__call__ disagree on NaN, and the array path is
#         position-dependent.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
def test_pcf_call_nan_scalar_array_agree(dtype):
    """Scalar and 1-element-array evaluation must agree on the same NaN query.

    The docstring states a scalar and a 1-element array are the same evaluation,
    differing only in container. Two separate C++ ``evaluate`` overloads handle
    NaN differently: the scalar path falls through to the last value, the array
    path to the first (or wherever the shared cursor was left).
    """
    # BUG: scalar evaluate vs array evaluate disagree on NaN (different C++ impls).
    # Expected: f(nan) == f([nan])[0] -- identical input yields identical result.
    # Observed today: scalar nan -> 0.5 (last value), array nan -> 1.0 (first).
    f = Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.5]], dtype=dtype))
    s = f(float("nan"))
    a = f(np.array([np.nan], dtype=dtype))[0]
    assert s == a, f"scalar/array NaN eval disagree: {s!r} vs {a!r}"


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
def test_pcf_call_nan_array_not_position_dependent(dtype):
    """A NaN slot's result must not depend on neighboring query values.

    The array evaluator threads one mutable cursor across elements; for a NaN it
    never advances, so the NaN slot returns whatever value the previous element
    left the cursor at -- making the result non-reproducible.
    """
    # BUG: array evaluate carries a shared cursor; NaN comparisons never advance
    #      it, so the NaN slot's value depends on the preceding element.
    # Expected: the NaN slot yields the same value regardless of its neighbors,
    #           i.e. matches the lone-NaN result (== the scalar/array convention).
    # Observed today: f([2, nan, 4]) -> [2, 2, 0.5]; the NaN slot is 2.0, not the
    #                 1.0 that f([nan]) produces -- position-dependent.
    f = Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.5]], dtype=dtype))
    lone_nan = f(np.array([np.nan], dtype=dtype))[0]
    surrounded = f(np.array([2.0, np.nan, 4.0], dtype=dtype))[1]
    assert surrounded == lone_nan, (
        f"NaN slot is position-dependent: {surrounded!r} surrounded "
        f"vs {lone_nan!r} alone"
    )
