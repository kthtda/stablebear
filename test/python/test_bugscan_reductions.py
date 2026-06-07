"""Red-until-fixed regression tests for known reduction bugs (bug scan).

These tests document *known, currently-unfixed* defects in ``masspcf``'s
reduction operations (``mean`` / ``max_time``). Each test asserts the CORRECT /
intended behavior, so it FAILS today and will pass once the underlying bug is
fixed. Do **not** weaken a test to make it green -- fix the bug instead. See
``bug-scan-findings.md`` at the repo root for the catalogue and root causes.

Some of these defects are hard crashes (SIGSEGV / SIGABRT) in the C++ backend.
Running such a repro in-process would abort the entire pytest session, so those
snippets run in a fresh subprocess (via ``_bugscan_support``) and we assert on
the *process outcome* instead.
"""

import numpy as np
import pytest

import masspcf as mpcf

from _bugscan_support import (
    assert_clean_raises,
    assert_no_hard_crash,
    run_snippet,
)


# ---------------------------------------------------------------------------
# Bug 23 + 25: out-of-range / negative ``dim`` in reductions is not validated.
#
# The Python wrapper (masspcf/reductions.py) passes ``dim`` straight to the C++
# backend, whose reduction (matrix_reduce.hpp) indexes/erases the shape vector
# with no bounds check. The single missing-validation root cause produces three
# distinct wrong behaviors:
#   * dim == ndim   -> silently returns a wrong-shaped tensor (exit 0)
#   * dim  > ndim   -> heap corruption -> SIGABRT (exit 134), uncatchable
#   * dim  < 0      -> raw pybind TypeError (size_t overload leak) instead of
#                      numpy-style negative-axis resolution
# numpy raises numpy.AxisError for an out-of-range axis and resolves negatives.
# ---------------------------------------------------------------------------


def test_mean_dim_equal_ndim_raises():
    """mean with dim == ndim must raise, not silently return a wrong shape."""
    # BUG: Out-of-range dim in reductions is unvalidated (dim == ndim)
    # Expected: an out-of-range axis raises a clean Python exception
    #           (numpy raises numpy.AxisError); never a silent wrong shape.
    # Observed today: returns a meaningless tensor of shape (3,), exit 0.
    A = mpcf.zeros((3, 4))
    with pytest.raises((IndexError, ValueError, np.exceptions.AxisError)):
        mpcf.mean(A, dim=2)


def test_max_time_dim_equal_ndim_raises():
    """max_time with dim == ndim must raise, not silently return a wrong shape."""
    # BUG: Out-of-range dim in reductions is unvalidated (dim == ndim)
    # Expected: an out-of-range axis raises a clean Python exception.
    # Observed today: returns a meaningless tensor of shape (3,), exit 0.
    A = mpcf.zeros((3, 4))
    with pytest.raises((IndexError, ValueError, np.exceptions.AxisError)):
        mpcf.max_time(A, dim=2)


def test_mean_dim_greater_than_ndim_raises_cleanly():
    """mean with dim > ndim must raise cleanly, not corrupt the heap / abort."""
    # BUG: Out-of-range dim in reductions is unvalidated (dim > ndim)
    # Expected: a clean, catchable Python exception (numpy.AxisError-equiv).
    # Observed today: glibc heap-corruption abort (SIGABRT, exit 134),
    #                 uncatchable from Python -- run in a subprocess.
    assert_clean_raises(
        """
        A = mpcf.zeros((3, 4))
        mpcf.mean(A, dim=5)
        """,
        exc=("IndexError", "ValueError", "AxisError"),
    )


def test_max_time_dim_greater_than_ndim_raises_cleanly():
    """max_time with dim > ndim must raise cleanly, not corrupt the heap / abort."""
    # BUG: Out-of-range dim in reductions is unvalidated (dim > ndim)
    # Expected: a clean, catchable Python exception (numpy.AxisError-equiv).
    # Observed today: glibc heap-corruption abort (SIGABRT, exit 134),
    #                 uncatchable from Python -- run in a subprocess.
    assert_clean_raises(
        """
        A = mpcf.zeros((3, 4))
        mpcf.max_time(A, dim=5)
        """,
        exc=("IndexError", "ValueError", "AxisError"),
    )


def test_mean_negative_dim_resolves_like_numpy():
    """mean(A, dim=-1) must reduce the last axis (numpy semantics), not error."""
    # BUG: negative dim raises a misleading pybind TypeError (overload leak)
    # Expected: dim=-1 reduces the last axis, matching numpy and the rest of the
    #           numpy-like tensor surface (A[-1], stack(axis=-1) both work):
    #           numpy: np.zeros((3,4)).mean(axis=-1).shape == (3,)
    # Observed today: TypeError "mean(): incompatible function arguments"
    #                 because the C++ binding's dim is size_t (unsigned).
    A = mpcf.zeros((3, 4))
    r = mpcf.mean(A, dim=-1)
    assert tuple(r.shape) == (3,)


def test_max_time_negative_dim_resolves_like_numpy():
    """max_time(A, dim=-1) must reduce the last axis (numpy semantics), not error."""
    # BUG: negative dim raises a misleading pybind TypeError (overload leak)
    # Expected: dim=-1 reduces the last axis -> shape (3,) for a (3,4) tensor.
    # Observed today: TypeError "max_time(): incompatible function arguments".
    A = mpcf.zeros((3, 4))
    r = mpcf.max_time(A, dim=-1)
    assert tuple(r.shape) == (3,)


# ---------------------------------------------------------------------------
# Bug 24: max_time segfaults on a size-0 reduced axis, while mean does not.
#
# max_element (matrix_reduce.hpp) dereferences begin() / forms begin()+1 on an
# empty tmp vector when the reduced dimension has size 0 -- undefined behavior.
# mean routes through reduce() seeded with a default TPcf() and survives. The
# precise correct value for an empty max is opinionated (0.0 / NaN / raise), so
# we assert the weakest defensible contracts: (a) it must never segfault, and
# (b) it must be consistent with mean (which returns a valid reduced tensor).
# ---------------------------------------------------------------------------


def test_max_time_empty_reduced_axis_does_not_crash():
    """max_time over a size-0 reduced axis must not segfault the interpreter."""
    # BUG: max_time segfaults on a size-0 reduced axis (empty inner dimension)
    # Expected: a valid Python call never segfaults -- return a well-defined
    #           result or raise a clean exception, consistent with mean.
    # Observed today: Segmentation fault (core dumped), exit 139 -- subprocess.
    assert_no_hard_crash(
        """
        mpcf.max_time(mpcf.zeros((0,)), dim=0)
        """
    )
    assert_no_hard_crash(
        """
        mpcf.max_time(mpcf.zeros((3, 0)), dim=1)
        """
    )


def test_max_time_empty_reduced_axis_consistent_with_mean():
    """max_time over an empty reduced axis must behave consistently with mean.

    mean(zeros((3,0)), dim=1) returns cleanly with shape (3,); max_time on the
    identical empty reduction should likewise yield a valid reduced result of
    shape (3,) rather than crashing.
    """
    # BUG: max_time segfaults on a size-0 reduced axis (empty inner dimension)
    # Expected: max_time(zeros((3,0)), dim=1) returns a valid tensor of shape
    #           (3,), matching mean's clean handling of the same empty reduce.
    # Observed today: Segmentation fault (core dumped), exit 139 -- subprocess.
    cp = run_snippet(
        """
        r = mpcf.max_time(mpcf.zeros((3, 0)), dim=1)
        assert tuple(r.shape) == (3,), tuple(r.shape)
        print('BUGSCAN_OK')
        """
    )
    assert cp.returncode == 0 and "BUGSCAN_OK" in cp.stdout, (
        f"returncode={cp.returncode}\n--- stdout ---\n{cp.stdout}\n"
        f"--- stderr (tail) ---\n{cp.stderr[-2000:]}"
    )
