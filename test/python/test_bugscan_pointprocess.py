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

"""Red-until-fixed regression tests for the ``pointprocess`` bug-scan area.

These tests document *known, currently-unfixed* defects in the random
generators (``masspcf.random.noisy_sin`` / ``noisy_cos``) and the Poisson
point-process sampler (``masspcf.point_process.sample_poisson``). Each test
asserts the CORRECT / intended behavior, so it FAILS today (red) and will pass
once the underlying bug is fixed. Do **not** weaken a test to make it green --
fix the bug instead. See ``bug-scan-findings.md`` at the repo root for the full
catalogue and root-cause hints.

Several of these defects are hard interpreter crashes (SIGSEGV) in the shared
C++ generation path. Running such a repro in-process would abort the whole
pytest session, so those snippets run in a fresh subprocess via the helpers in
``_bugscan_support`` and we assert on the *process outcome* instead. The
non-crashing wrong-value / missing-validation bugs run normally in-process.
"""

import numpy as np
import pytest

import masspcf as mpcf
from masspcf.point_process import sample_poisson

from _bugscan_support import (
    assert_clean_raises,
    run_snippet,
)


# ---------------------------------------------------------------------------
# Bug 30: noisy_sin / noisy_cos segfault when n_points == 0 (hard crash).
# ---------------------------------------------------------------------------
def test_noisy_trig_npoints_zero_raises_clean_error():
    """noisy_sin/noisy_cos with n_points=0 must not SIGSEGV the interpreter."""
    # BUG: noisy_sin / noisy_cos segfault (SIGSEGV) when n_points=0
    # Expected: a clean Python error (ValueError "n_points must be >= 1" or a
    #           TypeError), or a valid degenerate tensor -- never a crash.
    # Observed today: "Segmentation fault (core dumped)", shell exit 139, no
    #           traceback. randomTs.front()/pts.back() on an empty vector in
    #           include/mpcf/random.hpp noisy_function() is UB.
    assert_clean_raises(
        """
        system.force_cpu(True)
        try:
            random.noisy_sin((1,), n_points=0)
        finally:
            system.force_cpu(False)
        """,
        ("ValueError", "TypeError"),
    )
    assert_clean_raises(
        """
        system.force_cpu(True)
        try:
            random.noisy_cos((1,), n_points=0)
        finally:
            system.force_cpu(False)
        """,
        ("ValueError", "TypeError"),
    )


# ---------------------------------------------------------------------------
# Bugs 31 & 32: sample_poisson segfaults on non-finite rate / lo / hi.
# Same root cause: a non-finite mean (lambda = rate * volume = inf) reaches
# std::poisson_distribution, which is undefined behavior -> SIGSEGV.
# ---------------------------------------------------------------------------
def test_sample_poisson_nonfinite_inputs_raise_clean_error():
    """Non-finite rate/lo/hi must raise a clean ValueError, not SIGSEGV."""
    # BUG: sample_poisson segfaults (SIGSEGV) on non-finite rate or lo/hi
    # Expected: a clean ValueError rejecting the non-finite input, matching the
    #           existing lo<=hi / length validation in this same function.
    # Observed today: "Segmentation fault (core dumped)", shell exit 139, no
    #           traceback, for rate=inf, hi=[inf, 1.0] and lo=[-inf, 0.0].

    # rate = +inf
    assert_clean_raises(
        """
        system.force_cpu(True)
        try:
            poisson.sample_poisson((1,), dim=2, rate=float('inf'),
                                   generator=mpcf.random.Generator(1))
        finally:
            system.force_cpu(False)
        """,
        "ValueError",
    )
    # hi = +inf (volume -> inf -> lambda -> inf)
    assert_clean_raises(
        """
        system.force_cpu(True)
        try:
            poisson.sample_poisson((1,), dim=2, lo=[0.0, 0.0],
                                   hi=[float('inf'), 1.0],
                                   generator=mpcf.random.Generator(1))
        finally:
            system.force_cpu(False)
        """,
        "ValueError",
    )
    # lo = -inf (volume -> inf -> lambda -> inf)
    assert_clean_raises(
        """
        system.force_cpu(True)
        try:
            poisson.sample_poisson((1,), dim=2, lo=[-float('inf'), 0.0],
                                   hi=[1.0, 1.0],
                                   generator=mpcf.random.Generator(1))
        finally:
            system.force_cpu(False)
        """,
        "ValueError",
    )


# ---------------------------------------------------------------------------
# Bug 33: sample_poisson silently drops elements when the per-element
# allocation fails (std::bad_alloc swallowed by parallel_walk's Future::wait).
# ---------------------------------------------------------------------------
def test_sample_poisson_alloc_failure_is_surfaced_not_silently_dropped():
    """An unsatisfiable Poisson draw must raise, not silently return zeros."""
    # BUG: sample_poisson silently drops elements when the per-element
    #      allocation fails (bad_alloc swallowed by parallel_walk's
    #      Future::wait instead of .get()).
    # Expected: surface the failure as a Python exception (MemoryError /
    #           RuntimeError); the failed draw must NEVER be silently
    #           substituted with a default-constructed empty element.
    # Observed today: exit 0, no exception, every element comes back as a
    #           default-initialized 0-d (empty) element -> silent data loss.
    #
    # rate=1e10 over the unit square => mean ~1e10 points => ~160GB of
    # doubles. The allocation throws std::bad_alloc (it never commits memory,
    # so no OOM kill), which is the swallowed path under test.
    cp = run_snippet(
        """
        system.force_cpu(True)
        try:
            A = poisson.sample_poisson((3,), dim=2, rate=1e10,
                                       generator=mpcf.random.Generator(1))
        except (MemoryError, RuntimeError) as e:
            print('BUGSCAN_RAISED:' + type(e).__name__)
        else:
            sizes = [int(np.asarray(A[i]).size) for i in range(3)]
            ndims = [int(np.asarray(A[i]).ndim) for i in range(3)]
            # A correctly generated (N, 2) cloud is 2-d with size > 1; a
            # silently-dropped element is the default-constructed 0-d scalar.
            dropped = any(nd < 2 or sz <= 1 for nd, sz in zip(ndims, sizes))
            if dropped:
                print('BUGSCAN_SILENT_DROP:' + repr(list(zip(ndims, sizes))))
            else:
                print('BUGSCAN_OK')
        finally:
            system.force_cpu(False)
        """
    )
    assert cp.returncode >= 0, (
        f"interpreter hard-crashed:\nreturncode={cp.returncode}\n"
        f"--- stdout ---\n{cp.stdout}\n--- stderr (tail) ---\n{cp.stderr[-2000:]}"
    )
    assert "BUGSCAN_SILENT_DROP" not in cp.stdout, (
        "sample_poisson silently dropped elements on allocation failure "
        "instead of raising; output:\n"
        f"returncode={cp.returncode}\n--- stdout ---\n{cp.stdout}\n"
        f"--- stderr (tail) ---\n{cp.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Bug 34: noisy_sin / noisy_cos / sample_poisson on a 0-d shape () silently
# return an unfilled (all-zeros / default) element instead of generating it.
# Non-crashing (silent wrong result) -> normal in-process test.
# ---------------------------------------------------------------------------
def test_zero_dim_shape_generators_fill_their_single_element():
    """0-d generation must fill its single element, not return default zeros."""
    # BUG: noisy_sin/noisy_cos/sample_poisson on 0-d shape () silently return
    #      unfilled zeros (the single element is never generated).
    # Expected: a 0-d tensor is a valid 1-element tensor (size==1); its single
    #           element must be filled with generated data, exactly as for the
    #           identically-sized shape (1,) -- so the result must NOT be
    #           array_equal to the corresponding zeros tensor.
    # Observed today: array_equal(result, zeros) is True for all three
    #           generators on shape () (the walk skips the lone element).
    mpcf.system.force_cpu(True)
    try:
        zeros_pcf = mpcf.zeros((), dtype=mpcf.pcf32)
        assert zeros_pcf.size == 1  # 0-d is a genuine 1-element tensor

        mpcf.random.seed(123)
        a_sin = mpcf.random.noisy_sin((), n_points=6)
        assert not a_sin.array_equal(zeros_pcf), (
            "0-d noisy_sin returned all-zeros (single element never generated)"
        )

        mpcf.random.seed(123)
        a_cos = mpcf.random.noisy_cos((), n_points=6)
        assert not a_cos.array_equal(zeros_pcf), (
            "0-d noisy_cos returned all-zeros (single element never generated)"
        )

        zeros_pcloud = mpcf.zeros((), dtype=mpcf.pcloud64)
        mpcf.random.seed(123)
        p = sample_poisson((), dim=2, rate=5.0)
        assert not p.array_equal(zeros_pcloud), (
            "0-d sample_poisson returned the empty default cloud "
            "(single element never generated)"
        )
    finally:
        mpcf.system.force_cpu(False)


# ---------------------------------------------------------------------------
# Bug 35: sample_poisson accepts a negative rate with no validation, passing a
# negative mean to std::poisson_distribution (UB / silent no-op).
# Non-crashing here -> normal in-process test.
# ---------------------------------------------------------------------------
def test_sample_poisson_negative_rate_raises():
    """A negative Poisson intensity must be rejected with a clean ValueError."""
    # BUG: sample_poisson accepts negative rate with no validation, passing a
    #      negative mean to std::poisson_distribution (undefined behavior).
    # Expected: ValueError for rate < 0, consistent with the existing lo<=hi /
    #           lo-hi-length validation (test_lo_greater_than_hi_raises).
    # Observed today: returns clouds with 0 points, no error, exit 0 -- relying
    #           on UB that merely happens to yield empty clouds on this build.
    mpcf.system.force_cpu(True)
    try:
        with pytest.raises(ValueError):
            sample_poisson((5,), dim=2, rate=-3.0,
                           generator=mpcf.random.Generator(1))
    finally:
        mpcf.system.force_cpu(False)
