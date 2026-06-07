"""Red-until-fixed regression tests for known shape-op defects (bug scan).

These tests document *known, currently-unfixed* bugs in the installed
``masspcf`` package, area key ``shapeops``. Each test asserts the CORRECT /
intended behavior, so it FAILS today (red) and will pass once the underlying
bug is fixed. Do NOT weaken a test to make it green -- fix the bug instead.
See ``bug-scan-findings.md`` at the repo root for the catalogue / root causes.

All four defects in this area are *silent* (exit 0): none hard-crashes the
interpreter, so every test runs in-process. Two are silent data corruption
(``flatten``, ``broadcast_to``); two are misleading raw-pybind ``TypeError``s
on negative axes (``transpose``, ``squeeze``).
"""

import numpy as np
import pytest

import masspcf as mpcf


def test_flatten_numpy_view_is_row_major():
    """flatten() must expose row-major data through numpy/buffer, not stride-0."""
    # BUG: flatten() sets the flattened-axis stride to 0, corrupting np.asarray()
    # Expected: np.asarray(t.flatten()) == [0,1,2,3,4,5] (row-major), stride 1.
    # Observed today: strides (0,) -> np.asarray gives [0,0,0,0,0,0] (elem 0 repeated).
    t = mpcf.FloatTensor(np.arange(6, dtype=np.float64).reshape(2, 3))
    f = t.flatten()

    expected = np.arange(6, dtype=np.float64).reshape(2, 3).flatten()

    # The numpy/buffer view must match the canonical row-major flatten...
    assert np.array_equal(np.asarray(f), expected)
    # ...and must agree with the element-access path (which is already correct).
    assert [f[i] for i in range(6)] == expected.tolist()
    # The contiguous flattened axis must have stride 1, never 0.
    assert tuple(f.strides) == (1,)


def test_flatten_then_expand_dims_numpy_view_is_correct():
    """expand_dims after flatten must produce a correct numpy view (stride-0 propagation)."""
    # BUG: flatten()'s stride-0 propagates; expand_dims(0) gives strides (0,0).
    # Expected: np.asarray gives [[0,1,2,3,4,5]].
    # Observed today: [[0,0,0,0,0,0]].
    t = mpcf.FloatTensor(np.arange(6, dtype=np.float64).reshape(2, 3))
    e = t.flatten().expand_dims(0)

    expected = np.arange(6, dtype=np.float64).reshape(1, 6)
    assert np.array_equal(np.asarray(e), expected)


def test_flatten_of_noncontiguous_slice_numpy_view_is_correct():
    """flatten() of a non-contiguous slice must read the sliced data, not element 0 repeated."""
    # BUG: flatten() of a non-contiguous slice still sets stride 0 (after an
    # internal copy()) and leaves m_isContiguous=true, so numpy reads elem 0.
    # Expected: [1,2,5,6,9,10] (the row-major flatten of the [:,1:3] slice).
    # Observed today: [1,1,1,1,1,1].
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    big = mpcf.FloatTensor(base)
    f = big[:, 1:3].flatten()

    expected = base[:, 1:3].flatten()
    assert np.array_equal(np.asarray(f), expected)


def test_broadcast_to_view_is_read_only():
    """Writing into a broadcast (stride-0) view must be rejected, not silently corrupt the source."""
    # BUG: broadcast_to returns a writable stride-0 view; assigning aliases
    # across the broadcast axis and writes back into the source.
    # Expected (NumPy parity): assigning into a broadcast view raises
    # ValueError (read-only) and the source is unchanged.
    # Observed today: assignment succeeds, corrupts src and all rows alias.
    src = mpcf.FloatTensor(np.array([[1.0, 2.0, 3.0]]))
    b = src.broadcast_to((4, 3))

    with pytest.raises((ValueError, RuntimeError)):
        b[0, 0] = 999.0

    # Regardless of how the write is rejected, the source must NOT be corrupted.
    assert np.array_equal(np.asarray(src), np.array([[1.0, 2.0, 3.0]]))


def test_broadcast_to_view_does_not_corrupt_source_via_slice_assign():
    """Slice-assigning into a broadcast view must not fan out and corrupt the source."""
    # BUG: slice assignment through a broadcast view scatters across the
    # stride-0 axis and mutates the source (same root as the single-element case).
    # Expected: src unchanged (NumPy raises read-only; at minimum no corruption).
    # Observed today: src becomes [[555.0, 2.0, 3.0]].
    src = mpcf.IntTensor(np.array([[1, 2, 3]]))
    b = src.broadcast_to((4, 3))

    with pytest.raises((ValueError, RuntimeError)):
        b[:, 0] = 555

    assert np.array_equal(np.asarray(src), np.array([[1, 2, 3]]))


def test_transpose_accepts_negative_axes():
    """transpose() must resolve negative axes like NumPy and swapaxes, not raise a raw TypeError."""
    # BUG: transpose() passes negative axes straight to a size_t C++ binding,
    # so pybind rejects them with "incompatible function arguments".
    # Expected: transpose([-1,-2,-3]) -> shape (4,3,2) (NumPy parity); mixed
    # valid permutations like [2,1,-3] also work.
    # Observed today: raw pybind TypeError before any validation runs.
    t = mpcf.FloatTensor(np.arange(24, dtype=np.float64).reshape(2, 3, 4))

    assert tuple(t.transpose([-1, -2, -3]).shape) == (4, 3, 2)
    assert tuple(t.transpose([2, 1, -3]).shape) == (4, 3, 2)
    # Sanity: swapaxes already resolves the same negatives (the inconsistency).
    assert tuple(t.swapaxes(-1, -3).shape) == (4, 3, 2)


def test_squeeze_accepts_negative_axis():
    """squeeze(axis) must resolve a negative axis like NumPy, not raise a raw TypeError."""
    # BUG: squeeze(axis) passes a negative axis to a size_t C++ binding, so
    # pybind rejects it with "incompatible function arguments".
    # Expected (NumPy parity): squeeze(-1) -> (1,6); squeeze(-3) -> (6,1).
    # Observed today: raw pybind TypeError before any validation runs.
    ts = mpcf.FloatTensor(np.arange(6, dtype=np.float64).reshape(1, 6, 1))

    assert tuple(ts.squeeze(-1).shape) == (1, 6)
    assert tuple(ts.squeeze(-3).shape) == (6, 1)
