import numpy as np
import pytest

import stablebear as sb


def test_mean_dim_equal_ndim_raises():
    """mean with dim == ndim must raise, not silently return a wrong shape."""
    # BUG: Out-of-range dim in reductions is unvalidated (dim == ndim)
    # Expected: an out-of-range axis raises a clean Python exception; never a
    # silent wrong shape, and never heap corruption / interpreter abort.
    A = sb.zeros((3, 4))
    with pytest.raises((IndexError, ValueError)):
        sb.mean(A, dim=2)


def test_max_time_dim_equal_ndim_raises():
    """max_time with dim == ndim must raise, not silently return a wrong shape."""
    A = sb.zeros((3, 4))
    with pytest.raises((IndexError, ValueError)):
        sb.max_time(A, dim=2)


def test_mean_dim_greater_than_ndim_raises():
    """mean with dim > ndim must raise cleanly (previously corrupted the heap)."""
    A = sb.zeros((3, 4))
    for bad_dim in (3, 5, 6, 100):
        with pytest.raises((IndexError, ValueError)):
            sb.mean(A, dim=bad_dim)


def test_max_time_dim_greater_than_ndim_raises():
    """max_time with dim > ndim must raise cleanly (previously corrupted the heap)."""
    A = sb.zeros((3, 4))
    for bad_dim in (3, 5, 6, 100):
        with pytest.raises((IndexError, ValueError)):
            sb.max_time(A, dim=bad_dim)


def test_mean_1d_out_of_range_raises():
    """1-D reduction with dim >= ndim must raise rather than silently return."""
    A = sb.zeros((3,))
    for bad_dim in (1, 2):
        with pytest.raises((IndexError, ValueError)):
            sb.mean(A, dim=bad_dim)


def test_mean_in_range_dim_still_works():
    """Sanity check: valid dims keep producing the documented reduced shape."""
    A = sb.zeros((3, 4))
    assert sb.mean(A, dim=0).shape == (4,)
    assert sb.mean(A, dim=1).shape == (3,)
    assert sb.max_time(A, dim=0).shape == (4,)
    assert sb.max_time(A, dim=1).shape == (3,)


# ---------------------------------------------------------------------------
# Bug #26: negative dim raised a misleading pybind TypeError (the C++ binding's
# dim is size_t), instead of resolving from the end like NumPy and the rest of
# the tensor API (t[-1], stack(axis=-1)). dim is now normalized in the wrapper.
# ---------------------------------------------------------------------------


def test_mean_negative_dim_resolves_like_numpy():
    A = sb.zeros((3, 4))
    assert sb.mean(A, dim=-1).shape == (3,)   # numpy: zeros((3,4)).mean(axis=-1).shape == (3,)
    assert sb.mean(A, dim=-2).shape == (4,)


def test_max_time_negative_dim_resolves_like_numpy():
    A = sb.zeros((3, 4))
    assert sb.max_time(A, dim=-1).shape == (3,)
    assert sb.max_time(A, dim=-2).shape == (4,)


def test_negative_dim_matches_positive_dim():
    """dim=-k must reduce the same axis as dim=ndim-k."""
    B = sb.zeros((2, 3, 4))
    assert sb.mean(B, dim=-1).shape == sb.mean(B, dim=2).shape == (2, 3)
    assert sb.mean(B, dim=-2).shape == sb.mean(B, dim=1).shape == (2, 4)
    assert sb.mean(B, dim=-3).shape == sb.mean(B, dim=0).shape == (3, 4)


@pytest.mark.parametrize("bad_dim", [2, 3, 5, -3, -4, -100])
def test_out_of_range_dim_raises_index_error(bad_dim):
    """Out-of-range dim (positive or negative) raises IndexError."""
    A = sb.zeros((3, 4))
    for fn in (sb.mean, sb.max_time):
        with pytest.raises(IndexError):
            fn(A, dim=bad_dim)


# ---------------------------------------------------------------------------
# Bug #82: an out-of-range reduction dim now raises numpy.AxisError. AxisError
# subclasses BOTH IndexError and ValueError, so the existing
# pytest.raises(IndexError) tests above keep passing.
# ---------------------------------------------------------------------------

# numpy.AxisError moved under numpy.exceptions in NumPy 2 (and was dropped from
# the top-level namespace), so prefer that location and fall back for NumPy 1.
_AxisError = getattr(np.exceptions, "AxisError", None) or getattr(np, "AxisError")


@pytest.mark.parametrize("bad_dim", [2, 3, 5, 100, -3, -4, -100])
def test_out_of_range_dim_raises_axis_error(bad_dim):
    """Too-large and too-negative dims raise numpy.AxisError for both reductions."""
    A = sb.zeros((3, 4))
    for fn in (sb.mean, sb.max_time):
        with pytest.raises(_AxisError) as excinfo:
            fn(A, dim=bad_dim)
        # Backward compatibility: AxisError is still catchable as IndexError.
        assert issubclass(type(excinfo.value), IndexError)


def test_axis_error_is_subclass_of_index_error():
    """Sanity: numpy.AxisError stays a subclass of IndexError (and ValueError)."""
    assert issubclass(_AxisError, IndexError)
    assert issubclass(_AxisError, ValueError)


def test_valid_negative_dim_unchanged_by_axis_error():
    """The AxisError change must not regress valid negative-dim reductions."""
    A = sb.zeros((3, 4))
    assert sb.mean(A, dim=-1).shape == (3,)
    assert sb.mean(A, dim=-2).shape == (4,)
    assert sb.max_time(A, dim=-1).shape == (3,)
    assert sb.max_time(A, dim=-2).shape == (4,)


def test_max_time_empty_reduced_axis_raises_not_segfault():
    """max_time over a size-0 reduced axis must raise cleanly, never SIGSEGV.

    Issue #25 (bug scan): ``max_time`` segfaulted on an empty reduction
    dimension while ``mean`` handled it. Resolved together with #46 —
    ``max_element`` now rejects an empty reduction range with ``ValueError``
    (``max`` has no identity over an empty range). ``mean`` still returns a
    valid reduced tensor over the same empty axes, so the two stay consistent.
    """
    # 1-D empty tensor reduced along its only (empty) axis.
    with pytest.raises(ValueError):
        sb.max_time(sb.zeros((0,)), dim=0)
    # Empty inner dimension of a higher-rank tensor.
    with pytest.raises(ValueError):
        sb.max_time(sb.zeros((3, 0)), dim=1)
    # Contrast: mean returns a valid reduced tensor over the same empty axes.
    assert sb.mean(sb.zeros((0,)), dim=0).shape == (1,)
    assert sb.mean(sb.zeros((3, 0)), dim=1).shape == (3,)
