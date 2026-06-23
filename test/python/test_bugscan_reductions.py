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
