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
