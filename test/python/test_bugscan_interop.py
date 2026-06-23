import matplotlib

matplotlib.use("Agg")

import pytest

import stablebear as sb
import stablebear.plotting as plotting
from stablebear.reductions import max_time


def test_max_time_empty_raises_valueerror():
    """max_time on an empty reduction dimension raises ValueError, not SIGSEGV."""
    # BUG (#46): max_time() segfaulted on an empty PcfTensor -- max_element
    # seeded the reduction from *begin() of an empty range (UB). Expected: a
    # clean Python exception, mirroring numpy's zero-size reduction error.
    with pytest.raises(ValueError):
        max_time(sb.PcfTensor([]))


def test_max_time_empty_dim_of_higher_rank_raises():
    """An empty reduction dimension in a higher-rank tensor also raises cleanly."""
    A = sb.PcfTensor(sb.zeros((3, 0)))
    with pytest.raises(ValueError):
        max_time(A, dim=1)


def test_max_time_reduces_nonempty_dim_with_empty_sibling():
    """Reducing a non-empty dim is fine even when another dim is empty."""
    # shape (0, 4) reduced along the size-4 dim yields an empty (0,) result --
    # there are no slices to reduce, so this must not hit the empty-range guard.
    A = sb.PcfTensor(sb.zeros((0, 4)))
    assert max_time(A, dim=1).shape == (0,)


def test_plot_empty_is_graceful_noop():
    """plotting.plot on an empty PcfTensor is a graceful no-op, not a crash."""
    # BUG (#46): plotting.plot() reached max_time internally and segfaulted.
    # Expected: nothing to draw, so plot returns without raising.
    plotting.plot(sb.PcfTensor([]))
