import numpy as np
import pytest

import stablebear as sb
import stablebear.persistence as mp
from stablebear.typing import pcloud64


# ---------------------------------------------------------------------------
# Bug #27: a malformed point cloud (per-cloud rank != 2) must be rejected loudly
# rather than silently returning all-empty barcodes. A point cloud is now
# validated to be 2-D (n_points, dim) on assignment, so the bad value is caught
# immediately with a ValueError instead of slipping through to persistence.
# ---------------------------------------------------------------------------


def test_rank1_pcloud_element_raises():
    pc = sb.zeros((1,), dtype=pcloud64)

    # Expect the ValueError immediately upon assignment.
    with pytest.raises(ValueError, match="A point cloud must be 2-D"):
        pc[0] = sb.FloatTensor(np.array([1.0, 2.0, 3.0, 4.0]))   # rank-1, invalid


def test_1d_array_raises():
    # stablebear intercepts the raw NumPy array inside compute_persistent_homology
    # and throws a ValueError during internal assignment.
    with pytest.raises(ValueError, match="A point cloud must be 2-D"):
        mp.compute_persistent_homology(np.array([0.0, 5.0, 10.0]), max_dim=1, verbose=False)


def test_3d_array_raises():
    # Same intercept here: 3D input is rejected via ValueError inside compute_persistent_homology.
    with pytest.raises(ValueError, match="A point cloud must be 2-D"):
        mp.compute_persistent_homology(np.zeros((2, 3, 2)), max_dim=1, verbose=False)


def test_mixed_valid_and_invalid_raises_not_silent():
    pc = sb.zeros((2,), dtype=pcloud64)
    pc[0] = sb.FloatTensor(np.random.RandomState(0).randn(6, 2))   # valid

    # The crash occurs right here on assignment, so the context manager must wrap it.
    with pytest.raises(ValueError, match="A point cloud must be 2-D"):
        pc[1] = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))          # rank-1, invalid


def test_valid_pointcloud_still_computes():
    pts = np.random.RandomState(0).randn(10, 2)
    bcs = mp.compute_persistent_homology(pts, max_dim=1, verbose=False)
    # H0 of n points has n bars; the result must be non-empty (not swallowed).
    assert len(bcs[0].to_numpy()) == 10
