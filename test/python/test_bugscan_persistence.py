import numpy as np
import pytest

import stablebear as sb
import stablebear.persistence as mp
from stablebear.typing import pcloud64


# ---------------------------------------------------------------------------
# Bug #27: a malformed point cloud (per-cloud rank != 2) is rejected by a
# deliberate C++ runtime_error -- but the exception, thrown inside the parallel
# walk, was stored in the task future and never retrieved, so the call silently
# returned all-empty barcodes. The exception must now propagate to Python.
# ---------------------------------------------------------------------------


def test_rank1_pcloud_element_raises():
    pc = sb.zeros((1,), dtype=pcloud64)
    pc[0] = sb.FloatTensor(np.array([1.0, 2.0, 3.0, 4.0]))   # rank-1, invalid
    with pytest.raises(RuntimeError, match="unexpected shape"):
        mp.compute_persistent_homology(pc, max_dim=1, verbose=False)


def test_1d_array_raises():
    with pytest.raises(RuntimeError, match="unexpected shape"):
        mp.compute_persistent_homology(np.array([0.0, 5.0, 10.0]), max_dim=1, verbose=False)


def test_3d_array_raises():
    with pytest.raises(RuntimeError, match="unexpected shape"):
        mp.compute_persistent_homology(np.zeros((2, 3, 2)), max_dim=1, verbose=False)


def test_mixed_valid_and_invalid_raises_not_silent():
    pc = sb.zeros((2,), dtype=pcloud64)
    pc[0] = sb.FloatTensor(np.random.RandomState(0).randn(6, 2))   # valid
    pc[1] = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))             # rank-1, invalid
    with pytest.raises(RuntimeError, match="unexpected shape"):
        mp.compute_persistent_homology(pc, max_dim=1, verbose=False)


def test_valid_pointcloud_still_computes():
    pts = np.random.RandomState(0).randn(10, 2)
    bcs = mp.compute_persistent_homology(pts, max_dim=1, verbose=False)
    # H0 of n points has n bars; the result must be non-empty (not swallowed).
    assert len(bcs[0].to_numpy()) == 10
