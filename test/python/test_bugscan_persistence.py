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


# ---------------------------------------------------------------------------
# Bug #28: compute_persistent_homology squeezed the batch dimension for a
# genuine length-1 (1,) tensor input, so a real batch of one cloud/matrix lost
# its leading axis and became indistinguishable from a single-cloud result.
# Only scalar-convenience inputs (a single ndarray / FloatTensor /
# DistanceMatrix) should drop the leading axis.
# ---------------------------------------------------------------------------


def _euclidean_distmat(points):
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(d, 0.0)
    return d


def test_length1_pointcloud_batch_keeps_leading_axis():
    pc = sb.zeros((1,), dtype=pcloud64)
    pc[0] = sb.FloatTensor(np.random.RandomState(0).randn(8, 2))
    out = mp.compute_persistent_homology(pc, max_dim=1)
    assert tuple(out.shape) == (1, 2)
    # the (cloud 0, H0) barcode is still indexable
    assert out[0, 0].to_numpy().shape[0] == 8


def test_length1_distmat_batch_keeps_leading_axis():
    pts = np.random.RandomState(0).randn(8, 2)
    dt = sb.DistanceMatrixTensor.from_numpy(_euclidean_distmat(pts)[None, :, :])
    out = mp.compute_persistent_homology(dt, max_dim=1)
    assert tuple(out.shape) == (1, 2)


def test_multi_pointcloud_batch_shape_unchanged():
    pc = sb.zeros((2,), dtype=pcloud64)
    pc[0] = sb.FloatTensor(np.random.RandomState(0).randn(8, 2))
    pc[1] = sb.FloatTensor(np.random.RandomState(1).randn(8, 2))
    out = mp.compute_persistent_homology(pc, max_dim=1)
    assert tuple(out.shape) == (2, 2)


@pytest.mark.parametrize("make_single", [
    lambda: np.random.RandomState(0).randn(8, 2),
    lambda: sb.FloatTensor(np.random.RandomState(0).randn(8, 2)),
    lambda: sb.DistanceMatrix.from_dense(
        _euclidean_distmat(np.random.RandomState(0).randn(8, 2))),
])
def test_scalar_convenience_input_drops_leading_axis(make_single):
    out = mp.compute_persistent_homology(make_single(), max_dim=1)
    assert tuple(out.shape) == (2,)


# ---------------------------------------------------------------------------
# Bug #29: barcode_to_stable_rank / barcode_to_betti_curve /
# barcode_to_accumulated_persistence squeezed any 2-D (1, N) BarcodeTensor down
# to (N,), dropping a genuine length-1 leading axis. The single-Barcode
# convenience case is handled by a separate branch and still returns a Pcf.
# ---------------------------------------------------------------------------


def _length1_barcode_tensor():
    pc = sb.zeros((2,), dtype=pcloud64)
    pc[0] = sb.FloatTensor(np.random.RandomState(0).randn(8, 2))
    pc[1] = sb.FloatTensor(np.random.RandomState(1).randn(8, 2))
    return mp.compute_persistent_homology(pc, max_dim=1)[0:1]   # genuine (1, 2)


@pytest.mark.parametrize("summary", [
    mp.barcode_to_stable_rank,
    mp.barcode_to_betti_curve,
    mp.barcode_to_accumulated_persistence,
])
def test_barcode_summaries_preserve_length1_leading_axis(summary):
    bc = _length1_barcode_tensor()
    assert tuple(bc.shape) == (1, 2)
    out = summary(bc)
    assert tuple(out.shape) == (1, 2)
    # both elements are still reachable by 2-D index (no axis dropped)
    assert isinstance(out[0, 1], sb.Pcf)


@pytest.mark.parametrize("summary", [
    mp.barcode_to_stable_rank,
    mp.barcode_to_betti_curve,
    mp.barcode_to_accumulated_persistence,
])
def test_barcode_summaries_single_barcode_returns_pcf(summary):
    b = mp.Barcode(np.array([[0.0, 1.0], [0.0, 2.0]]))
    assert isinstance(summary(b), sb.Pcf)
