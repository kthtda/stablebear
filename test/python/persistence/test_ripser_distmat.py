import numpy as np
from scipy.spatial.distance import pdist, squareform

import stablebear as sb
import stablebear.persistence as pers


def _points_to_distmat(points, dtype):
    """Build a DistanceMatrix from a NumPy point array."""
    dists = squareform(pdist(points))
    n = dists.shape[0]
    dm = sb.DistanceMatrix(n, dtype=dtype)
    for i in range(n):
        for j in range(i):
            dm[i, j] = dists[i, j]
    return dm


def test_distmat_single_matches_point_cloud():
    """Single DistanceMatrix gives the same barcodes as the equivalent point cloud."""
    points = np.array([
        [0.0, 0.0],
        [0.0, 4.0],
        [3.0, 0.0],
        [3.0, 4.0],
    ])

    bcs_pc = pers.compute_persistent_homology(points, max_dim=2, verbose=False)

    dm = _points_to_distmat(points, sb.float64)
    bcs_dm = pers.compute_persistent_homology(dm, max_dim=2, verbose=False)

    assert isinstance(bcs_dm, pers.BarcodeTensor)
    assert bcs_dm.dtype == sb.barcode64
    assert bcs_dm.shape == bcs_pc.shape

    for k in range(bcs_pc.shape[0]):
        assert bcs_dm[k].is_isomorphic_to(bcs_pc[k])


def test_distmat_single_f32():
    """DistanceMatrix with f32 dtype produces BarcodeTensor with barcode32."""
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    dm = _points_to_distmat(points, sb.float32)
    bcs = pers.compute_persistent_homology(dm, max_dim=1, verbose=False)

    assert isinstance(bcs, pers.BarcodeTensor)
    assert bcs.dtype == sb.barcode32
    assert bcs.shape == (2,)


def test_distmat_tensor_matches_point_cloud():
    """DistanceMatrixTensor gives the same barcodes as PointCloudTensor."""
    np.random.seed(42)

    pclouds = sb.zeros((3,), dtype=sb.pcloud64)
    dmats = sb.zeros((3,), dtype=sb.distmat64)

    for i in range(3):
        pts = np.random.randn(8, 3)
        pclouds[i] = pts
        dmats[i] = _points_to_distmat(pts, sb.float64)

    bcs_pc = pers.compute_persistent_homology(pclouds, max_dim=1, verbose=False)
    bcs_dm = pers.compute_persistent_homology(dmats, max_dim=1, verbose=False)

    assert isinstance(bcs_dm, pers.BarcodeTensor)
    assert bcs_dm.dtype == sb.barcode64
    assert bcs_dm.shape == bcs_pc.shape

    for i in range(3):
        for k in range(2):
            assert bcs_dm[i, k].is_isomorphic_to(bcs_pc[i, k])


def test_distmat_reduced_homology():
    """Reduced homology with DistanceMatrix omits the essential H0 bar."""
    points = np.array([
        [0.0, 0.0],
        [0.0, 4.0],
        [3.0, 0.0],
        [3.0, 4.0],
    ])

    dm = _points_to_distmat(points, sb.float64)

    bcs_unreduced = pers.compute_persistent_homology(dm, max_dim=1, reduced=False, verbose=False)
    bcs_reduced = pers.compute_persistent_homology(dm, max_dim=1, reduced=True, verbose=False)

    # Unreduced H0 should have one more bar (the essential [0, inf))
    assert len(bcs_unreduced[0]) == len(bcs_reduced[0]) + 1

    # H1 should be the same
    assert bcs_unreduced[1].is_isomorphic_to(bcs_reduced[1])
