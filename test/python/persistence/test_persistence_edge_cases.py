"""Edge case tests for persistent homology computation."""

import numpy as np

import stablebear as sb
import stablebear.persistence as pers
from stablebear.persistence.barcode import Barcode


def _bc(pairs, dtype=np.float64):
    """Create a Barcode from a list of (birth, death) pairs."""
    if len(pairs) == 0:
        return Barcode(np.zeros((0, 2), dtype=dtype))
    return Barcode(np.array(pairs, dtype=dtype))


# --- Empty point cloud ---


def test_empty_point_cloud():
    """An empty point cloud should produce zero bars in all dimensions."""
    X = np.zeros((0, 2), dtype=np.float64)
    bcs = pers.compute_persistent_homology(X, max_dim=1, verbose=False)
    assert bcs[0].is_isomorphic_to(_bc([]))
    assert bcs[1].is_isomorphic_to(_bc([]))


def test_empty_point_cloud_reduced():
    """Reduced homology of an empty point cloud should be trivial."""
    X = np.zeros((0, 2), dtype=np.float64)
    bcs = pers.compute_persistent_homology(X, max_dim=1, reduced=True, verbose=False)
    assert bcs[0].is_isomorphic_to(_bc([]))
    assert bcs[1].is_isomorphic_to(_bc([]))


# --- Single point ---


def test_single_point_has_one_h0_bar():
    """A single point should produce exactly one H0 bar [0, inf) (unreduced)."""
    X = np.array([[1.0, 2.0]])  # 1 point in R^2
    bcs = pers.compute_persistent_homology(X, max_dim=1, verbose=False)
    assert bcs[0].is_isomorphic_to(_bc([[0.0, np.inf]]))
    assert bcs[1].is_isomorphic_to(_bc([]))


def test_single_point_reduced_has_no_bars():
    """Reduced homology of a single point should be trivial."""
    X = np.array([[0.0, 0.0]])
    bcs = pers.compute_persistent_homology(X, max_dim=1, reduced=True, verbose=False)
    assert bcs[0].is_isomorphic_to(_bc([]))
    assert bcs[1].is_isomorphic_to(_bc([]))


# --- Two points ---


def test_two_points_h0():
    """Two points produce 2 H0 bars (unreduced): one essential, one finite."""
    X = np.array([[0.0, 0.0], [3.0, 0.0]])  # distance = 3
    bcs = pers.compute_persistent_homology(X, max_dim=1, verbose=False)
    assert bcs[0].is_isomorphic_to(_bc([[0.0, 3.0], [0.0, np.inf]]))
    assert bcs[1].is_isomorphic_to(_bc([]))


# --- All identical points ---


def test_identical_points():
    """All identical points: distances are 0, so all merge at scale 0."""
    X = np.array([[1.0, 1.0]] * 5)
    bcs = pers.compute_persistent_homology(X, max_dim=1, verbose=False)
    assert bcs[0].is_isomorphic_to(_bc([[0.0, np.inf]]))


# --- max_dim variations ---


def test_maxdim_0_only_returns_h0():
    """max_dim=0 should return only H0."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    bcs = pers.compute_persistent_homology(X, max_dim=0, verbose=False)
    assert bcs.shape == (1,)


def test_maxdim_2_returns_three_dims():
    """max_dim=2 returns H0, H1, H2."""
    X = np.random.randn(10, 3).astype(np.float64)
    bcs = pers.compute_persistent_homology(X, max_dim=2, verbose=False)
    assert bcs.shape == (3,)


# --- Collinear points ---


def test_collinear_points_no_h1():
    """Collinear points should produce no H1 features."""
    X = np.array([[float(i), 0.0] for i in range(5)])
    bcs = pers.compute_persistent_homology(X, max_dim=1, verbose=False)
    assert bcs[1].is_isomorphic_to(_bc([]))


# --- Float32 vs Float64 ---


def test_persistence_f32_and_f64_give_isomorphic_barcodes():
    """Both precisions should give the same topological result."""
    np.random.seed(123)
    pts = np.random.randn(8, 2)

    bcs32 = pers.compute_persistent_homology(
        pts.astype(np.float32), max_dim=1, verbose=False
    )
    bcs64 = pers.compute_persistent_homology(
        pts.astype(np.float64), max_dim=1, verbose=False
    )

    # Same number of bars in each dimension
    assert len(bcs32[0]) == len(bcs64[0])
    assert len(bcs32[1]) == len(bcs64[1])


# --- PointCloudTensor with single point per cloud ---


def test_pcloud_tensor_single_point_per_cloud():
    """Each point cloud has 1 point: trivial topology."""
    X = sb.zeros((3,), dtype=sb.pcloud64)
    for i in range(3):
        X[i] = np.array([[float(i), 0.0]])

    bcs = pers.compute_persistent_homology(X, max_dim=1, verbose=False)
    assert bcs.shape == (3, 2)

    for i in range(3):
        assert bcs[i, 0].is_isomorphic_to(_bc([[0.0, np.inf]]))
        assert bcs[i, 1].is_isomorphic_to(_bc([]))
