"""Whole-pipeline tests for compute_homological_kernel through the public API.

Expected barcodes are hand-computed. The 2D point-cloud cases use the
orthogonal projection onto the diagonal y = x as the dominated input Y:
a point (x, y) maps to (m, m) with m = (x + y) / 2.
"""

import numpy as np
import pytest

import stablebear as sb
import stablebear.persistence as pers
from stablebear.persistence.barcode import Barcode

S2 = np.sqrt(2.0)
S10 = np.sqrt(10.0)


def _bc(pairs, dtype=np.float64):
    """Create a Barcode from a list of (birth, death) pairs."""
    if len(pairs) == 0:
        return Barcode(np.zeros((0, 2), dtype=dtype))
    return Barcode(np.array(pairs, dtype=dtype))


def _distance_matrix(points, dtype=sb.float64):
    """Build a DistanceMatrix of Euclidean pairwise distances."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    dm = sb.DistanceMatrix(n, dtype=dtype)
    for i in range(n):
        for j in range(i + 1, n):
            dm[i, j] = float(np.linalg.norm(pts[i] - pts[j]))
    return dm


def _diag(points):
    """Orthogonal projection onto the diagonal: (x, y) -> (m, m), m = mean."""
    pts = np.asarray(points)
    m = pts.mean(axis=-1, keepdims=True)
    return np.broadcast_to(m, pts.shape).copy()


# --- Hand-computed 2D cases for the diagonal projection ---

# All points on the diagonal are fixed by the projection, so d' = d and
# every bar is empty [w, w).
DIAGONAL_POINTS = [[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]
DIAGONAL_BARS = [[S2, S2], [2 * S2, 2 * S2]]

# (2, -2) projects onto (0, 0), coinciding with the projection of the first
# point: the pair merges at 0 in d', giving a bar born at 0. The third point
# lies on the diagonal, so the second bar is empty.
COINCIDENT_POINTS = [[0.0, 0.0], [2.0, -2.0], [3.0, 3.0]]
COINCIDENT_BARS = [[0.0, 2 * S2], [3 * S2, 3 * S2]]

# As above, but the third point is off the diagonal, so the second bar is
# non-empty as well.
OFF_DIAGONAL_POINTS = [[0.0, 0.0], [2.0, -2.0], [4.0, 2.0]]
OFF_DIAGONAL_BARS = [[0.0, 2 * S2], [3 * S2, np.sqrt(20.0)]]

# Coincidence and merge-order inversion combined: (4, 0) and (0, 4) both
# project onto (2, 2), and the d' merge order differs from the d merge order.
# In d the coincident pair is not directly connected; it joins through
# (3, 3), so its bar dies at the subdominant distance sqrt(10), carried by a
# pair touching neither of the coincident points.
FLAGSHIP_POINTS = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [3.0, 3.0]]
FLAGSHIP_PROJECTED = [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [3.0, 3.0]]
FLAGSHIP_BARS = [[0.0, S10], [S2, S10], [2 * S2, 4.0]]


# --- Hand-computed barcodes through the diagonal projection ---


@pytest.mark.parametrize(
    "points, expected",
    [
        (DIAGONAL_POINTS, DIAGONAL_BARS),
        (COINCIDENT_POINTS, COINCIDENT_BARS),
        (OFF_DIAGONAL_POINTS, OFF_DIAGONAL_BARS),
        (FLAGSHIP_POINTS, FLAGSHIP_BARS),
    ],
    ids=["diagonal", "coincident", "off-diagonal-third", "coincidence-and-inversion"],
)
def test_diagonal_projection_gives_hand_computed_barcode(points, expected):
    bcs = pers.compute_homological_kernel(np.array(points), _diag(points))
    assert bcs.shape == (1,)
    assert bcs[0].is_isomorphic_to(_bc(expected))


def test_kernel_uses_indexed_point_clouds_without_materializing_owners():
    selection = sb.NestedTensor(sb.tensor([3, 0, 2, 1], dtype=sb.uint64))
    points = sb.PointCloudTensor(np.asarray(FLAGSHIP_POINTS))[selection]
    projected = sb.PointCloudTensor(np.asarray(FLAGSHIP_PROJECTED))[selection]
    point_storage_type = type(points._data)
    projected_storage_type = type(projected._data)

    bcs = pers.compute_homological_kernel(points[()], projected[()])

    assert type(points._data) is point_storage_type
    assert type(projected._data) is projected_storage_type
    assert bcs.shape == (1,)
    assert bcs[0].is_isomorphic_to(_bc(FLAGSHIP_BARS))


def test_coordinate_projection_gives_hand_computed_barcode():
    # Dropping the last coordinate gives d' = |x_i - x_j| for 2D input.
    # Points A=(0,0), B=(1,2), C=(3,1): both d MST edges (A-B, B-C) weigh
    # sqrt(5), so every cophenetic distance is sqrt(5). d' merges (A,B)@1,
    # then ({A,B},C)@2.
    points = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    projected = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    s5 = np.sqrt(5.0)

    bcs = pers.compute_homological_kernel(points, projected)
    assert bcs[0].is_isomorphic_to(_bc([[1.0, s5], [2.0, s5]]))


# --- Deaths are computed against the quotient space ---
#
# Contracting a d-far pair (which every nonempty bar does) creates a shortcut
# through the contracted component that can lower the death of a LATER merge
# between two other components.


def test_contraction_shortcut_lowers_later_death():
    # Diagonal frame: 1=(-1,11) and 3=(5,5) project to the same point and
    # contract at 0. The merge (0,2)@2*S2 then dies at 5*S2 via the chain
    # 0 -5*S2- 1 ≡ 3 -3*S2- 2, through a component containing neither 0 nor 2.
    points = np.array([[-6.0, 6.0], [-1.0, 11.0], [2.0, 2.0], [5.0, 5.0]])
    bcs = pers.compute_homological_kernel(points, _diag(points))
    assert bcs[0].is_isomorphic_to(
        _bc([[0.0, 6 * S2], [2 * S2, 5 * S2], [3 * S2, 3 * S2]])
    )


def test_staircase_death_needs_chained_contraction_shortcuts():
    # Two stacked rungs contract at 0; the (A,B)@1 merge dies at 8 via the
    # two-hop chain A -8- R1 ≡ R1' -3- R2 ≡ R2' -4- B (a k-rung staircase
    # needs k hops). The deaths equal the d-MST weight multiset
    # {3, 4, 8, 10, 10}.
    points = np.array(
        [[0.0, 20.0], [8.0, 20.0], [8.0, 10.0], [5.0, 10.0], [5.0, 0.0], [1.0, 0.0]]
    )
    projected = points.copy()
    projected[:, -1] = 0.0
    bcs = pers.compute_homological_kernel(points, projected)
    assert bcs[0].is_isomorphic_to(
        _bc([[0.0, 10.0], [0.0, 10.0], [1.0, 8.0], [3.0, 3.0], [4.0, 4.0]])
    )


# --- Randomized cross-check against an independent reference implementation ---


def _subdominant(D):
    """Subdominant (minimax) ultrametric of a dissimilarity matrix."""
    U = D.copy()
    for k in range(len(D)):
        U = np.minimum(U, np.maximum(U[:, k : k + 1], U[k : k + 1, :]))
    np.fill_diagonal(U, 0.0)
    return U


def _single_linkage_merges(D):
    """The n-1 single-linkage merges of D as (i, j, scale), ascending."""
    n = len(D)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    min_d = D[0].copy()
    parent = np.zeros(n, dtype=int)
    edges = []
    for _ in range(n - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, min_d)))
        edges.append((int(parent[j]), j, float(min_d[j])))
        in_tree[j] = True
        closer = D[j] < min_d
        min_d = np.where(closer, D[j], min_d)
        parent = np.where(closer, j, parent)
    edges.sort(key=lambda e: e[2])
    return edges


def _reference_kernel_barcode(D, Dp):
    """O(n^3) dense reference for ker mu: replays the d' merges while
    maintaining the full subdominant ultrametric of the quotient space
    explicitly (Corollary 4.1.1). Independent of the C++ sweep."""
    n = len(D)
    c = _subdominant(D)
    uf = list(range(n))

    def find(i):
        while uf[i] != i:
            uf[i] = uf[uf[i]]
            i = uf[i]
        return i

    alive = np.ones(n, dtype=bool)
    bars = []
    for a, b, w in _single_linkage_merges(Dp):
        ra, rb = find(a), find(b)
        bars.append([w, max(float(c[ra, rb]), w)])
        row = np.minimum(c[ra], c[rb])
        alive[rb] = False
        uf[rb] = ra
        c[ra], c[:, ra] = row, row
        c[ra, ra] = 0.0
        others = np.where(alive & (np.arange(n) != ra))[0]
        if len(others) > 1:
            sub = c[np.ix_(others, others)]
            c[np.ix_(others, others)] = np.minimum(
                sub, np.maximum(row[others][:, None], row[others][None, :])
            )
    return bars


def test_random_clouds_match_reference_implementation():
    rng = np.random.default_rng(20260716)
    for _ in range(25):
        n = int(rng.integers(3, 13))
        points = rng.normal(size=(n, 2))
        m = points.mean(axis=1)
        D = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        Dp = np.sqrt(2.0) * np.abs(m[:, None] - m[None, :])
        expected = _reference_kernel_barcode(D, np.minimum(Dp, D))

        bcs = pers.compute_homological_kernel(points, _diag(points))
        assert bcs[0].is_isomorphic_to(_bc(expected))


def test_random_distance_matrices_match_reference_implementation():
    rng = np.random.default_rng(20260716)
    for _ in range(25):
        n = int(rng.integers(3, 13))
        D = rng.uniform(0.1, 1.1, size=(n, n))
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
        scale = rng.uniform(0.0, 1.0, size=(n, n))
        Dp = D * ((scale + scale.T) / 2.0)
        expected = _reference_kernel_barcode(D, Dp)

        d = sb.DistanceMatrix(n)
        d_prime = sb.DistanceMatrix(n)
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = D[i, j]
                d_prime[i, j] = Dp[i, j]
        bcs = pers.compute_homological_kernel(d, d_prime)
        assert bcs[0].is_isomorphic_to(_bc(expected))


# --- Hand-computed distance-matrix cases, both precisions ---

DTYPES = [(sb.float32, np.float32), (sb.float64, np.float64)]

# Four points on a line at 0, 1, 3, 7. d merges: {0,1}@1, {1,2}@2, {2,3}@4.
LINE_METRIC = np.array(
    [
        [0.0, 1.0, 3.0, 7.0],
        [1.0, 0.0, 2.0, 6.0],
        [3.0, 2.0, 0.0, 4.0],
        [7.0, 6.0, 4.0, 0.0],
    ]
)


def _dm_from_full(full, dtype=sb.float64):
    """Build a DistanceMatrix from a full square matrix."""
    n = len(full)
    dm = sb.DistanceMatrix(n, dtype=dtype)
    for i in range(n):
        for j in range(i + 1, n):
            dm[i, j] = float(full[i][j])
    return dm


@pytest.mark.parametrize("sb_dtype, np_dtype", DTYPES, ids=["float32", "float64"])
def test_halved_metric_gives_doubling_bars(sb_dtype, np_dtype):
    # d' = d/2 keeps the merge structure, so every bar is [w, 2w) for the
    # d'-merge scales w = 0.5, 1, 2 of the line metric.
    bcs = pers.compute_homological_kernel(
        _dm_from_full(LINE_METRIC, sb_dtype),
        _dm_from_full(LINE_METRIC / 2.0, sb_dtype),
    )
    assert bcs[0].is_isomorphic_to(
        _bc([[0.5, 1.0], [1.0, 2.0], [2.0, 4.0]], dtype=np_dtype)
    )


@pytest.mark.parametrize("sb_dtype, np_dtype", DTYPES, ids=["float32", "float64"])
def test_identical_metrics_give_empty_bars(sb_dtype, np_dtype):
    # d' == d: every death equals its birth -- empty bars [w, w), no throw.
    bcs = pers.compute_homological_kernel(
        _dm_from_full(LINE_METRIC, sb_dtype), _dm_from_full(LINE_METRIC, sb_dtype)
    )
    assert bcs[0].is_isomorphic_to(
        _bc([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]], dtype=np_dtype)
    )


@pytest.mark.parametrize("sb_dtype, np_dtype", DTYPES, ids=["float32", "float64"])
def test_death_carried_by_non_merged_pair(sb_dtype, np_dtype):
    # Points x=0, y=1, a=2, b=3. d is the ultrametric of the tree
    # (((x,y)@1, a)@5, b)@9. The d' merges are (a,x)@0.1, (b,y)@0.2, then
    # {a,x} joins {b,y} at 0.5 via the edge (a,b). The death of that third bar
    # is min cross-pair d_sd = d_sd(x,y) = 1, carried by the pair (x,y) --
    # which involves neither endpoint of the merge edge: the death is
    # determined by the whole component pair.
    d = [
        [0.0, 1.0, 5.0, 9.0],
        [1.0, 0.0, 5.0, 9.0],
        [5.0, 5.0, 0.0, 9.0],
        [9.0, 9.0, 9.0, 0.0],
    ]
    d_prime = [
        [0.0, 0.9, 0.1, 0.9],
        [0.9, 0.0, 0.9, 0.2],
        [0.1, 0.9, 0.0, 0.5],
        [0.9, 0.2, 0.5, 0.0],
    ]
    bcs = pers.compute_homological_kernel(
        _dm_from_full(d, sb_dtype), _dm_from_full(d_prime, sb_dtype)
    )
    assert bcs[0].is_isomorphic_to(
        _bc([[0.1, 5.0], [0.2, 9.0], [0.5, 1.0]], dtype=np_dtype)
    )


@pytest.mark.parametrize("sb_dtype, np_dtype", DTYPES, ids=["float32", "float64"])
def test_merge_order_inversion(sb_dtype, np_dtype):
    # Pure merge-order inversion (abstract metrics): d' merges A-B first, but
    # in d the tight pair is B-C. Deaths come from d, order from d'.
    d = [
        [0.0, 5.0, 6.0],
        [5.0, 0.0, 3.0],
        [6.0, 3.0, 0.0],
    ]
    d_prime = [
        [0.0, 1.0, 3.0],
        [1.0, 0.0, 2.0],
        [3.0, 2.0, 0.0],
    ]
    bcs = pers.compute_homological_kernel(
        _dm_from_full(d, sb_dtype), _dm_from_full(d_prime, sb_dtype)
    )
    assert bcs[0].is_isomorphic_to(_bc([[1.0, 5.0], [2.0, 3.0]], dtype=np_dtype))


@pytest.mark.parametrize("sb_dtype, np_dtype", DTYPES, ids=["float32", "float64"])
def test_tie_invariant_barcode(sb_dtype, np_dtype):
    # Two disjoint pairs merge at the same birth (tie). The barcode multiset
    # must be independent of which tied merge is processed first.
    # Note d_sd(C,D) = 4 (MST path C-A-D), not the raw 5.
    d = [
        [0.0, 3.0, 4.0, 4.0],
        [3.0, 0.0, 4.0, 4.0],
        [4.0, 4.0, 0.0, 5.0],
        [4.0, 4.0, 5.0, 0.0],
    ]
    d_prime = [
        [0.0, 1.0, 2.0, 2.0],
        [1.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 0.0, 1.0],
        [2.0, 2.0, 1.0, 0.0],
    ]
    bcs = pers.compute_homological_kernel(
        _dm_from_full(d, sb_dtype), _dm_from_full(d_prime, sb_dtype)
    )
    assert bcs[0].is_isomorphic_to(
        _bc([[1.0, 3.0], [1.0, 4.0], [2.0, 4.0]], dtype=np_dtype)
    )


@pytest.mark.parametrize("sb_dtype, np_dtype", DTYPES, ids=["float32", "float64"])
def test_roundoff_level_domination_violation_is_clamped(sb_dtype, np_dtype):
    # d is one ULP below d' on the only pair: mathematically d = d' (an empty
    # bar), and the discrepancy is pure roundoff from computing the two sides
    # through different arithmetic. This must clamp to [w, w), not raise.
    w = np_dtype(1.5)
    just_below = np.nextafter(w, np_dtype(0.0))

    d = sb.DistanceMatrix(2, dtype=sb_dtype)
    d[0, 1] = float(just_below)
    d_prime = sb.DistanceMatrix(2, dtype=sb_dtype)
    d_prime[0, 1] = float(w)

    bcs = pers.compute_homological_kernel(d, d_prime)
    assert bcs[0].is_isomorphic_to(_bc([[w, w]], dtype=np_dtype))


def test_trivial_distance_matrices_give_empty_barcode():
    # A single point (no merges) and no points at all.
    for n in (1, 0):
        bcs = pers.compute_homological_kernel(
            sb.DistanceMatrix(n), sb.DistanceMatrix(n)
        )
        assert bcs.shape == (1,)
        assert len(bcs[0]) == 0


# --- Route consistency: distance matrices, float32 ---


def test_distmat_route_matches_pcloud_route():
    via_pcloud = pers.compute_homological_kernel(
        np.array(FLAGSHIP_POINTS), np.array(FLAGSHIP_PROJECTED)
    )
    via_distmat = pers.compute_homological_kernel(
        _distance_matrix(FLAGSHIP_POINTS), _distance_matrix(FLAGSHIP_PROJECTED)
    )
    assert via_distmat.shape == (1,)
    assert via_distmat[0].is_isomorphic_to(via_pcloud[0])


def test_float32_input_gives_float32_barcode():
    X = np.array(FLAGSHIP_POINTS, dtype=np.float32)
    bcs = pers.compute_homological_kernel(X, _diag(X))
    assert bcs.dtype == sb.barcode32
    assert bcs[0].is_isomorphic_to(_bc(FLAGSHIP_BARS, dtype=np.float32))


# --- Small-n edge cases ---


def test_two_points_give_single_bar():
    # The minimal nontrivial input: exactly one merge, one bar.
    points = np.array([[0.0, 0.0], [3.0, 4.0]])  # d = 5, projected gap m = 3.5
    bcs = pers.compute_homological_kernel(points, _diag(points))
    assert bcs[0].is_isomorphic_to(_bc([[S2 * 3.5, 5.0]]))

    d = sb.DistanceMatrix(2)
    d[0, 1] = 5.0
    d_prime = sb.DistanceMatrix(2)
    d_prime[0, 1] = 2.5
    bcs = pers.compute_homological_kernel(d, d_prime)
    assert bcs[0].is_isomorphic_to(_bc([[2.5, 5.0]]))


def test_single_point_gives_empty_barcode():
    points = np.array([[1.0, 2.0]])
    bcs = pers.compute_homological_kernel(points, _diag(points))
    assert bcs.shape == (1,)
    assert len(bcs[0]) == 0


# --- Multi-element tensors ---


def test_multi_element_tensor_computes_per_element():
    X = sb.zeros((2,), dtype=sb.pcloud64)
    X[0] = np.array(FLAGSHIP_POINTS)
    X[1] = np.array(COINCIDENT_POINTS)
    Y = sb.zeros((2,), dtype=sb.pcloud64)
    Y[0] = _diag(FLAGSHIP_POINTS)
    Y[1] = _diag(COINCIDENT_POINTS)

    bcs = pers.compute_homological_kernel(X, Y)

    assert bcs.shape == (2,)
    assert bcs[0].is_isomorphic_to(_bc(FLAGSHIP_BARS))
    assert bcs[1].is_isomorphic_to(_bc(COINCIDENT_BARS))


# --- Error paths through the public API ---


def test_missing_y_raises():
    with pytest.raises(TypeError):
        pers.compute_homological_kernel(np.array(DIAGONAL_POINTS))


def test_positive_dim_raises_not_implemented():
    X = np.array(DIAGONAL_POINTS)
    with pytest.raises(NotImplementedError):
        pers.compute_homological_kernel(X, _diag(X), dim=1)


def test_negative_dim_raises():
    X = np.array(DIAGONAL_POINTS)
    with pytest.raises(ValueError):
        pers.compute_homological_kernel(X, _diag(X), dim=-1)


def test_mixed_input_kinds_raise():
    with pytest.raises(TypeError):
        pers.compute_homological_kernel(
            np.array(DIAGONAL_POINTS), _distance_matrix(DIAGONAL_POINTS)
        )


def test_unsupported_input_types_raise():
    with pytest.raises(TypeError):
        pers.compute_homological_kernel(1, 2)


def test_dtype_mismatch_raises():
    X = np.array(DIAGONAL_POINTS, dtype=np.float32)
    Y = np.array(DIAGONAL_POINTS, dtype=np.float64)
    with pytest.raises(TypeError, match="same dtype"):
        pers.compute_homological_kernel(X, Y)


def test_outer_shape_mismatch_raises():
    X = sb.zeros((2,), dtype=sb.pcloud64)
    Y = sb.zeros((3,), dtype=sb.pcloud64)
    with pytest.raises(ValueError, match="same shape"):
        pers.compute_homological_kernel(X, Y)


def test_non_dominated_metric_raises():
    # X' scales distances up, so d' does not stay below d.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    with pytest.raises(RuntimeError):
        pers.compute_homological_kernel(X, 2.0 * X)


def test_mismatched_point_counts_raise():
    X = np.array(DIAGONAL_POINTS)
    with pytest.raises(RuntimeError):
        pers.compute_homological_kernel(X, X[:2].copy())
