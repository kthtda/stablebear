import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import barcode_to_stable_rank, compute_persistent_homology
from stablebear.reductions import mean
from stablebear.sampling import Gaussian, Uniform, subsample_relative


def _ref_distmat(n, dim=4, seed=0, np_float=np.float64):
    """A valid (symmetric, zero-diagonal, nonnegative) distance matrix."""
    pts = np.random.default_rng(seed).standard_normal((n, dim)).astype(np_float)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).astype(np_float)
    np.fill_diagonal(d, 0.0)
    return d


@pytest.fixture(params=[(np.float32, sb.distmat32), (np.float64, sb.distmat64)],
                ids=["f32", "f64"])
def float_kind(request):
    return request.param


def test_distmat_output_shape_and_dtype(float_kind):
    np_float, distmat_dtype = float_kind
    D = _ref_distmat(40, np_float=np_float)
    dm = sb.DistanceMatrix.from_dense(D)

    query = np.array([0, 5, 10], dtype=np.uint64)
    subs = subsample_relative(dm, query, sample_size=12, n_instances=7,
                              generator=sb.random.Generator(0))

    assert isinstance(subs, sb.DistanceMatrixTensor)
    assert subs.dtype == distmat_dtype
    assert subs.shape == (3, 7)
    assert subs[0, 0].size == 12
    assert subs[2, 6].size == 12


def test_distmat_query_none_uses_all_rows():
    D = _ref_distmat(20)
    dm = sb.DistanceMatrix.from_dense(D)
    subs = subsample_relative(dm, sample_size=5, n_instances=3,
                              generator=sb.random.Generator(0))
    assert subs.shape == (20, 3)   # one row per reference point


def test_distmat_subsamples_are_indexed_views():
    D = _ref_distmat(50)
    dm = sb.DistanceMatrix.from_dense(D)
    subs = subsample_relative(dm, sample_size=10, n_instances=4,
                              distribution=Gaussian(0.0, 1.0),
                              generator=sb.random.Generator(0))
    el = subs[0, 0]
    assert el.is_indexed
    idx = np.asarray(el.indices)
    assert idx.shape == (10,)
    # The materialized principal submatrix is exactly D over the drawn indices
    # (no entries copied until materialization).
    assert np.allclose(el.materialize().to_dense(), D[np.ix_(idx, idx)])


def test_distmat_callable_path_matches_builtin():
    # The fused C++ Gaussian and an equivalent Python callable must draw the same
    # subsamples for a fixed seed.
    D = _ref_distmat(60)
    dm = sb.DistanceMatrix.from_dense(D)

    fused = subsample_relative(dm, sample_size=8, n_instances=4,
                               distribution=Gaussian(0.0, 1.0),
                               generator=sb.random.Generator(3))
    callable_ = subsample_relative(dm, sample_size=8, n_instances=4,
                                   distribution=lambda v: np.exp(-0.5 * np.asarray(v) ** 2),
                                   generator=sb.random.Generator(3))

    for i in range(fused.shape[0]):
        for j in range(4):
            assert np.array_equal(np.asarray(fused[i, j].indices),
                                  np.asarray(callable_[i, j].indices))


def test_distmat_zero_weight_points_never_sampled():
    # A hard distance cutoff: reference points farther than the cutoff from the
    # query row must never appear in any subsample.
    D = _ref_distmat(40, seed=1)
    dm = sb.DistanceMatrix.from_dense(D)
    q = 7
    cutoff = np.median(D[q])

    subs = subsample_relative(dm, np.array([q], dtype=np.uint64),
                              sample_size=5, n_instances=80,
                              distribution=lambda v: (np.asarray(v) < cutoff).astype(float),
                              generator=sb.random.Generator(0))

    seen = set()
    for j in range(subs.shape[1]):
        seen.update(int(i) for i in np.asarray(subs[0, j].indices))
    assert seen
    assert all(D[q, i] < cutoff for i in seen)


def test_distmat_per_query_points_differ():
    # Two different query rows should pull from different neighbourhoods.
    D = _ref_distmat(60, seed=2)
    dm = sb.DistanceMatrix.from_dense(D)
    qa, qb = 3, 40

    subs = subsample_relative(dm, np.array([qa, qb], dtype=np.uint64),
                              sample_size=10, n_instances=60,
                              distribution=Gaussian(0.0, 0.5),
                              generator=sb.random.Generator(0))

    drawn = []
    for row in range(2):
        idxs = []
        for j in range(subs.shape[1]):
            idxs.extend(int(i) for i in np.asarray(subs[row, j].indices))
        drawn.append(idxs)

    # The two query points' subsamples should not be identical sets.
    assert set(drawn[0]) != set(drawn[1])


def test_distmat_without_replacement_distinct():
    D = _ref_distmat(15)
    dm = sb.DistanceMatrix.from_dense(D)
    subs = subsample_relative(dm, np.array([0], dtype=np.uint64),
                              sample_size=10, n_instances=20,
                              distribution=Uniform(), replace=False,
                              generator=sb.random.Generator(0))
    for j in range(subs.shape[1]):
        idx = np.asarray(subs[0, j].indices)
        assert len(set(int(i) for i in idx)) == len(idx)


def test_distmat_tensor_of_one_is_accepted():
    D = _ref_distmat(25)
    dm = sb.DistanceMatrixTensor.from_numpy([D])    # shape (1,)
    subs = subsample_relative(dm, sample_size=6, n_instances=3,
                              generator=sb.random.Generator(0))
    assert isinstance(subs, sb.DistanceMatrixTensor)
    assert subs.shape == (25, 3)


def test_distmat_bad_query_raises():
    D = _ref_distmat(10)
    dm = sb.DistanceMatrix.from_dense(D)
    with pytest.raises(ValueError):
        subsample_relative(dm, np.array([10], dtype=np.uint64),  # out of range
                           sample_size=3, n_instances=1)
    with pytest.raises(ValueError):
        subsample_relative(dm, np.array([], dtype=np.uint64),    # empty
                           sample_size=3, n_instances=1)


def test_distmat_query_by_int_list():
    # A plain Python integer list of row indices is accepted as the query.
    D = _ref_distmat(30)
    dm = sb.DistanceMatrix.from_dense(D)
    subs = subsample_relative(dm, [0, 5, 10], sample_size=6, n_instances=3,
                              generator=sb.random.Generator(0))
    assert subs.shape == (3, 3)


def test_distmat_coordinate_query_raises():
    # A 2-D coordinate query is meaningless for a distance matrix.
    D = _ref_distmat(20)
    dm = sb.DistanceMatrix.from_dense(D)
    coords = np.random.default_rng(0).standard_normal((3, 4))
    with pytest.raises(ValueError):
        subsample_relative(dm, coords, sample_size=3, n_instances=1)


def test_distmat_multi_matrix_reference_raises():
    D = _ref_distmat(8)
    dm = sb.DistanceMatrixTensor.from_numpy([D, D])   # two matrices
    with pytest.raises(ValueError):
        subsample_relative(dm, sample_size=3, n_instances=1)


def test_distmat_pipeline_to_relative_stable_rank():
    D = _ref_distmat(120, seed=5)
    dm = sb.DistanceMatrix.from_dense(D)
    query = np.array([0, 30, 60, 90], dtype=np.uint64)

    subs = subsample_relative(dm, query, sample_size=25, n_instances=30,
                              distribution=Gaussian(0.0, 1.0),
                              generator=sb.random.Generator(0))
    assert subs.shape == (4, 30)

    bcs = compute_persistent_homology(subs, max_dim=1)
    assert bcs.shape == (4, 30, 2)

    srs = barcode_to_stable_rank(bcs)
    rel = mean(srs, dim=1)
    assert rel.shape == (4, 2)
