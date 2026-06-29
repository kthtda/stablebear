import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import barcode_to_stable_rank, compute_persistent_homology
from stablebear.reductions import mean
from stablebear.sampling import Gaussian, Uniform, subsample_relative


@pytest.fixture(params=[(np.float32, sb.pcloud32), (np.float64, sb.pcloud64)],
                ids=["f32", "f64"])
def float_kind(request):
    return request.param


def _ref_index_map(R):
    return {tuple(np.round(r, 5)): i for i, r in enumerate(R)}


def _sampled_indices(subs_elem, idx_map):
    return [idx_map[tuple(np.round(p, 5))] for p in np.asarray(subs_elem)]


def test_output_shape_and_dtype(float_kind):
    np_float, pcloud_dtype = float_kind
    R = np.random.default_rng(0).standard_normal((50, 3)).astype(np_float)
    X = np.random.default_rng(1).standard_normal((4, 3)).astype(np_float)

    subs = subsample_relative(R, X, sample_size=12, n_instances=7,
                     generator=sb.random.Generator(0))

    assert isinstance(subs, sb.PointCloudTensor)
    assert subs.dtype == pcloud_dtype
    assert subs.shape == (4, 7)
    assert subs[0, 0].shape == (12, 3)
    assert subs[3, 6].shape == (12, 3)


def test_zero_weight_points_never_sampled():
    # Distance filter + a hard cutoff distribution: reference points farther than
    # the cutoff get weight 0 and must never appear in any subsample.
    R = (np.arange(20, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[0.0]])
    cutoff = 5.5

    subs = subsample_relative(R, X, sample_size=8, n_instances=50,
                     distribution=lambda v: (np.asarray(v) < cutoff).astype(float),
                     generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    seen = set()
    for j in range(subs.shape[1]):
        seen.update(_sampled_indices(subs[0, j], idx_map))

    assert seen, "expected some points to be sampled"
    assert all(R[i, 0] < cutoff for i in seen)


def test_gaussian_concentrates_on_nearest():
    # A small-sigma Gaussian over distance should sample the nearest reference
    # point far more than any other.
    R = (np.arange(30, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[7.0]])  # exactly reference point index 7

    subs = subsample_relative(R, X, sample_size=20, n_instances=200,
                     distribution=Gaussian(0.0, 0.3),
                     generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    counts = np.zeros(len(R), dtype=int)
    for j in range(subs.shape[1]):
        for i in _sampled_indices(subs[0, j], idx_map):
            counts[i] += 1

    assert counts.argmax() == 7


def test_per_query_point_probabilities_differ():
    R = (np.arange(30, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[3.0], [25.0]])

    subs = subsample_relative(R, X, sample_size=20, n_instances=100,
                     distribution=Gaussian(0.0, 1.0),
                     generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    means = []
    for q in range(2):
        idxs = []
        for j in range(subs.shape[1]):
            idxs.extend(_sampled_indices(subs[q, j], idx_map))
        means.append(np.mean(idxs))

    # The query near 3 should pull from small indices, the one near 25 from large.
    assert means[0] < means[1]


def test_reproducible_with_seed(float_kind):
    np_float, _ = float_kind
    R = np.random.default_rng(2).standard_normal((40, 2)).astype(np_float)
    X = np.random.default_rng(3).standard_normal((3, 2)).astype(np_float)

    a = subsample_relative(R, X, sample_size=10, n_instances=5, generator=sb.random.Generator(7))
    b = subsample_relative(R, X, sample_size=10, n_instances=5, generator=sb.random.Generator(7))
    c = subsample_relative(R, X, sample_size=10, n_instances=5, generator=sb.random.Generator(8))

    for i in range(3):
        for j in range(5):
            assert np.array_equal(np.asarray(a[i, j]), np.asarray(b[i, j]))

    any_diff = any(
        not np.array_equal(np.asarray(a[i, j]), np.asarray(c[i, j]))
        for i in range(3) for j in range(5)
    )
    assert any_diff


def test_verbose_matches_nonverbose():
    # verbose=True drives the result through the stoppable-task progress loop;
    # it must produce exactly the same (deterministic) subsamples as verbose=False.
    R = np.random.default_rng(6).standard_normal((40, 2))
    X = np.random.default_rng(7).standard_normal((3, 2))

    quiet = subsample_relative(R, X, sample_size=10, n_instances=5,
                      generator=sb.random.Generator(11), verbose=False)
    loud = subsample_relative(R, X, sample_size=10, n_instances=5,
                     generator=sb.random.Generator(11), verbose=True)

    assert isinstance(loud, sb.PointCloudTensor)
    assert loud.shape == quiet.shape
    for i in range(3):
        for j in range(5):
            assert np.array_equal(np.asarray(loud[i, j]), np.asarray(quiet[i, j]))


def test_callable_path_matches_builtin():
    R = np.random.default_rng(4).standard_normal((60, 2))
    X = np.random.default_rng(5).standard_normal((3, 2))

    fused = subsample_relative(R, X, sample_size=8, n_instances=4,
                      distribution=Gaussian(0.0, 1.0),
                      generator=sb.random.Generator(3))
    callable_ = subsample_relative(R, X, sample_size=8, n_instances=4,
                          distribution=lambda v: np.exp(-0.5 * np.asarray(v) ** 2),
                          generator=sb.random.Generator(3))

    for i in range(3):
        for j in range(4):
            assert np.array_equal(np.asarray(fused[i, j]), np.asarray(callable_[i, j]))


def test_uniform_fused_matches_callable():
    # The fused C++ Uniform path and a constant Python distribution must agree.
    R = np.random.default_rng(6).standard_normal((50, 2))
    X = np.random.default_rng(7).standard_normal((3, 2))

    fused = subsample_relative(R, X, sample_size=8, n_instances=4,
                      distribution=Uniform(),
                      generator=sb.random.Generator(2))
    callable_ = subsample_relative(R, X, sample_size=8, n_instances=4,
                          distribution=lambda v: np.ones_like(np.asarray(v)),
                          generator=sb.random.Generator(2))

    for i in range(3):
        for j in range(4):
            assert np.array_equal(np.asarray(fused[i, j]), np.asarray(callable_[i, j]))


def test_uniform_samples_all_reference_points():
    # With equal weights every reference point should eventually be drawn.
    R = (np.arange(20, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[100.0]])  # far from R: a distance-based distribution would skew

    subs = subsample_relative(R, X, sample_size=5, n_instances=400,
                     distribution=Uniform(), generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    seen = set()
    for j in range(subs.shape[1]):
        seen.update(_sampled_indices(subs[0, j], idx_map))
    assert seen == set(range(20))


def test_uniform_disk_samples_only_within_radius():
    # Uniform(high=r) is a disk: only reference points within distance r of the
    # query may be drawn, and all of them should be (eventually).
    R = (np.arange(20, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[7.0]])  # reference index 7
    radius = 3.0

    subs = subsample_relative(R, X, sample_size=5, n_instances=400,
                     distribution=Uniform(high=radius),
                     generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    seen = set()
    for j in range(subs.shape[1]):
        seen.update(_sampled_indices(subs[0, j], idx_map))

    assert seen == {i for i in range(20) if abs(R[i, 0] - 7.0) <= radius}


def test_uniform_annulus_samples_only_within_band():
    # Uniform(low, high) is a ring: only points whose distance to the query
    # falls in [low, high] may be drawn.
    R = (np.arange(30, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[15.0]])  # reference index 15
    low, high = 4.0, 8.0

    subs = subsample_relative(R, X, sample_size=5, n_instances=500,
                     distribution=Uniform(low=low, high=high),
                     generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    seen = set()
    for j in range(subs.shape[1]):
        seen.update(_sampled_indices(subs[0, j], idx_map))

    assert seen == {i for i in range(30) if low <= abs(R[i, 0] - 15.0) <= high}


def test_uniform_disk_fused_matches_callable():
    # The fused C++ band and the equivalent Python distribution must agree.
    R = np.random.default_rng(6).standard_normal((60, 2))
    X = np.random.default_rng(7).standard_normal((3, 2))

    fused = subsample_relative(R, X, sample_size=8, n_instances=4,
                      distribution=Uniform(low=0.5, high=2.0),
                      generator=sb.random.Generator(2))
    callable_ = subsample_relative(R, X, sample_size=8, n_instances=4,
                          distribution=lambda v: ((np.asarray(v) >= 0.5)
                                                  & (np.asarray(v) <= 2.0)).astype(float),
                          generator=sb.random.Generator(2))

    for i in range(3):
        for j in range(4):
            assert np.array_equal(np.asarray(fused[i, j]), np.asarray(callable_[i, j]))


@pytest.mark.parametrize("kwargs", [{"low": -1.0}, {"high": 0.0},
                                    {"low": 2.0, "high": 1.0},
                                    {"low": 1.0, "high": 1.0}])
def test_uniform_invalid_radii_raise(kwargs):
    with pytest.raises(ValueError):
        Uniform(**kwargs)


def test_without_replacement_gives_distinct_points():
    R = (np.arange(15, dtype=np.float64)).reshape(-1, 1)
    X = np.array([[0.0]])

    subs = subsample_relative(R, X, sample_size=10, n_instances=20,
                     distribution=Uniform(), replace=False,
                     generator=sb.random.Generator(0))

    idx_map = _ref_index_map(R)
    for j in range(subs.shape[1]):
        idxs = _sampled_indices(subs[0, j], idx_map)
        assert len(idxs) == len(set(idxs))


def test_without_replacement_too_large_raises():
    R = np.zeros((5, 2))
    X = np.zeros((1, 2))
    with pytest.raises(ValueError):
        subsample_relative(R, X, sample_size=6, n_instances=1, replace=False)


def test_dimension_mismatch_raises():
    R = np.zeros((10, 3))
    X = np.zeros((2, 2))
    with pytest.raises(ValueError):
        subsample_relative(R, X, sample_size=3, n_instances=1)


def test_negative_weights_raise():
    R = np.zeros((10, 2))
    X = np.zeros((1, 2))
    with pytest.raises(ValueError):
        subsample_relative(R, X, sample_size=3, n_instances=1,
                  distribution=lambda v: -np.ones_like(np.asarray(v)))


def test_all_zero_weights_raise():
    R = np.zeros((10, 2))
    X = np.zeros((1, 2))
    with pytest.raises(ValueError):
        subsample_relative(R, X, sample_size=3, n_instances=1,
                  distribution=lambda v: np.zeros_like(np.asarray(v)))


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_counts_raise(bad):
    R = np.zeros((10, 2))
    X = np.zeros((1, 2))
    with pytest.raises(ValueError):
        subsample_relative(R, X, sample_size=bad, n_instances=1)
    with pytest.raises(ValueError):
        subsample_relative(R, X, sample_size=3, n_instances=bad)


def test_subsamples_are_indexed_views():
    R = np.random.default_rng(0).standard_normal((100, 8))
    X = np.random.default_rng(1).standard_normal((3, 8))

    subs = subsample_relative(R, X, sample_size=10, n_instances=5,
                     distribution=Gaussian(0.0, 1.0), generator=sb.random.Generator(0))

    el = subs[0, 0]
    assert el.is_indexed
    assert el.shape == (10, 8)

    idx = np.asarray(el.indices)
    assert idx.shape == (10,)
    # The materialized coordinates are exactly the referenced source rows
    # (no coordinates are copied until materialization).
    assert np.array_equal(el.to_numpy(), R[idx])


def test_pipeline_to_relative_stable_rank():
    R = np.random.default_rng(0).standard_normal((200, 2))
    X = np.random.default_rng(1).standard_normal((4, 2))

    subs = subsample_relative(R, X, sample_size=25, n_instances=30,
                     distribution=Gaussian(0.0, 1.0),
                     generator=sb.random.Generator(0))

    bcs = compute_persistent_homology(subs, max_dim=1)
    assert bcs.shape == (4, 30, 2)

    srs = barcode_to_stable_rank(bcs)
    rel = mean(srs, dim=1)
    assert rel.shape == (4, 2)
