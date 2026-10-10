import re
import warnings

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


def test_coordinate_queries_select_points_from_their_own_regions():
    reference = sb.PointCloud([
        [0.0],
        [4.0],
        [9.0],
    ])
    query = sb.PointCloud([
        [3.0],
        [8.0],
    ])

    # Each query has exactly one reference point within distance 1.5.
    samples = sb.random.subsample_relative(
        reference, query, n_points=1,
        distribution=sb.distributions.Uniform(0, 1.5),
    )

    npt.assert_array_equal(samples[0, 0], [[4.0]])
    npt.assert_array_equal(samples[1, 0], [[9.0]])


def test_default_keeps_every_eligible_point_when_fewer_than_requested():
    reference = sb.PointCloud([
        [0.0],
        [1.0],
        [4.0],
    ])
    query = sb.PointCloud([
        [0.5],
    ])

    # Points 0 and 1 lie at distance 0.5, inside [0, 1.5); point 4 lies outside.
    samples = sb.random.subsample_relative(
        reference, query, n_points=3, replace=False,
        distribution=sb.distributions.Uniform(0, 1.5),
    )

    # Draw order is random, so sort the two kept points.
    sample = np.asarray(samples[0, 0])
    npt.assert_array_equal(np.sort(sample, axis=0), [
        [0.0],
        [1.0],
    ])


def test_default_sampling_draws_with_replacement():
    reference = sb.PointCloud([
        [0.0],
        [4.0],
    ])
    query = sb.PointCloud([
        [0.0],
    ])

    # Only point 0 lies inside [0, 1). Without replacement, the default
    # "keep" policy would stop after one draw.
    samples = sb.random.subsample_relative(
        reference, query, n_points=3,
        distribution=sb.distributions.Uniform(0, 1),
    )

    npt.assert_array_equal(samples[0, 0], [
        [0.0],
        [0.0],
        [0.0],
    ])


def test_index_queries_keep_their_order_and_repetitions():
    reference = sb.PointCloud([
        [0.0],
        [4.0],
        [9.0],
    ])

    # Each query selects only itself. -1 is the last point and -3 the first.
    samples = sb.random.subsample_relative(
        reference, [-1, 0, -3, -1], n_points=1,
        distribution=sb.distributions.Uniform(0, 0.5),
    )

    npt.assert_array_equal(samples[0, 0], [[9.0]])
    npt.assert_array_equal(samples[1, 0], [[0.0]])
    npt.assert_array_equal(samples[2, 0], [[0.0]])
    npt.assert_array_equal(samples[3, 0], [[9.0]])


def test_no_query_uses_every_reference_point_in_order():
    reference = sb.PointCloud([
        [0.0],
        [4.0],
        [9.0],
    ])

    # Each reference point selects only itself.
    samples = sb.random.subsample_relative(
        reference, None, n_points=1,
        distribution=sb.distributions.Uniform(0, 0.5),
    )

    assert samples.shape == (3, 1)
    npt.assert_array_equal(samples[0, 0], [[0.0]])
    npt.assert_array_equal(samples[1, 0], [[4.0]])
    npt.assert_array_equal(samples[2, 0], [[9.0]])


@pytest.mark.parametrize("dtype", [sb.uint32, sb.uint64])
def test_strided_unsigned_index_tensor_selects_the_viewed_indices(dtype):
    reference = sb.PointCloud([
        [0.0],
        [4.0],
        [9.0],
    ])
    # The view holds indices 2 and 0; the backing tensor holds 2, 1, 0, 1.
    query = sb.IntTensor([2, 1, 0, 1], dtype=dtype)[::2]

    # Each query selects only itself.
    samples = sb.random.subsample_relative(
        reference, query, n_points=1,
        distribution=sb.distributions.Uniform(0, 0.5),
    )

    assert samples.shape == (2, 1)
    npt.assert_array_equal(samples[0, 0], [[9.0]])
    npt.assert_array_equal(samples[1, 0], [[0.0]])


@pytest.mark.parametrize("np_dtype", [np.float32, np.float64])
def test_float_tensor_query_selects_points_near_its_coordinates(np_dtype):
    reference = sb.PointCloud([
        [0.0],
        [4.0],
        [9.0],
    ])
    query = sb.FloatTensor(np.array([
        [8.0],
        [3.0],
    ], dtype=np_dtype))

    # Each query has exactly one reference point within distance 1.5.
    samples = sb.random.subsample_relative(
        reference, query, n_points=1, n_samples=1, replace=True,
        distribution=sb.distributions.Uniform(0, 1.5),
    )

    npt.assert_array_equal(samples[0, 0], [[9.0]])
    npt.assert_array_equal(samples[1, 0], [[4.0]])


def test_output_axes_are_query_distribution_sample():
    reference = sb.PointCloud([
        [0.0],
        [4.0],
        [9.0],
    ])
    # The first distribution selects the query point itself; the second
    # selects point 4, at distance 4 from point 0 and 5 from point 9.
    distributions = [
        sb.distributions.Uniform(0, 0.5),
        sb.distributions.Uniform(3.5, 5.5),
    ]

    samples = sb.random.subsample_relative(
        reference, [0, 2], n_points=1, n_samples=2,
        distribution=distributions,
    )

    assert samples.shape == (2, 2, 2)
    # Query 0 (point 0).
    npt.assert_array_equal(samples[0, 0, 0], [[0.0]])
    npt.assert_array_equal(samples[0, 0, 1], [[0.0]])
    npt.assert_array_equal(samples[0, 1, 0], [[4.0]])
    npt.assert_array_equal(samples[0, 1, 1], [[4.0]])
    # Query 2 (point 9).
    npt.assert_array_equal(samples[1, 0, 0], [[9.0]])
    npt.assert_array_equal(samples[1, 0, 1], [[9.0]])
    npt.assert_array_equal(samples[1, 1, 0], [[4.0]])
    npt.assert_array_equal(samples[1, 1, 1], [[4.0]])


@pytest.mark.parametrize(("query", "distribution", "expected_shape"), [
    ([0, 1], sb.distributions.Uniform(0, 0.5), (2, 3)),
    # A one-element list keeps its distribution axis.
    ([0, 1], [sb.distributions.Uniform(0, 0.5)], (2, 1, 3)),
    # Empty queries keep the remaining axes.
    ([], sb.distributions.Uniform(0, 0.5), (0, 3)),
    ([], [sb.distributions.Uniform(0, 0.5)], (0, 1, 3)),
])
def test_output_shape(query, distribution, expected_shape):
    reference = sb.PointCloud([
        [0.0],
        [4.0],
        [9.0],
    ])

    samples = sb.random.subsample_relative(
        reference, query, n_points=1, n_samples=3,
        distribution=distribution,
    )

    assert samples.shape == expected_shape


def test_each_reference_kind_and_precision_selects_by_query_index(make_pcloud_or_distmat_tensor):
    source = make_pcloud_or_distmat_tensor([
        [0.0],
        [4.0],
        [9.0],
    ])
    reference = source[()]

    # Each query selects only itself. A matrix row taken from the query's
    # position instead of its index would select 0, then 1.
    samples = sb.random.subsample_relative(
        reference, [2, 0], n_points=1,
        distribution=sb.distributions.Uniform(0, 0.5),
    )

    assert samples.dtype == source.dtype
    indices = samples.indices
    npt.assert_array_equal(np.asarray(indices[0, 0]), [2])
    npt.assert_array_equal(np.asarray(indices[1, 0]), [0])


def test_distances_are_euclidean():
    # Euclidean distances from the origin are 5, 6.36, and 6, so only point 0
    # lies inside [0, 5.5).
    reference = sb.PointCloud([
        [3.0, 4.0],
        [4.5, 4.5],
        [6.0, 0.0],
    ])
    query = sb.PointCloud([
        [0.0, 0.0],
    ])

    samples = sb.random.subsample_relative(
        reference, query, n_points=3, replace=False,
        distribution=sb.distributions.Uniform(0, 5.5),
    )

    npt.assert_array_equal(np.asarray(samples.indices[0, 0]), [0])


def uniform(x, start, end):
    """Uniform density on [start, end)."""
    return ((start <= x) & (x < end)) / (end - start)


def assert_histogram_matches_exact(counts, expected):
    """Assert each count's frequency is within five standard errors of its
    probability."""
    n_draws = counts.sum()
    frequencies = counts / n_draws
    # The tolerance assumes a normal approximation, which needs a reasonable
    # expected count both drawn and not drawn wherever the probability is
    # strictly between 0 and 1.
    certain = (expected == 0) | (expected == 1)
    assert np.all(certain | (np.minimum(expected, 1 - expected) * n_draws >= 30))
    # Zero-probability points must never be drawn.
    tolerance = 5 * np.sqrt(expected * (1 - expected) / n_draws)
    assert np.all(np.abs(frequencies - expected) <= tolerance), (
        f"frequencies {frequencies}\nexpected {expected}\ntolerance {tolerance}"
    )


def assert_histograms_agree(ours, theirs, n_samples):
    """Assert two frequency histograms from n_samples each differ by at most
    five standard errors in every bin."""
    q = (ours + theirs) / 2
    # The normal approximation needs at least 30 samples on each side of
    # every bin with 0 < q < 1.
    assert np.all((q == 0) | (q == 1) | (np.minimum(q, 1 - q) * n_samples >= 30))
    tolerance = 5 * np.sqrt(2 * q * (1 - q) / n_samples)
    assert np.all(np.abs(ours - theirs) <= tolerance), f"{ours}\n{theirs}\n{tolerance}"


def gaussian(x, mean, std):
    """Gaussian density with the given mean and standard deviation."""
    return np.exp(-((x - mean) / std)**2 / 2) / (std * np.sqrt(2 * np.pi))


@pytest.mark.parametrize(("n_reference", "n_draws"), [
    (10, 20_000),
    pytest.param(200, 10_000_000, marks=pytest.mark.statistical),
])
@pytest.mark.parametrize(("distribution", "density", "reference_range"), [
    pytest.param(
        sb.distributions.Gaussian(0, 1),
        lambda x: gaussian(x, 0, 1),
        (-1.5, 3),
        id="gaussian",
    ),
    pytest.param(
        sb.distributions.Gaussian(mean=1.5, std=0.5),
        lambda x: gaussian(x, 1.5, 0.5),
        (-1.5, 3),
        id="gaussian-mean-std",
    ),
    pytest.param(
        sb.distributions.Mixture(
            [sb.distributions.Uniform(0, 0.5), sb.distributions.Uniform(0, 3)],
            [1, 3],
        ),
        lambda x: 1 / 4 * uniform(x, 0, 0.5) + 3 / 4 * uniform(x, 0, 3),
        (-1, 8),
        id="mixture",
    ),
    pytest.param(
        sb.distributions.Mixture(
            [
                sb.distributions.Mixture(
                    [sb.distributions.Uniform(0, 1), sb.distributions.Uniform(10, 11)],
                    [1, 9],
                ),
                sb.distributions.Uniform(2, 3),
            ],
            [1, 3],
        ),
        lambda x: 1 / 4 * (1 / 10 * uniform(x, 0, 1) + 9 / 10 * uniform(x, 10, 11)) + 3 / 4 * uniform(x, 2, 3),
        (-1, 8),
        id="nested-mixture",
    ),
    pytest.param(
        sb.distributions.Mixture(
            [sb.distributions.Gaussian(0, 3), sb.distributions.Uniform(8, 10)],
            [1, 1],
        ),
        lambda x: 1 / 2 * gaussian(x, 0, 3) + 1 / 2 * uniform(x, 8, 10),
        (-1, 8),
        id="gaussian-uniform-mixture",
    ),
])
def test_draws_with_replacement_follow_the_weights(
    distribution, density, reference_range, n_reference, n_draws,
):
    x = np.linspace(*reference_range, n_reference)
    reference = sb.PointCloud(x[:, None])

    samples = sb.random.subsample_relative(
        reference, [[0.0]], n_points=n_draws,
        distribution=distribution,
        generator=sb.random.Generator(42),
    )

    weights = density(np.abs(x))
    counts = np.bincount(np.asarray(samples.indices[0, 0]), minlength=n_reference)
    assert_histogram_matches_exact(counts, weights / weights.sum())


def numpy_inclusion_frequencies(p, n_points, n_samples, generator):
    """Fraction of NumPy samples without replacement that include each point."""
    # Generator.choice without replacement draws one point at a time in
    # proportion to p among the points not yet drawn. A request for more
    # points than are eligible keeps every eligible point.
    size = min(n_points, np.count_nonzero(p))
    counts = np.zeros(p.size)
    for _ in range(n_samples):
        counts[generator.choice(p.size, size=size, replace=False, p=p)] += 1
    return counts / n_samples


@pytest.mark.parametrize(("n_reference", "n_samples", "n_points"), [
    (100, 5000, 10),
    *[
        pytest.param(200, 120_000, n_points, marks=pytest.mark.statistical)
        for n_points in [1, 10, 50, 150]
    ],
])
@pytest.mark.parametrize(("distribution", "density", "reference_range"), [
    pytest.param(
        sb.distributions.Gaussian(0, 1),
        lambda x: gaussian(x, 0, 1),
        (-1.5, 2),
        id="gaussian",
    ),
    pytest.param(
        sb.distributions.Mixture(
            [sb.distributions.Uniform(0, 1), sb.distributions.Uniform(1, 3)],
            [1, 1],
        ),
        lambda x: 1 / 2 * uniform(x, 0, 1) + 1 / 2 * uniform(x, 1, 3),
        # Descending, so the ineligible points come first.
        (5.5, -1.25),
        id="mixture",
    ),
])
def test_draws_without_replacement_follow_the_weights(
    distribution, density, reference_range, n_points, n_reference, n_samples,
):
    x = np.linspace(*reference_range, n_reference)
    reference = sb.PointCloud(x[:, None])

    samples = sb.random.subsample_relative(
        reference, [[0.0]], n_points=n_points, n_samples=n_samples, replace=False,
        distribution=distribution,
        generator=sb.random.Generator(42),
    )

    weights = density(np.abs(x))
    indices = samples.indices
    drawn = np.concatenate([np.asarray(indices[0, s]) for s in range(n_samples)])
    # Each point appears at most once per sample, so this is the fraction of
    # samples that include it.
    frequencies = np.bincount(drawn, minlength=n_reference) / n_samples
    numpy_frequencies = numpy_inclusion_frequencies(
        weights / weights.sum(), n_points, n_samples, np.random.default_rng(42),
    )
    assert_histograms_agree(frequencies, numpy_frequencies, n_samples)


@pytest.mark.parametrize("partial_kwargs", [
    pytest.param({"allow_partial": "no"}, id="no"),
    pytest.param({}, id="default"),
])
def test_one_eligible_point_fills_a_sample_with_replacement(partial_kwargs):
    reference = sb.PointCloud([
        [0.0],
        [4.0],
        [9.0],
    ])

    # Only point 1 lies within distance 1 of itself.
    samples = sb.random.subsample_relative(
        reference, [1], n_points=3, replace=True,
        distribution=sb.distributions.Uniform(0, 1),
        **partial_kwargs,
    )

    npt.assert_array_equal(np.asarray(samples.indices[0, 0]), [1, 1, 1])


@pytest.mark.parametrize("replace", [True, False])
def test_far_query_keeps_an_eligible_point(replace):
    reference = sb.PointCloud([
        [0.0],
        [1.0],
    ])

    # Point 0 has probability exp(-999.5) relative to point 1. Without
    # replacement it is never drawn first; with replacement only a uniform
    # draw of exactly 0 selects it (about 2**-64).
    samples = sb.random.subsample_relative(
        reference, [[1000.0]], n_points=1, replace=replace, allow_partial="no",
        distribution=sb.distributions.Gaussian(0, 1),
    )

    npt.assert_array_equal(np.asarray(samples.indices[0, 0]), [1])


def test_tiny_positive_weights_stay_eligible():
    reference = sb.PointCloud([
        [0.0],
        [10000.0],
    ])

    # Point 1 has log weight about -5e7 under the Gaussian and lies outside
    # the Uniform, so its weight is positive but underflows to zero.
    samples = sb.random.subsample_relative(
        reference, [0], n_points=2, replace=False, allow_partial="no",
        distribution=sb.distributions.Mixture(
            [sb.distributions.Gaussian(0, 1), sb.distributions.Uniform(0, 1)],
            [1, 1],
        ),
    )

    assert set(np.asarray(samples.indices[0, 0])) == {0, 1}


@pytest.mark.parametrize("n_samples", [
    2000,
    pytest.param(120_000, marks=pytest.mark.statistical),
])
def test_equal_far_tail_weights_keep_a_random_order(n_samples):
    reference = sb.PointCloud([
        [0.0],
        [-1e20],
        [1e20],
    ])

    # Points 1 and 2 have equal log weights of about -5e39, so each is the
    # second draw with probability 1/2.
    samples = sb.random.subsample_relative(
        reference, [0], n_points=2, n_samples=n_samples, replace=False,
        distribution=sb.distributions.Gaussian(0, 1),
        generator=sb.random.Generator(42),
    )

    indices = samples.indices
    draws = np.array([np.asarray(indices[0, s]) for s in range(n_samples)])
    first_draws = draws[:, 0]
    second_draws = draws[:, 1]
    npt.assert_array_equal(first_draws, 0)
    counts = np.bincount(second_draws, minlength=3)
    assert_histogram_matches_exact(counts, np.array([0, 1 / 2, 1 / 2]))


@pytest.mark.parametrize("distribution", [
    pytest.param(sb.distributions.Gaussian(0, 1), id="gaussian"),
    pytest.param(sb.distributions.Uniform(0, np.inf), id="unbounded-uniform"),
])
def test_infinite_matrix_distances_have_zero_weight(distribution):
    inf = np.inf
    reference = sb.DistanceMatrix(np.array([
        [0.0, 1.0, inf, inf],
        [1.0, 0.0, inf, inf],
        [inf, inf, 0.0, 2.0],
        [inf, inf, 2.0, 0.0],
    ]))

    # Points 2 and 3 are infinitely far from point 0, so only 0 and 1 are eligible.
    samples = sb.random.subsample_relative(
        reference, [0], n_points=4, replace=False, allow_partial="keep",
        distribution=distribution,
    )

    assert set(np.asarray(samples.indices[0, 0])) == {0, 1}


def test_no_partial_names_the_failing_query_and_distribution():
    reference = sb.PointCloud([
        [0.0],
        [1.0],
        [4.0],
    ])

    # Query 1 (point 2) has only itself within distance 1.5, one point short.
    with pytest.raises(
        ValueError,
        match="insufficient positive support for query 1, distribution 0",
    ):
        sb.random.subsample_relative(
            reference, [0, 2], n_points=2, replace=False, allow_partial="no",
            distribution=[
                sb.distributions.Uniform(0, 1.5),
                sb.distributions.Uniform(0, 4.5),
            ],
        )


def test_no_partial_with_replacement_raises_when_no_point_is_eligible():
    reference = sb.PointCloud([
        [0.0],
        [4.0],
        [9.0],
    ])

    # Distances 4, 0, 5 from point 1; none lies in [1, 2).
    with pytest.raises(
        ValueError,
        match="insufficient positive support for query 0, distribution 0",
    ):
        sb.random.subsample_relative(
            reference, [1], n_points=3, n_samples=1, replace=True, allow_partial="no",
            distribution=sb.distributions.Uniform(1, 2),
        )


def test_empty_cells_keep_their_axes():
    reference = sb.PointCloud([
        [0.0],
        [4.0],
    ])

    # The first distribution selects only point 0; the second selects nothing.
    samples = sb.random.subsample_relative(
        reference, [0], n_points=2, n_samples=2, replace=False, allow_partial="keep",
        distribution=[
            sb.distributions.Uniform(0, 0.5),
            sb.distributions.Uniform(10, 11),
        ],
    )

    assert samples.shape == (1, 2, 2)
    for s in range(2):
        npt.assert_array_equal(samples[0, 0, s], [[0.0]])
        assert np.asarray(samples[0, 1, s]).shape == (0, 1)


@pytest.mark.parametrize("replace", [True, False])
def test_drop_partial_applies_with_and_without_replacement(replace):
    reference = sb.PointCloud([
        [0.0],
        [1.0],
        [4.0],
    ])

    # Query 0 has exactly two eligible points, 0 and 1; query 1 (point 2) has one.
    samples = sb.random.subsample_relative(
        reference, [0, 2], n_points=2, n_samples=1, replace=replace, allow_partial="drop",
        distribution=sb.distributions.Uniform(0, 1.5),
    )

    first_query_indices = np.asarray(samples.indices[0, 0])
    assert len(first_query_indices) == 2
    assert set(first_query_indices) <= {0, 1}
    assert len(np.asarray(samples.indices[1, 0])) == 0



def test_verbose_warns_about_empty_cells():
    reference = sb.PointCloud([
        [0.0],
        [1.0],
        [4.0],
    ])

    # Query 1 (point 2) has one eligible point, so "drop" leaves its cell empty.
    with pytest.warns(UserWarning, match=re.escape(
            "Empty samples at (query, distribution) indices: [(1, 0)]")):
        sb.random.subsample_relative(
            reference, [0, 2], n_points=2, n_samples=1, replace=False, allow_partial="drop",
            distribution=sb.distributions.Uniform(0, 1.5), verbose=True,
        )


def test_quiet_mode_does_not_warn_about_empty_cells():
    reference = sb.PointCloud([
        [0.0],
        [1.0],
        [4.0],
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sb.random.subsample_relative(
            reference, [0, 2], n_points=2, n_samples=1, replace=False, allow_partial="drop",
            distribution=sb.distributions.Uniform(0, 1.5), verbose=False,
        )

def test_eligibility_is_counted_before_duplicate_removal():
    reference = sb.PointCloud([
        [0.0],
        [0.0],
        [5.0],
    ])

    # Points 0 and 1 are eligible and share coordinates, so removing
    # duplicates leaves one point after the "no" check has passed.
    samples = sb.random.subsample_relative(
        reference, [[0.0]], n_points=2, n_samples=1, replace=False, allow_partial="no",
        discard_duplicates=True,
        distribution=sb.distributions.Uniform(0, 1),
    )

    npt.assert_array_equal(samples[0, 0], [[0.0]])


def test_global_seed_replays_and_calls_advance():
    reference = sb.PointCloud(np.arange(100.0).reshape(100, 1))

    def draw_from_hundred_points():
        # Uniform(0, 100) covers every point, so all 100 are equally likely.
        samples = sb.random.subsample_relative(
            reference, [0], n_points=5, n_samples=1, replace=True,
            distribution=sb.distributions.Uniform(0, 100),
        )
        return np.asarray(samples.indices[0, 0])

    sb.random.seed(7)
    first = draw_from_hundred_points()
    second = draw_from_hundred_points()
    sb.random.seed(7)
    first_again = draw_from_hundred_points()
    second_again = draw_from_hundred_points()

    assert not np.array_equal(first, second)
    npt.assert_array_equal(first_again, first)
    npt.assert_array_equal(second_again, second)


def test_every_output_cell_draws_from_its_own_stream():
    reference = sb.PointCloud(np.arange(100.0).reshape(100, 1))

    # Equal queries and equal distributions, so only the cell position
    # can tell the streams apart.
    pointclouds = sb.random.subsample_relative(
        reference, [0, 0], n_points=5, n_samples=2, replace=True,
        distribution=[sb.distributions.Uniform(0, 100), sb.distributions.Uniform(0, 100)],
        generator=sb.random.Generator(7),
    )

    # pointclouds[query, distribution].indices[sample]
    cells = [
        pointclouds[0, 0].indices[0],
        pointclouds[0, 0].indices[1],
        pointclouds[0, 1].indices[0],
        pointclouds[0, 1].indices[1],
        pointclouds[1, 0].indices[0],
        pointclouds[1, 0].indices[1],
        pointclouds[1, 1].indices[0],
        pointclouds[1, 1].indices[1],
    ]
    # With the seed fixed, the outcome is fixed. If the streams change, two of
    # the 28 cell pairs match by chance with probability about 3e-9.
    for i in range(8):
        for j in range(i + 1, 8):
            assert not cells[i].array_equal(cells[j]), (i, j)


def test_indexed_reference_uses_its_selected_points():
    source = sb.PointCloudTensor([
        np.array([
            [0.0],
            [4.0],
            [9.0],
            [20.0],
        ]),
    ])
    # The view holds source points 3 and 1 ([20.0], [4.0]), in that order.
    reference = source[sb.NestedTensor([sb.indices([3, 1])])][0]

    # Uniform(0, 0.5) keeps only the query point itself.
    samples = sb.random.subsample_relative(
        reference, [0, 1], n_points=1, n_samples=1, replace=True,
        distribution=sb.distributions.Uniform(0, 0.5),
    )

    # samples[query].indices[sample]
    npt.assert_array_equal(samples[0, 0], [[20.0]])
    npt.assert_array_equal(samples[0].indices[0], [0])
    npt.assert_array_equal(samples[1, 0], [[4.0]])
    npt.assert_array_equal(samples[1].indices[0], [1])


def test_samples_ignore_later_writes_to_the_reference():
    reference = sb.PointCloud([
        [0.0],
        [4.0],
        [9.0],
    ])

    # Uniform(0, 0.5) keeps only the query point itself.
    samples = sb.random.subsample_relative(
        reference, [0], n_points=1, n_samples=1, replace=True,
        distribution=sb.distributions.Uniform(0, 0.5),
    )
    reference[0, 0] = 99.0

    assert reference[0, 0] == 99.0
    npt.assert_array_equal(samples[0, 0], [[0.0]])


def test_empty_distribution_list_raises():
    reference = sb.PointCloud([
        [0.0],
        [4.0],
    ])

    with pytest.raises(ValueError, match="distribution list must not be empty"):
        sb.random.subsample_relative(
            reference, [0], n_points=1, n_samples=1, replace=True, distribution=[],
        )


def test_non_distribution_argument_raises():
    reference = sb.PointCloud([
        [0.0],
        [4.0],
    ])

    with pytest.raises(TypeError, match="distribution must be a Distribution or a list of them"):
        sb.random.subsample_relative(
            reference, [0], n_points=1, n_samples=1, replace=True, distribution="gaussian",
        )
