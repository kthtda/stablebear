import numpy.testing as npt

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
        distribution=sb.distributions.Uniform(end=1.5),
    )

    npt.assert_array_equal(samples[0, 0], [[4.0]])
    npt.assert_array_equal(samples[1, 0], [[9.0]])
