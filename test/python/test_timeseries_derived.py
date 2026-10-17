import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import compute_persistent_homology
from stablebear.timeseries import TimeDelayEmbedding, Times, TimeSeries


def _ms_series():
    # Each observation is 10 * t, so a value carried from an earlier
    # observation stands out: x(1.5) is 10, not 15.
    return TimeSeries([0.0, 10.0, 25.0, 40.0], times=Times([0.0, 1.0, 2.5, 4.0], unit="ms"))


def test_embedding_has_a_delayed_point_at_each_observation_time():
    embedding = TimeDelayEmbedding(dimension=3, delay=1.0)(_ms_series())

    # Points (x(t - 2), x(t - 1), x(t)) at 2.5 and 4.0, oldest first.
    assert np.array_equal(np.asarray(embedding.values), [[0.0, 10.0, 25.0], [10.0, 25.0, 40.0]])
    assert np.array_equal(np.asarray(embedding(3.25)), [0.0, 10.0, 25.0])


def test_embedding_starts_where_every_component_has_a_value():
    embedding = TimeDelayEmbedding(dimension=3, delay=1.0)(_ms_series())

    assert embedding.unit == "ms"
    assert embedding.shape == (3,)
    # 0.0 and 1.0 have no value two milliseconds earlier.
    assert np.array_equal(np.asarray(embedding.times), [2.5, 4.0])
    with pytest.raises(ValueError, match="outside the domain"):
        embedding(2.0)


def test_embedding_of_shaped_observations_adds_a_leading_axis():
    series = TimeSeries([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], times=[0.0, 1.0, 2.0])

    embedding = TimeDelayEmbedding(dimension=2, delay=1.0)(series)

    assert embedding.shape == (2, 2)
    # Points at 1.0 and 2.0, each (x(t - 1), x(t)).
    assert np.array_equal(np.asarray(embedding.values), [
        [[1.0, -1.0], [2.0, -2.0]],
        [[2.0, -2.0], [3.0, -3.0]],
    ])


@pytest.mark.parametrize("make_embedding, error", [
    pytest.param(lambda: TimeDelayEmbedding(dimension=0, delay=1.0), ValueError, id="dimension-0"),
    pytest.param(lambda: TimeDelayEmbedding(dimension=2, delay=0.0), ValueError, id="delay-0"),
    pytest.param(lambda: TimeDelayEmbedding(dimension=2, delay=-1.0), ValueError, id="delay-negative"),
    pytest.param(lambda: TimeDelayEmbedding(dimension=3, delay=1.0)(
        TimeSeries([1.0, 2.0], times=[0.0, 1.0])), ValueError, id="too-short"),
    pytest.param(lambda: TimeDelayEmbedding(dimension=2, delay=1.0)(
        TimeSeries(sb.zeros((2,), dtype=sb.barcode64), times=[0.0, 1.0])), TypeError, id="barcodes"),
])
def test_embedding_rejects_invalid_input(make_embedding, error):
    with pytest.raises(error):
        make_embedding()


def test_embedding_of_several_channels_converts_the_delay_per_channel():
    a = TimeSeries([10.0, 20.0, 30.0], times=Times([0.0, 1.0, 2.5], unit="s"))
    b = TimeSeries([-1.0, -2.0, -3.0], times=Times([500.0, 1500.0, 3000.0], unit="ms"))
    series = TimeSeries.from_channels([a, b], unit="s")

    embedding = TimeDelayEmbedding(dimension=2, delay=0.5)(series)

    # A's points are at 1.0 and 2.5 s, B's at 1500 and 3000 ms, reported in s.
    times = embedding.times
    assert np.array_equal(np.asarray(times[0]), [1.0, 2.5])
    assert np.array_equal(np.asarray(times[1]), [1.5, 3.0])
    values = embedding(1.5)
    assert np.array_equal(np.asarray(values[0]), [10.0, 20.0])
    assert np.array_equal(np.asarray(values[1]), [-1.0, -2.0])
    embedding(2.5)
    with pytest.raises(ValueError, match="outside the domain"):
        embedding(1.25)


def test_embedding_components_snap_to_observations():
    # 0.3 - 0.1 is 0.19999999999999998, just below the observation at 0.2.
    series = TimeSeries([1.0, 2.0, 3.0], times=[0.1, 0.2, 0.3])

    embedding = TimeDelayEmbedding(dimension=2, delay=0.1)(series)

    assert np.array_equal(np.asarray(embedding(0.3)), [2.0, 3.0])


def test_embedding_of_an_embedding_stacks_points():
    # source x:  0.0 -> 0,  1.0 -> 10,  2.5 -> 25,  4.0 -> 40
    #
    # inner, dimension 3, delay 1: (x(t - 2), x(t - 1), x(t)) from 0.0 + 2 on
    #   2.5 -> (x(0.5), x(1.5), x(2.5)) = (0, 10, 25)
    #   4.0 -> (x(2.0), x(3.0), x(4.0)) = (10, 25, 40)
    #
    # outer, dimension 2, delay 1.5: (inner(t - 1.5), inner(t)) from 2.5 + 1.5 on
    #   4.0 -> (inner(2.5), inner(4.0)) = ((0, 10, 25), (10, 25, 40))
    inner = TimeDelayEmbedding(dimension=3, delay=1.0)(_ms_series())

    outer = TimeDelayEmbedding(dimension=2, delay=1.5)(inner)

    assert outer.shape == (2, 3)
    assert np.array_equal(np.asarray(outer.times), [4.0])
    assert np.array_equal(np.asarray(outer(4.0)), [[0.0, 10.0, 25.0], [10.0, 25.0, 40.0]])


def test_embedding_evaluates_many_times_at_once():
    embedding = TimeDelayEmbedding(dimension=3, delay=1.0)(_ms_series())

    assert np.array_equal(np.asarray(embedding([4.0, 3.25])), [[10.0, 25.0, 40.0], [0.0, 10.0, 25.0]])


def test_persistence_of_an_embedding_matches_its_stored_points():
    # Each point of a one-dimensional series embedded with dimension 3 is a
    # (3, 1) matrix: a cloud of three points on a line.
    series = TimeSeries([[0.0], [1.0], [3.0], [6.0]], times=[0.0, 1.0, 2.5, 4.0])
    embedding = TimeDelayEmbedding(dimension=3, delay=1.0)(series)
    stored = TimeSeries(embedding.values, times=embedding.times)

    result = compute_persistent_homology(embedding, max_dim=0)

    assert result.values.is_isomorphic_to(
        compute_persistent_homology(stored, max_dim=0).values, atol=0, rtol=0
    )
