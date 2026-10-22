from dataclasses import dataclass

import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import compute_persistent_homology
from stablebear.timeseries import TimeDelayEmbedding, Times, TimeSeries

from _element_values import TWO_ELEMENTS_OF_EVERY_DTYPE
from _round_trip import ROUND_TRIPS


def _ms_series():
    # Each observation is 10 * t, so a value carried from an earlier
    # observation stands out: x(1.5) is 10, not 15.
    return TimeSeries([0.0, 10.0, 25.0, 40.0], times=Times([0.0, 1.0, 2.5, 4.0], unit="ms"))


def test_embedding_has_a_point_wherever_a_component_reaches_an_observation():
    embedding = TimeDelayEmbedding(dimension=3, delay=1.0)(_ms_series())

    # x changes at 0.0, 1.0, 2.5, and 4.0, so (x(t - 2), x(t - 1), x(t)),
    # oldest first, changes at those times and 1 and 2 later, from 2.0 on.
    assert np.array_equal(np.asarray(embedding.times), [2.0, 2.5, 3.0, 3.5, 4.0])
    assert np.array_equal(np.asarray(embedding.values), [
        [0.0, 10.0, 10.0],    # 2.0: x(0.0), x(1.0), x(2.0)
        [0.0, 10.0, 25.0],    # 2.5: x(0.5), x(1.5), x(2.5)
        [10.0, 10.0, 25.0],   # 3.0: x(1.0), x(2.0), x(3.0)
        [10.0, 25.0, 25.0],   # 3.5: x(1.5), x(2.5), x(3.5)
        [10.0, 25.0, 40.0],   # 4.0: x(2.0), x(3.0), x(4.0)
    ])
    # (x(1.25), x(2.25), x(3.25)), between two times.
    assert np.array_equal(np.asarray(embedding(3.25)), [10.0, 10.0, 25.0])


def test_embedding_starts_where_every_component_has_a_value():
    embedding = TimeDelayEmbedding(dimension=3, delay=1.0)(_ms_series())

    assert embedding.unit == "ms"
    assert embedding.shape == (3,)
    # The domain starts at 0.0 + 2 * 1.0.
    assert np.array_equal(np.asarray(embedding(2.0)), [0.0, 10.0, 10.0])
    with pytest.raises(ValueError, match="outside the domain"):
        embedding(1.75)


def test_embedding_of_shaped_observations_adds_a_leading_axis():
    # Each observation is (10 * t, -10 * t).
    series = TimeSeries([[0.0, 0.0], [10.0, -10.0], [20.0, -20.0]], times=[0.0, 1.0, 2.0])

    embedding = TimeDelayEmbedding(dimension=2, delay=1.0)(series)

    assert embedding.shape == (2, 2)
    assert np.array_equal(np.asarray(embedding.values), [
        [[0.0, 0.0], [10.0, -10.0]],     # 1.0: x(0.0), x(1.0)
        [[10.0, -10.0], [20.0, -20.0]],  # 2.0: x(1.0), x(2.0)
    ])


def test_embedding_of_pcfs_stacks_delayed_pcfs():
    # PCFs of values 0, 10, and 20 at times 0.0, 1.0, and 2.0.
    pcfs = [sb.Pcf(np.array([[0.0, 10.0 * t], [1.0, 0.0]])) for t in [0.0, 1.0, 2.0]]
    series = TimeSeries(sb.tensor(pcfs), times=[0.0, 1.0, 2.0])

    embedding = TimeDelayEmbedding(dimension=2, delay=1.0)(series)

    assert embedding.dtype is sb.pcf64
    assert embedding.shape == (2,)
    assert np.array_equal(np.asarray(embedding.times), [1.0, 2.0])
    assert embedding.values.array_equal(sb.tensor([
        [pcfs[0], pcfs[1]],   # 1.0: x(0.0), x(1.0)
        [pcfs[1], pcfs[2]],   # 2.0: x(1.0), x(2.0)
    ]))


@pytest.mark.parametrize("make_embedding, error", [
    pytest.param(lambda: TimeDelayEmbedding(dimension=0, delay=1.0), ValueError, id="dimension-0"),
    pytest.param(lambda: TimeDelayEmbedding(dimension=2, delay=0.0), ValueError, id="delay-0"),
    pytest.param(lambda: TimeDelayEmbedding(dimension=2, delay=-1.0), ValueError, id="delay-negative"),
    pytest.param(lambda: TimeDelayEmbedding(dimension=3, delay=1.0)(
        TimeSeries([0.0, 10.0], times=[0.0, 1.0])), ValueError, id="too-short"),
])
def test_embedding_rejects_invalid_input(make_embedding, error):
    with pytest.raises(error):
        make_embedding()


def test_embedding_of_several_channels_converts_the_delay_per_channel():
    a = TimeSeries([10.0, 20.0, 30.0], times=Times([0.0, 1.0, 2.5], unit="s"))
    b = TimeSeries([-1.0, -2.0, -3.0], times=Times([500.0, 1500.0, 3000.0], unit="ms"))
    series = TimeSeries.from_channels([a, b], unit="s")

    embedding = TimeDelayEmbedding(dimension=2, delay=0.5)(series)

    # Each channel changes at its own times and 0.5 s later, from 0.5 s
    # after its first time on: A at 0.5, 1.0, 1.5, and 2.5 s, and B at 1000,
    # 1500, 2000, and 3000 ms, reported in s.
    times = embedding.times
    assert np.array_equal(np.asarray(times[0]), [0.5, 1.0, 1.5, 2.5])
    assert np.array_equal(np.asarray(times[1]), [1.0, 1.5, 2.0, 3.0])
    # (x(1.25 s), x(1.75 s)) in each channel.
    values = embedding(1.75)
    assert np.array_equal(np.asarray(values[0]), [20.0, 20.0])
    assert np.array_equal(np.asarray(values[1]), [-1.0, -2.0])
    embedding(1.0)
    embedding(2.5)
    with pytest.raises(ValueError, match="outside the domain"):
        embedding(0.75)


@dataclass(frozen=True)
class RoundingCase:
    """A series and its embedding with dimension 2 and delay 0.1."""

    times: list[float]
    values: list[float]
    embedding_times: list[float]
    points: list[list[float]]


@pytest.mark.parametrize("case", [
    # 0.2 + 0.1 is 0.30000000000000004, just above the observation at 0.3,
    # and 0.3 - 0.1 is 0.19999999999999998, just below the one at 0.2. The
    # point is recorded at the observation, 0.3.
    pytest.param(RoundingCase(
        times=[0.1, 0.2, 0.3, 0.4],
        values=[1.0, 2.0, 3.0, 4.0],
        embedding_times=[0.2, 0.3, 0.4],
        points=[
            [1.0, 2.0],   # 0.2: x(0.1), x(0.2)
            [2.0, 3.0],   # 0.3: x(0.2), x(0.3)
            [3.0, 4.0],   # 0.4: x(0.3), x(0.4)
        ],
    ), id="above-an-observation"),
    # 0.7 + 0.1 is 0.7999999999999999, just below the observation at 0.8,
    # which it snaps to. The point is recorded at 0.8.
    pytest.param(RoundingCase(
        times=[0.0, 0.7, 0.8, 0.9],
        values=[0.0, 7.0, 8.0, 9.0],
        embedding_times=[0.1, 0.7, 0.8, 0.9],
        points=[
            [0.0, 0.0],   # 0.1: x(0.0), x(0.1)
            [0.0, 7.0],   # 0.7: x(0.6), x(0.7)
            [7.0, 8.0],   # 0.8: x(0.7), x(0.8)
            [8.0, 9.0],   # 0.9: x(0.8), x(0.9)
        ],
    ), id="below-an-observation"),
    # The domain starts at 0.7 + 0.1, just below the observation at 0.8. The
    # first point stays at the start.
    pytest.param(RoundingCase(
        times=[0.7, 0.8, 0.9],
        values=[7.0, 8.0, 9.0],
        embedding_times=[0.7 + 0.1, 0.9],
        points=[
            [7.0, 8.0],   # 0.7 + 0.1: x(0.7), x(0.8)
            [8.0, 9.0],   # 0.9: x(0.8), x(0.9)
        ],
    ), id="at-the-start"),
])
def test_embedding_records_a_point_once_at_the_time_its_rounded_times_snap_to(case):
    series = TimeSeries(case.values, times=case.times)

    embedding = TimeDelayEmbedding(dimension=2, delay=0.1)(series)

    assert np.array_equal(np.asarray(embedding.times), case.embedding_times)
    assert np.array_equal(np.asarray(embedding.values), case.points)


def test_embedding_reaches_components_many_observations_back():
    series = TimeSeries(
        [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        times=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )

    embedding = TimeDelayEmbedding(dimension=2, delay=6.0)(series)

    # Each older component is six observations before the newer one.
    assert np.array_equal(np.asarray(embedding.times), [6.0, 7.0, 8.0])
    assert np.array_equal(np.asarray(embedding.values), [
        [0.0, 60.0],    # 6.0: x(0.0), x(6.0)
        [10.0, 70.0],   # 7.0: x(1.0), x(7.0)
        [20.0, 80.0],   # 8.0: x(2.0), x(8.0)
    ])


def test_embedding_allows_delayed_times_that_round_to_one_value():
    # Without snapping, -1.5 and the next float up give different points,
    # but both minus 1.0 round to -2.5.
    last = np.nextafter(-1.5, 0.0)
    # Each value is 10 times its timestamp, with -15.0 for last.
    series = TimeSeries([-30.0, -25.0, -15.0], times=[-3.0, -2.5, last], snap_tolerance=0.0)

    embedding = TimeDelayEmbedding(dimension=2, delay=1.0)(series)

    assert np.array_equal(np.asarray(embedding.times), [-2.0, -1.5, last])
    assert np.array_equal(np.asarray(embedding.values), [
        [-30.0, -25.0],   # -2.0: x(-3.0), x(-2.0)
        [-25.0, -25.0],   # -1.5: x(-2.5), x(-1.5)
        [-25.0, -15.0],   # last: x(last - 1.0), x(last), with last - 1.0 == -2.5
    ])


def test_embedding_of_an_embedding_stacks_points():
    # source x:  0.0 -> 0,  1.0 -> 10,  2.5 -> 25,  4.0 -> 40
    #
    # inner, dimension 3, delay 1: (x(t - 2), x(t - 1), x(t)) from 0.0 + 2 on
    #   2.0 -> (0, 10, 10)    2.5 -> (0, 10, 25)    3.0 -> (10, 10, 25)
    #   3.5 -> (10, 25, 25)   4.0 -> (10, 25, 40)
    #
    # outer, dimension 2, delay 1.5: (inner(t - 1.5), inner(t)) from 2.0 + 1.5 on
    inner = TimeDelayEmbedding(dimension=3, delay=1.0)(_ms_series())

    outer = TimeDelayEmbedding(dimension=2, delay=1.5)(inner)

    assert outer.shape == (2, 3)
    assert np.array_equal(np.asarray(outer.times), [3.5, 4.0])
    assert np.array_equal(np.asarray(outer.values), [
        # 3.5: inner(2.0), inner(3.5)
        [[0.0, 10.0, 10.0],
         [10.0, 25.0, 25.0]],

        # 4.0: inner(2.5), inner(4.0)
        [[0.0, 10.0, 25.0],
         [10.0, 25.0, 40.0]],
    ])


def test_embedding_evaluates_many_times_at_once():
    embedding = TimeDelayEmbedding(dimension=3, delay=1.0)(_ms_series())

    assert np.array_equal(np.asarray(embedding([4.0, 3.25])), [
        [10.0, 25.0, 40.0],   # 4.0: x(2.0), x(3.0), x(4.0)
        [10.0, 10.0, 25.0],   # 3.25: x(1.25), x(2.25), x(3.25)
    ])


def test_persistence_of_an_embedding_matches_its_stored_points():
    # Each point of a one-dimensional series embedded with dimension 3 is a
    # (3, 1) matrix: a cloud of three points on a line.
    series = TimeSeries([[0.0], [10.0], [25.0], [40.0]], times=[0.0, 1.0, 2.5, 4.0])
    embedding = TimeDelayEmbedding(dimension=3, delay=1.0)(series)
    stored = TimeSeries(embedding.values, times=embedding.times)

    result = compute_persistent_homology(embedding, max_dim=0)

    assert result.values.is_isomorphic_to(
        compute_persistent_homology(stored, max_dim=0).values, atol=0, rtol=0
    )


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
def test_embedding_survives_saving(round_trip):
    embedding = TimeDelayEmbedding(dimension=3, delay=1.0)(_ms_series())

    loaded = round_trip(embedding)

    assert loaded.array_equal(embedding)


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
@pytest.mark.parametrize("make_values", TWO_ELEMENTS_OF_EVERY_DTYPE)
def test_embeddings_of_every_dtype_survive_saving(make_values, round_trip):
    # The observations are a, b, b, a, so the points (a, b), (b, b), and
    # (b, a) differ.
    two = make_values()
    series = TimeSeries(sb.concatenate([two, two[[1, 0]]]), times=[0.0, 1.0, 2.0, 3.0])
    embedding = TimeDelayEmbedding(dimension=2, delay=1.0)(series)

    loaded = round_trip(embedding)

    assert loaded.array_equal(embedding)
