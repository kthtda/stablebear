
import numpy as np
import pytest

import stablebear as sb
from stablebear.timeseries import TimeDelayEmbedding, Times, TimeSeries, Window

from _element_values import TWO_ELEMENTS_OF_EVERY_DTYPE
from _round_trip import ROUND_TRIPS


def test_times_keeps_values_and_unit():
    times = Times([0.0, 1.5, 4.0], unit="ms")

    assert np.asarray(times).dtype == np.float64
    assert np.array_equal(np.asarray(times), [0.0, 1.5, 4.0])
    assert times.unit == "ms"


@pytest.mark.parametrize("times, unit", [
    ([0.0, 0.0], "s"),  # duplicate timestamps
    ([1.0, 0.0], "s"),  # decreasing timestamps
    ([0.0, np.nan], "s"),
    ([0.0, np.inf], "s"),
    ([[0.0, 1.0]], "s"),  # two-dimensional
    ([0.0, 1.0], "yr"),  # unknown unit
])
def test_times_rejects_invalid_input(times, unit):
    with pytest.raises(ValueError):
        Times(times, unit=unit)



@pytest.mark.parametrize("alias, unit", [
    ("sec", "s"),
    ("seconds", "s"),
    ("hr", "h"),
    ("msec", "ms"),
    ("\u00b5s", "us"),  # micro sign
    ("\u03bcs", "us"),  # Greek small letter mu
])
def test_times_accepts_unit_aliases(alias, unit):
    assert Times([0.0, 1.0], unit=alias).unit == unit


def test_times_cannot_be_assigned_to():
    times = Times([0.0, 1.0], unit="s")

    with pytest.raises(TypeError):
        times[0] = 5.0

    assert np.array_equal(np.asarray(times), [0.0, 1.0])


@pytest.mark.parametrize("times, unit, target, expected", [
    ([0.0, 1.5], "min", "s", [0.0, 90.0]),
    ([0.0, 1.5], "fs", "as", [0.0, 1500.0]),
    ([0.0, 2.25], "as", "fs", [0.0, 0.00225]),
])
def test_times_converts_units(times, unit, target, expected):
    converted = Times(times, unit=unit).to(target)

    assert np.array_equal(np.asarray(converted), expected)
    assert converted.unit == target


def test_series_defaults_to_seconds_and_one_channel():
    # This test checks the default unit and channel name.
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    assert series.unit == "s"
    assert series.n_channels == 1
    assert series.shape == ()
    assert len(series) == 3
    assert np.array_equal(np.asarray(series.values), [10.0, 20.0, 30.0])


def test_series_keeps_explicit_unit():
    series = TimeSeries([1.0, 2.0], times=Times([0.0, 0.5], unit="ms"))

    assert series.unit == "ms"


@pytest.mark.parametrize("times", [
    [0.0, 1.0],
    [0.0, 1.0, 2.0, 3.0],
])
def test_series_needs_one_timestamp_per_observation(times):
    with pytest.raises(ValueError):
        TimeSeries([10.0, 20.0, 30.0], times=times)


@pytest.mark.parametrize("values, dtype", [
    (np.array([1.0, 2.0], dtype=np.float32), sb.float32),
    (np.array([1.0, 2.0], dtype=np.float64), sb.float64),
    (np.array([1, 2], dtype=np.int32), sb.int32),
    (np.array([1, 2], dtype=np.int64), sb.int64),
    (np.array([True, False]), sb.boolean),
])
def test_series_value_precision(values, dtype):
    series = TimeSeries(values, times=[0.0, 1.0])

    assert series.dtype is dtype


def test_series_of_points():
    values = [
        [1.0, -1.0],
        [2.0, -2.0],
        [3.0, -3.0],
    ]
    series = TimeSeries(values, times=[0.0, 1.0, 2.0])

    assert series.shape == (2,)
    assert len(series) == 3
    assert np.array_equal(np.asarray(series.values), values)


def test_series_of_matrices():
    values = [
        [[1.0, 2.0],
         [3.0, 4.0]],

        [[5.0, 6.0],
         [7.0, 8.0]],
    ]
    series = TimeSeries(values, times=[0.0, 1.0])

    assert series.shape == (2, 2)
    assert np.array_equal(np.asarray(series.values), values)



@pytest.mark.parametrize("make_values", TWO_ELEMENTS_OF_EVERY_DTYPE)
def test_series_of_every_element_type(make_values):
    values = make_values()
    series = TimeSeries(values, times=[0.0, 1.0])

    assert series.dtype is values.dtype
    assert len(series) == 2
    assert series.values.array_equal(values)


@pytest.mark.parametrize("values", [
    pytest.param(
        [sb.PointCloud(np.zeros((2, 2))), sb.PointCloud(np.ones((2, 2)))],
        id="list of point clouds",
    ),
    pytest.param(np.array(["a", "b"]), id="strings"),
    pytest.param(
        sb.NestedTensor([
            sb.FloatTensor(np.array([1.0, 2.0])),
            sb.FloatTensor(np.array([1.0])),
        ]),
        id="nested tensor",
    ),
])
def test_series_rejects_unsupported_values(values):
    with pytest.raises(TypeError):
        TimeSeries(values, times=[0.0, 1.0])


def test_series_evaluates_to_previous_value():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    assert series(0.0) == 10.0
    assert series(1.0) == 10.0
    assert series(1.5) == 20.0
    assert series(3.75) == 20.0
    assert series(4.0) == 30.0
    assert series(1) == 10.0


@pytest.mark.parametrize("t", [-0.5, 4.5])
def test_series_rejects_times_outside_observations(t):
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    with pytest.raises(ValueError, match="outside the domain"):
        series(t)


def test_empty_series_cannot_be_evaluated():
    series = TimeSeries(sb.zeros((0,), dtype=sb.pcloud64), times=[])

    assert len(series) == 0
    assert series.dtype is sb.pcloud64
    with pytest.raises(ValueError, match="without observations"):
        series(0.0)


def test_singleton_series_evaluates_at_its_time():
    series = TimeSeries([7.0], times=[2.0])

    assert series(2.0) == 7.0


def test_query_just_below_an_observation_snaps_to_it():
    series = TimeSeries([1.0, 2.0, 3.0], times=[0.0, 0.25, 0.5], snap_tolerance=1e-6,
                        snap_eps=1024)

    assert series(np.nextafter(0.25, 0.0)) == 2.0


def test_query_further_below_an_observation_does_not_snap():
    series = TimeSeries([1.0, 2.0, 3.0], times=[0.0, 0.25, 0.5], snap_tolerance=1e-6,
                        snap_eps=1024)

    assert series(0.24) == 1.0


@pytest.mark.parametrize("times, t, expected", [
    # Sampled every 0.1 ns, in seconds.
    ([0.0, 1e-10, 2e-10], 1.5e-10, 20.0),
    # Microsecond spacing on epoch-sized timestamps.
    ([1.7e9, 1.7e9 + 1e-6, 1.7e9 + 2e-6], 1.7e9 + 1.5e-6, 20.0),
])
def test_snapping_does_not_cross_a_gap(times, t, expected):
    series = TimeSeries([10.0, 20.0, 30.0], times=times, snap_tolerance=1e-6,
                        snap_eps=1024)

    assert series(t) == expected


def test_snapping_is_capped_for_wide_gaps():
    series = TimeSeries([1.0, 2.0], times=[0.0, 86400.0], snap_tolerance=1e-6,
                        snap_eps=1024)

    assert series(np.nextafter(86400.0, 0.0)) == 2.0
    assert series(86400.0 - 0.05) == 1.0


def test_snap_tolerance_zero_disables_snapping():
    series = TimeSeries([1.0, 2.0, 3.0], times=[0.0, 0.25, 0.5], snap_tolerance=0.0,
                        snap_eps=1024)

    assert series.snap_tolerance == 0.0
    assert series(np.nextafter(0.25, 0.0)) == 1.0


@pytest.mark.parametrize("tolerance", [-0.1, 1.0])
def test_snap_tolerance_outside_unit_interval_raises(tolerance):
    with pytest.raises(ValueError):
        TimeSeries([1.0, 2.0], times=[0.0, 1.0], snap_tolerance=tolerance)


def test_snap_eps_bounds_the_snap_distance():
    times = [0.0, 1.0]
    query = 1.0 - 4 * np.finfo(np.float64).eps
    loose = TimeSeries([1.0, 2.0], times=times, snap_tolerance=0.5,
                       snap_eps=8.0)
    tight = TimeSeries([1.0, 2.0], times=times, snap_tolerance=0.5,
                       snap_eps=2.0)

    assert loose.snap_eps == 8.0
    assert loose(query) == 2.0
    assert tight(query) == 1.0


@pytest.mark.parametrize("eps", [-1.0, np.inf, np.nan])
def test_snap_eps_must_be_finite_and_nonnegative(eps):
    with pytest.raises(ValueError):
        TimeSeries([1.0, 2.0], times=[0.0, 1.0], snap_eps=eps)


def test_to_changes_the_query_unit():
    series = TimeSeries([10.0, 20.0], times=Times([0.0, 0.5], unit="s")).to("ms")

    assert series.unit == "ms"
    assert np.array_equal(np.asarray(series.times), [0.0, 500.0])
    assert series(250.0) == 10.0
    assert series(500.0) == 20.0  # the last observation, in the new unit


def _channel_a():
    return TimeSeries([10.0, 20.0, 30.0], times=Times([0.0, 1.0, 2.5], unit="s"))


def _channel_b():
    return TimeSeries([-1.0, -2.0, -3.0], times=Times([500.0, 1500.0, 3000.0], unit="ms"))


def test_from_channels_keeps_every_channel():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    assert series.n_channels == 2
    assert series.unit == "s"


def _point_cloud_channel():
    clouds = sb.zeros((2,), dtype=sb.pcloud64)
    return TimeSeries(clouds, times=[0.0, 1.0])


@pytest.mark.parametrize("make_channels, unit, error", [
    (lambda: [_channel_a(), _channel_b()], None, ValueError),  # units differ
    (lambda: [], "s", ValueError),
    (lambda: [TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")],
     "s", ValueError),  # a multi-channel series as a channel
    (lambda: [_channel_a(), _point_cloud_channel()], "s", TypeError),
])
def test_from_channels_rejects_invalid_channels(make_channels, unit, error):
    with pytest.raises(error):
        TimeSeries.from_channels(make_channels(), unit=unit)


@pytest.mark.parametrize("t", [
    0.25,  # only channel A covers it
    2.75,  # only channel B covers it
])
def test_several_channels_evaluate_only_where_all_are_observed(t):
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    # Both ends of the shared domain, 0.5 (B's first) and 2.5 (A's last), work.
    assert series(0.5)[1][()] == -1.0
    assert series(2.5)[0][()] == 30.0
    with pytest.raises(ValueError, match="outside the domain"):
        series(t)


def test_times_of_several_channels_are_in_the_query_unit():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    times = series.times

    assert isinstance(times, sb.NestedTensor)
    assert times.shape == (2,)
    assert np.array_equal(np.asarray(times[0]), [0.0, 1.0, 2.5])
    assert np.array_equal(np.asarray(times[1]), [0.5, 1.5, 3.0])


@pytest.mark.parametrize("t, expected", [
    (1.25, [20.0, -1.0]),  # between B's observations at 500 and 1500 ms
    (1.5, [20.0, -2.0]),  # at B's observation at 1500 ms
])
def test_several_channels_evaluate_to_one_leaf_per_channel(t, expected):
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    values = series(t)

    assert isinstance(values, sb.NestedTensor)
    assert values.dtype is sb.float64
    assert values.shape == (2,)
    assert [values[0].shape, values[1].shape] == [(), ()]
    assert [values[0][()], values[1][()]] == expected


def test_each_channel_snaps_by_its_own_settings():
    a = TimeSeries([10.0, 20.0, 30.0], times=Times([0.0, 1.0, 2.5], unit="s"),
                   snap_tolerance=0.0)
    series = TimeSeries.from_channels([a, _channel_b()], unit="s")

    # Just below channel A's 1.0 s and channel B's 1500 ms.
    assert series(np.nextafter(1.0, 0.0))[0][()] == 10.0
    assert series(np.nextafter(1.5, 0.0))[1][()] == -2.0


def test_several_point_cloud_channels_evaluate_to_their_clouds():
    first = sb.PointCloudTensor([
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ], dtype=sb.pcloud64)
    second = sb.PointCloudTensor([
        np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]]),
    ], dtype=sb.pcloud64)
    series = TimeSeries.from_channels([
        TimeSeries(first, times=[0.0, 1.0]),
        TimeSeries(second, times=[0.0]),
    ])

    values = series(0.0)

    assert values.dtype is sb.pcloud64
    assert np.array_equal(np.asarray(values[0][()]), [[0.0, 0.0]])
    assert np.array_equal(np.asarray(values[1][()]), [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])


def test_many_times_keep_the_caller_order_and_shape():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    assert np.array_equal(np.asarray(series([4.0, 0.0, 1.0, 1.5])), [30.0, 10.0, 10.0, 20.0])
    assert np.array_equal(np.asarray(series([[0.0, 4.0], [1.5, 1.0]])),
                          [[10.0, 30.0], [20.0, 10.0]])


def test_many_times_on_several_channels():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    values = series([1.25, 1.5])

    assert isinstance(values, sb.NestedTensor)
    assert np.array_equal(np.asarray(values[0]), [20.0, 20.0])
    assert np.array_equal(np.asarray(values[1]), [-1.0, -2.0])


def test_many_times_snap_and_check_the_domain_like_single_times():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])
    queries = [0.0, np.nextafter(1.5, 0.0)]

    assert np.array_equal(np.asarray(series(queries)), [series(t) for t in queries])
    with pytest.raises(ValueError, match="outside the domain"):
        series([0.0, 4.5])
    with pytest.raises(ValueError, match="outside the domain"):
        series([0.0, np.nan, 4.0])


@pytest.mark.parametrize("index", [1, -1])
def test_selecting_one_channel_keeps_the_query_unit(index):
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    selected = series[index]

    assert selected.n_channels == 1
    assert selected.unit == "s"
    assert selected(1.5) == -2.0


def test_selecting_several_channels_keeps_their_order():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    values = series[[1, 0]](1.5)

    assert [values[0][()], values[1][()]] == [-2.0, 20.0]


def test_selection_keeps_each_channel_snap_settings():
    a = TimeSeries([10.0, 20.0, 30.0], times=Times([0.0, 1.0, 2.5], unit="s"),
                   snap_tolerance=0.25)
    series = TimeSeries.from_channels([a, _channel_b()], unit="s")

    assert series[0].snap_tolerance == 0.25
    assert series[1].snap_tolerance != series[0].snap_tolerance
    assert np.array_equal(np.asarray(series.snap_tolerance),
                          [series[0].snap_tolerance, series[1].snap_tolerance])


def test_values_are_a_copy():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.0, 2.0])

    values = series.values
    values[1] = 0.0

    assert values[1] == 0.0
    assert series(1.0) == 20.0


@pytest.mark.parametrize("make_values", [
    p for p in TWO_ELEMENTS_OF_EVERY_DTYPE
    if p.id in ("pcloud64", "distmat64", "symmat64")
])
@pytest.mark.parametrize("read", [
    pytest.param(lambda values, series, embedding: values[0], id="input"),
    pytest.param(lambda values, series, embedding: series.values[0], id="values"),
    pytest.param(lambda values, series, embedding: series(0.0), id="evaluation"),
    pytest.param(lambda values, series, embedding: embedding.values[0, 0], id="embedding values"),
])
def test_mutating_an_observation_leaves_the_series_unchanged(make_values, read):
    values = make_values()
    expected = np.asarray(values[0]).copy()
    series = TimeSeries(values, times=[0.0, 1.0])
    embedding = TimeDelayEmbedding(dimension=2, delay=1.0)(series)

    observation = read(values, series, embedding)
    observation[0, 1] = 99.0

    assert observation[0, 1] == 99.0
    assert np.array_equal(np.asarray(series(0.0)), expected)
    assert np.array_equal(np.asarray(embedding.values[0, 0]), expected)


def test_values_of_several_channels_have_one_leaf_per_channel():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    values = series.values

    assert isinstance(values, sb.NestedTensor)
    assert np.array_equal(np.asarray(values[0]), [10.0, 20.0, 30.0])
    assert np.array_equal(np.asarray(values[1]), [-1.0, -2.0, -3.0])


def test_shapes_and_counts_of_several_channels():
    points = TimeSeries([[1.0, -1.0], [2.0, -2.0]], times=[0.0, 1.0])
    series = TimeSeries.from_channels([_channel_a(), points])

    assert series.shape == ((), (2,))
    assert np.array_equal(np.asarray(series.n_observations), [3, 2])
    assert series[1].n_observations == 2


def test_len_of_several_channels_raises():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    with pytest.raises(ValueError, match="several channels"):
        len(series)


def test_channel_index_out_of_range_raises():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    with pytest.raises(IndexError):
        series[2]


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
@pytest.mark.parametrize("as_shape", [False, True], ids=["scalar", "shape-2"])
@pytest.mark.parametrize("make_values", TWO_ELEMENTS_OF_EVERY_DTYPE)
def test_one_channel_survives_saving(make_values, as_shape, round_trip):
    values = make_values()
    if as_shape:
        # Two observations of shape (2,): [a, b] and [b, a].
        values = sb.concatenate([values, values[[1, 0]]]).reshape((2, 2))
    times = Times(0.25 * np.arange(values.shape[0]), unit="ms")
    series = TimeSeries(values, times=times, snap_tolerance=0.25, snap_eps=8.0)

    loaded = round_trip(series)

    assert isinstance(loaded, TimeSeries)
    assert loaded.dtype is series.dtype
    assert loaded.shape == series.shape
    assert loaded.times == series.times
    assert loaded.unit == "ms"
    assert (loaded.snap_tolerance, loaded.snap_eps) == (0.25, 8.0)
    assert loaded.values.array_equal(values)


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
def test_several_channels_survive_saving(round_trip):
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    loaded = round_trip(series)

    assert loaded.n_channels == 2
    assert loaded.unit == "s"
    for i in range(2):
        assert np.array_equal(np.asarray(loaded.times[i]), np.asarray(series.times[i]))
    for t in (0.5, 1.5):
        assert [loaded(t)[i][()] for i in range(2)] == [series(t)[i][()] for i in range(2)]


def test_series_dtype_is_one_object_per_inner_dtype():
    assert sb.series(sb.float64) is sb.series(sb.float64)
    assert sb.series(sb.float64) is not sb.series(sb.float32)
    assert sb.series(sb.float64).inner is sb.float64


def _array_equal_series():
    return TimeSeries([0.0, 10.0], times=Times([0.0, 1.0], unit="ms"), snap_tolerance=0.25, snap_eps=8.0)


def test_series_array_equal_an_equal_series():
    assert _array_equal_series().array_equal(_array_equal_series())


@pytest.mark.parametrize("other", [
    pytest.param(TimeSeries([0.0, 20.0], times=Times([0.0, 1.0], unit="ms"), snap_tolerance=0.25, snap_eps=8.0),
                 id="values"),
    pytest.param(TimeSeries([0.0, 10.0], times=Times([0.0, 2.0], unit="ms"), snap_tolerance=0.25, snap_eps=8.0),
                 id="times"),
    pytest.param(TimeSeries([0.0, 10.0], times=Times([0.0, 1.0], unit="s"), snap_tolerance=0.25, snap_eps=8.0),
                 id="unit"),
    pytest.param(TimeSeries([0.0, 10.0], times=Times([0.0, 1.0], unit="ms"), snap_tolerance=0.5, snap_eps=8.0),
                 id="snap-tolerance"),
    pytest.param(TimeSeries([0.0, 10.0], times=Times([0.0, 1.0], unit="ms"), snap_tolerance=0.25, snap_eps=16.0),
                 id="snap-eps"),
    pytest.param(TimeSeries([[0.0], [10.0]], times=Times([0.0, 1.0], unit="ms"), snap_tolerance=0.25, snap_eps=8.0),
                 id="shape"),
    pytest.param(TimeSeries.from_channels([_array_equal_series(), _array_equal_series()]), id="channels"),
    pytest.param(TimeSeries(np.array([0.0, 10.0], dtype=np.float32), times=Times([0.0, 1.0], unit="ms"),
                            snap_tolerance=0.25, snap_eps=8.0),
                 id="dtype"),
    pytest.param(sb.FloatTensor(np.array([0.0, 10.0])), id="tensor"),
])
def test_series_array_equal_rejects_a_different_series(other):
    assert not _array_equal_series().array_equal(other)


def _window_at_one(closed):
    # Each observation is 10 * t; the window at 1.0 holds 10.0 at 0 and 15.0
    # at 0.5, and 20.0 at 1 when its end is closed.
    series = TimeSeries([0.0, 5.0, 10.0, 15.0, 20.0], times=[0.0, 0.5, 1.0, 1.5, 2.0])
    return Window(size=1.0, step=1.0, closed=closed, partial=False)(series)[1]


def test_series_array_equal_compares_the_start():
    # The windows differ only in whether their start is closed.
    assert not _window_at_one("both").array_equal(_window_at_one("right"))


def test_series_array_equal_compares_the_end():
    # The window [0, 1) can be evaluated after its last observation at 0.5;
    # the series cannot.
    assert not _window_at_one("left").array_equal(TimeSeries([10.0, 15.0], times=[0.0, 0.5]))


def _gap_window_at_one(size, closed):
    # Each observation is 10 * t, with none at 3.0; the window at 1.0 holds
    # 10.0 at 0 and 20.0 at 1 and ends after its last observation.
    series = TimeSeries([0.0, 10.0, 20.0, 40.0], times=[0.0, 1.0, 2.0, 4.0])
    return Window(size=size, step=1.0, closed=closed, partial=False)(series)[1]


def test_series_array_equal_compares_where_the_end_is():
    assert not _gap_window_at_one(2.0, "left").array_equal(_gap_window_at_one(1.5, "left"))


def test_series_array_equal_compares_whether_the_end_is_closed():
    assert not _gap_window_at_one(2.0, "right").array_equal(_gap_window_at_one(2.0, "none"))


def test_series_array_equal_compares_the_channel_order():
    a = _array_equal_series()
    b = TimeSeries([5.0, 15.0], times=Times([0.0, 1.0], unit="ms"), snap_tolerance=0.25, snap_eps=8.0)

    assert not TimeSeries.from_channels([a, b]).array_equal(TimeSeries.from_channels([b, a]))
