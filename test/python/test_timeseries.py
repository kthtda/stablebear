
import datetime
import re
import pickle

import numpy as np
import pytest

import stablebear as sb
from stablebear.timeseries import DateTime, Domain, TimeDelayEmbedding, Times, TimeSeries, Window

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
    ("D", "d"),
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


@pytest.mark.parametrize("value, unit, expected", [
    pytest.param("2023-01-01", "d", np.datetime64("2023-01-01", "D"), id="str"),
    pytest.param(np.datetime64("2023-01-01T12:30", "m"), "min", np.datetime64("2023-01-01T12:30", "m"),
                 id="datetime64"),
    pytest.param(datetime.datetime(2023, 1, 1, 12, 30), "us", np.datetime64("2023-01-01T12:30", "us"),
                 id="datetime"),
    pytest.param(datetime.date(2023, 1, 1), "d", np.datetime64("2023-01-01", "D"), id="date"),
])
def test_date_time_keeps_the_resolution_of_its_value(value, unit, expected):
    date_time = DateTime(value)

    assert date_time.unit == unit
    assert date_time.to_datetime64() == expected
    assert date_time.to_datetime64().dtype == expected.dtype


def test_date_time_converts_back_to_the_same_datetime64():
    value = np.datetime64("2023-01-01T12:30:15.250", "ms")

    result = DateTime(value).to_datetime64()

    assert result == value
    assert result.dtype == value.dtype


def test_date_times_are_equal_only_at_the_same_resolution():
    # A string date and a date count days; a datetime counts microseconds.
    assert DateTime("2023-01-01") == DateTime(datetime.date(2023, 1, 1))
    assert DateTime("2023-01-01") != DateTime("2023-01-02")
    assert DateTime("2023-01-01") != DateTime(datetime.datetime(2023, 1, 1))


def test_date_time_survives_pickling():
    date_time = DateTime(np.datetime64("2023-01-01T12:30", "m"))

    assert pickle.loads(pickle.dumps(date_time)) == date_time


def test_date_time_repr_shows_the_date():
    assert repr(DateTime("2023-01-01T12:30")) == "DateTime('2023-01-01T12:30')"


@pytest.mark.parametrize("value", [
    pytest.param(datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc), id="timezone"),
    pytest.param(np.datetime64("2023-01-02", "W"), id="weeks"),
    pytest.param(np.datetime64("2023-01", "M"), id="months"),
    pytest.param(np.datetime64("2023", "Y"), id="years"),
    pytest.param(np.datetime64("NaT"), id="nat"),
])
def test_date_time_rejects_values_without_a_fixed_instant_or_resolution(value):
    with pytest.raises(ValueError):
        DateTime(value)


def test_date_time_rejects_a_number():
    with pytest.raises(TypeError):
        DateTime(0)


def test_times_have_no_epoch_by_default():
    assert Times([0.0, 1.5], unit="s").epoch is None


def test_times_epoch_from_a_string_equals_one_from_a_date_time():
    from_string = Times([0.0, 1.5], unit="s", epoch="2023-01-01")
    from_date_time = Times([0.0, 1.5], unit="s", epoch=DateTime("2023-01-01"))

    assert from_string.epoch == DateTime("2023-01-01")
    assert from_string == from_date_time


def test_times_keep_their_epoch_in_another_unit():
    times = Times([0.0, 1.5], unit="s", epoch="2023-01-01").to("ms")

    assert times.epoch == DateTime("2023-01-01")


def test_times_with_different_epochs_are_not_equal():
    times = Times([0.0, 1.5], unit="s", epoch="2023-01-01")

    assert times != Times([0.0, 1.5], unit="s", epoch="2023-01-02")
    assert times != Times([0.0, 1.5], unit="s")


def test_times_with_epoch_sets_replaces_and_removes_the_epoch():
    times = Times([0.0, 1.5], unit="s")

    with_epoch = times.with_epoch("2023-01-01")
    replaced = with_epoch.with_epoch(DateTime("2023-01-02"))
    removed = replaced.with_epoch(None)

    assert with_epoch.epoch == DateTime("2023-01-01")
    assert replaced.epoch == DateTime("2023-01-02")
    assert removed.epoch is None


def test_times_with_epoch_keeps_the_timestamps_and_unit():
    times = Times([0.0, 1.5], unit="ms").with_epoch("2023-01-01")

    assert sb.FloatTensor([0.0, 1.5], dtype=sb.float64).array_equal(sb.tensor(times))
    assert times.unit == "ms"


def test_times_with_epoch_rejects_an_invalid_epoch():
    with pytest.raises(ValueError):
        Times([0.0, 1.5], unit="s").with_epoch(np.datetime64("2023-01", "M"))


def _stamps():
    return np.array(["2023-01-01T06:00", "2023-01-01T07:30", "2023-01-02T06:00"], dtype="datetime64[m]")


def test_times_from_datetimes_count_whole_units_from_the_first():
    times = Times(_stamps(), unit="min")

    assert times.epoch == DateTime(np.datetime64("2023-01-01T06:00", "m"))
    assert sb.FloatTensor([0.0, 90.0, 1440.0], dtype=sb.float64).array_equal(sb.tensor(times))


def test_times_from_datetimes_are_converted_to_the_unit():
    times = Times(_stamps(), unit="h")

    assert sb.FloatTensor([0.0, 1.5, 24.0], dtype=sb.float64).array_equal(sb.tensor(times))


def test_times_from_datetimes_count_from_an_explicit_epoch():
    times = Times(_stamps(), unit="h", epoch="2023-01-01")

    assert times.epoch == DateTime("2023-01-01")
    assert sb.FloatTensor([6.0, 7.5, 30.0], dtype=sb.float64).array_equal(sb.tensor(times))


def test_times_to_datetime64_defaults_to_microseconds():
    dates = Times([0.0, 1.5], unit="h", epoch="2023-01-01").to_datetime64()

    assert dates.dtype == np.dtype("datetime64[us]")
    assert np.array_equal(dates, np.array(["2023-01-01T00:00", "2023-01-01T01:30"], dtype="datetime64[us]"))


@pytest.mark.parametrize("resolution, expected", [
    pytest.param("ns", np.array(["2023-01-01T00:00:00", "2023-01-01T00:00:01.5"], dtype="datetime64[ns]"), id="ns"),
    pytest.param("s", np.array(["2023-01-01T00:00:00", "2023-01-01T00:00:02"], dtype="datetime64[s]"), id="s"),
])
def test_times_to_datetime64_rounds_to_the_resolution(resolution, expected):
    # 1.5 s rounds to 2 s, the nearest even step.
    dates = Times([0.0, 1.5], unit="s", epoch="2023-01-01").to_datetime64(resolution)

    assert dates.dtype == expected.dtype
    assert np.array_equal(dates, expected)


def test_times_to_datetime64_needs_an_epoch():
    with pytest.raises(ValueError, match="with_epoch"):
        Times([0.0, 1.5], unit="s").to_datetime64()


def test_times_repr_shows_the_epoch():
    times = Times([0.0, 1.5], unit="s", epoch="2023-01-01")

    assert repr(times) == "Times([0.0, 1.5], unit='s', epoch=DateTime('2023-01-01'))"


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


def test_series_domain_runs_from_first_to_last_timestamp_closed():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.5, 1.5, 4.0])

    assert series.domain == Domain(0.5, 4.0, True, True)


def test_series_domain_is_in_the_query_unit():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.5, 1.5, 4.0]).to("ms")

    assert series.domain == Domain(500.0, 4000.0, True, True)


def test_series_domain_fields_by_name_and_by_unpacking():
    domain = TimeSeries([10.0, 20.0], times=[0.5, 4.0]).domain

    start, end, start_closed, end_closed = domain

    assert (domain.start, domain.end) == (0.5, 4.0)
    assert (domain.start_closed, domain.end_closed) == (True, True)
    assert (start, end, start_closed, end_closed) == (0.5, 4.0, True, True)


def test_series_domain_of_several_channels_is_their_intersection():
    a = TimeSeries([10.0, 20.0, 30.0], times=Times([0.0, 1.0, 2.5], unit="s"))
    b = TimeSeries([-10.0, -20.0], times=Times([500.0, 3000.0], unit="ms"))

    series = TimeSeries.from_channels([a, b], unit="s")

    assert series.domain == Domain(0.5, 2.5, True, True)


@pytest.mark.parametrize("closed, expected", [
    pytest.param("left", Domain(0.0, 2.0, True, False), id="open-end"),
    pytest.param("right", Domain(0.0, 2.0, False, True), id="open-start"),
])
def test_series_domain_of_several_channels_excludes_an_end_any_channel_excludes(closed, expected):
    source = TimeSeries([10.0, 20.0, 30.0, 40.0], times=[0.0, 1.0, 2.0, 3.0])
    window = Window(size=2.0, step=2.0, closed=closed, partial=False, anchor="start")(source)[0]
    observed = TimeSeries([-10.0, -20.0, -30.0], times=[0.0, 1.0, 2.0])

    series = TimeSeries.from_channels([window, observed])

    assert series.domain == expected


def test_series_epoch_is_the_epoch_of_its_times():
    series = TimeSeries([10.0, 20.0], times=Times([0.0, 1.0], unit="h", epoch="2023-01-01"))

    assert series.epoch == DateTime("2023-01-01")
    assert series.to("min").epoch == DateTime("2023-01-01")


def test_series_from_datetimes_counts_seconds_from_the_first():
    stamps = np.array(["2023-01-01T06:00", "2023-01-01T06:01", "2023-01-01T07:00"], dtype="datetime64[m]")

    series = TimeSeries([10.0, 20.0, 30.0], times=stamps)

    assert series.unit == "s"
    assert series.epoch == DateTime(np.datetime64("2023-01-01T06:00", "m"))
    assert sb.FloatTensor([0.0, 60.0, 3600.0], dtype=sb.float64).array_equal(sb.tensor(series.times))


def _dated_series():
    return TimeSeries([10.0, 20.0, 30.0], times=Times([0.0, 1.5, 4.0], unit="h", epoch="2023-01-01"))


@pytest.mark.parametrize("date, t", [
    pytest.param(np.datetime64("2023-01-01T02:00"), 2.0, id="datetime64"),
    pytest.param(datetime.datetime(2023, 1, 1, 4), 4.0, id="datetime"),
    pytest.param(datetime.date(2023, 1, 1), 0.0, id="date"),
    pytest.param("2023-01-01T02:00", 2.0, id="str"),
])
def test_series_at_a_date_equals_the_series_at_its_time(date, t):
    series = _dated_series()

    assert series(date) == series(t)


def test_series_at_an_array_of_dates_equals_the_series_at_their_times():
    series = _dated_series()
    dates = np.array([["2023-01-01T00:30", "2023-01-01T02:00"],
                      ["2023-01-01T04:00", "2023-01-01T01:30"]], dtype="datetime64[m]")

    values = series(dates)

    assert values.shape == (2, 2)
    assert values.array_equal(series([[0.5, 2.0], [4.0, 1.5]]))


@pytest.mark.parametrize("dates", [
    pytest.param(["2023-01-01T00:30", "2023-01-01T02:00"], id="strings"),
    pytest.param([datetime.datetime(2023, 1, 1, 0, 30), datetime.datetime(2023, 1, 1, 2)], id="datetimes"),
])
def test_series_at_a_list_of_dates_equals_the_series_at_their_times(dates):
    series = _dated_series()

    assert series(dates).array_equal(series([0.5, 2.0]))


def test_series_of_several_channels_at_a_date_gives_each_channel():
    a = TimeSeries([10.0, 20.0], times=Times([0.0, 1.0], unit="h", epoch="2023-01-01"))
    b = TimeSeries([-10.0, -20.0, -30.0], times=Times([0.0, 30.0, 60.0], unit="min", epoch="2023-01-01"))
    series = TimeSeries.from_channels([a, b], unit="h")

    values = series(np.datetime64("2023-01-01T00:45"))

    assert (float(values[0]), float(values[1])) == (10.0, -20.0)


def test_series_of_several_channels_at_an_array_of_dates_gives_each_channel():
    a = TimeSeries([10.0, 20.0], times=Times([0.0, 1.0], unit="h", epoch="2023-01-01"))
    b = TimeSeries([-10.0, -20.0, -30.0], times=Times([0.0, 30.0, 60.0], unit="min", epoch="2023-01-01"))
    series = TimeSeries.from_channels([a, b], unit="h")

    values = series(np.array(["2023-01-01T00:15", "2023-01-01T00:45"], dtype="datetime64[m]"))

    assert values[0].array_equal(sb.FloatTensor([10.0, 10.0], dtype=sb.float64))
    assert values[1].array_equal(sb.FloatTensor([-10.0, -20.0], dtype=sb.float64))


def test_series_without_an_epoch_cannot_be_evaluated_at_a_date():
    series = TimeSeries([10.0, 20.0], times=[0.0, 1.0])

    with pytest.raises(ValueError, match="with_epoch"):
        series(np.datetime64("2023-01-01T00:30"))


def test_series_at_a_date_takes_no_unit():
    with pytest.raises(TypeError):
        _dated_series()(np.datetime64("2023-01-01T00:30"), "h")


def test_series_rejects_a_mix_of_dates_and_numbers():
    with pytest.raises(TypeError, match="mix"):
        _dated_series()([np.datetime64("2023-01-01T00:30"), 0.5])


def test_series_rejects_a_date_with_a_timezone():
    with pytest.raises(ValueError):
        _dated_series()(datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc))


def test_series_at_a_date_outside_its_domain_names_the_dates():
    with pytest.raises(ValueError, match=re.escape("2023-01-01T05:00 is outside the domain [2023-01-01T00:00, 2023-01-01T04:00]")):
        _dated_series()(np.array(["2023-01-01T01:00", "2023-01-01T05:00"], dtype="datetime64[m]"))


def test_window_at_a_date_outside_it_names_its_dates():
    window = Window(size=2.0, step=1.0, closed="left", partial=False, anchor="start")(_dated_series())[1]

    with pytest.raises(ValueError, match=re.escape("is outside the domain [2023-01-01T01:00, 2023-01-01T03:00)")):
        window("2023-01-01T03:00")


def test_window_at_a_date_equals_its_series_at_that_date():
    series = _dated_series()
    # The window covers 01:00 to 03:00; the value changes from 10 to 20 at
    # 01:30.
    window = Window(size=2.0, step=1.0, closed="left", partial=False, anchor="start")(series)[1]

    assert window(np.datetime64("2023-01-01T01:15")) == 10.0
    assert window(np.datetime64("2023-01-01T02:30")) == 20.0


def test_series_without_an_epoch_has_none():
    assert TimeSeries([10.0, 20.0], times=[0.0, 1.0]).epoch is None


def test_series_epoch_is_one_value_for_several_channels():
    a = TimeSeries([10.0, 20.0], times=Times([0.0, 1.0], unit="h", epoch="2023-01-01"))
    b = TimeSeries([-10.0, -20.0], times=Times([0.0, 60.0], unit="min", epoch="2023-01-01"))

    assert TimeSeries.from_channels([a, b], unit="h").epoch == DateTime("2023-01-01")


def test_series_with_epoch_sets_the_epoch_of_every_channel():
    a = TimeSeries([10.0, 20.0], times=Times([0.0, 1.0], unit="h"))
    b = TimeSeries([-10.0, -20.0], times=Times([0.0, 60.0], unit="min"))

    series = TimeSeries.from_channels([a, b], unit="h").with_epoch("2023-01-01")

    assert series[0].epoch == DateTime("2023-01-01")
    assert series[1].epoch == DateTime("2023-01-01")


def test_series_with_epoch_keeps_its_times_and_values():
    series = TimeSeries([10.0, 20.0], times=[0.0, 1.5])

    dated = series.with_epoch("2023-01-01")

    assert dated.array_equal(series)
    assert dated(1.5) == 20.0


def test_series_with_epoch_none_removes_the_epoch():
    series = TimeSeries([10.0, 20.0], times=Times([0.0, 1.5], unit="s", epoch="2023-01-01"))

    assert series.with_epoch(None).epoch is None


@pytest.mark.parametrize("epoch_b", [
    pytest.param("2023-01-02", id="different"),
    pytest.param(None, id="missing"),
    # The same instant in microseconds; the epoch of channel a counts days.
    pytest.param(datetime.datetime(2023, 1, 1), id="other-resolution"),
])
def test_series_channels_must_share_their_epoch(epoch_b):
    a = TimeSeries([10.0, 20.0], times=Times([0.0, 1.0], unit="h", epoch="2023-01-01"))
    b = TimeSeries([-10.0, -20.0], times=Times([0.0, 1.0], unit="h", epoch=epoch_b))

    with pytest.raises(ValueError, match="different epochs"):
        TimeSeries.from_channels([a, b])


def test_series_origin_is_zero_for_recorded_observations():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.5, 1.5, 4.0])

    assert series.origin == 0.0


def test_series_origin_has_one_entry_per_channel():
    a = TimeSeries([10.0, 20.0], times=[0.0, 1.0])
    b = TimeSeries([-10.0, -20.0], times=Times([0.0, 1000.0], unit="ms"))

    origin = TimeSeries.from_channels([a, b], unit="s").origin

    assert sb.tensor([0.0, 0.0], dtype=sb.float64).array_equal(origin)


def test_series_origin_cannot_be_assigned_to():
    series = TimeSeries([10.0, 20.0], times=[0.0, 1.0])

    with pytest.raises(AttributeError):
        series.origin = 1.0


def test_query_just_below_an_observation_snaps_to_it():
    series = TimeSeries([1.0, 2.0, 3.0], times=[0.0, 0.25, 0.5], snap_tolerance=1e-14)

    assert series(np.nextafter(0.25, 0.0)) == 2.0


def test_query_further_below_an_observation_does_not_snap():
    series = TimeSeries([1.0, 2.0, 3.0], times=[0.0, 0.25, 0.5], snap_tolerance=1e-14)

    assert series(0.24) == 1.0


@pytest.mark.parametrize("times, t, expected", [
    # Sampled every 0.1 ns, in seconds.
    ([0.0, 1e-10, 2e-10], 1.5e-10, 20.0),
    # Microsecond spacing on epoch-sized timestamps.
    ([1.7e9, 1.7e9 + 1e-6, 1.7e9 + 2e-6], 1.7e9 + 1.5e-6, 20.0),
])
def test_snapping_does_not_cross_a_gap(times, t, expected):
    series = TimeSeries([10.0, 20.0, 30.0], times=times, snap_tolerance=1e-14)

    assert series(t) == expected


def test_a_query_one_float_spacing_below_an_epoch_timestamp_snaps_to_it():
    times = [1.7e9, 1.7e9 + 1e-6, 1.7e9 + 2e-6]
    series = TimeSeries([10.0, 20.0, 30.0], times=times, snap_tolerance=1e-14)

    assert series(np.nextafter(times[1], 0.0)) == 20.0


def test_the_snap_distance_is_relative_to_the_magnitude_of_the_timestamps():
    # With a tolerance of 2**-20, the snap distance is 2**-10 below 1024 but
    # only 2**-20 below 1.
    large = TimeSeries([1.0, 2.0], times=[0.0, 1024.0], snap_tolerance=2.0**-20)
    small = TimeSeries([1.0, 2.0], times=[0.0, 1.0], snap_tolerance=2.0**-20)

    assert large(1024.0 - 2.0**-11) == 2.0
    assert large(1024.0 - 2.0**-9) == 1.0
    assert small(1.0 - 2.0**-11) == 1.0


def test_snapping_reaches_at_most_a_quarter_of_the_gap():
    # The tolerance allows 2**-10 below 1024, but the gap is 2**-9, so a
    # quarter of it, 2**-11, is the snap distance.
    series = TimeSeries([1.0, 2.0], times=[1024.0 - 2.0**-9, 1024.0],
                        snap_tolerance=2.0**-20)

    assert series(1024.0 - 2.0**-12) == 2.0
    assert series(1024.0 - 3 * 2.0**-12) == 1.0


def test_snap_tolerance_zero_disables_snapping():
    series = TimeSeries([1.0, 2.0, 3.0], times=[0.0, 0.25, 0.5], snap_tolerance=0.0)

    assert series.snap_tolerance == 0.0
    assert series(np.nextafter(0.25, 0.0)) == 1.0


@pytest.mark.parametrize("tolerance", [-0.1, 1.0, np.nan])
def test_snap_tolerance_outside_unit_interval_raises(tolerance):
    with pytest.raises(ValueError):
        TimeSeries([1.0, 2.0], times=[0.0, 1.0], snap_tolerance=tolerance)


def test_to_changes_the_query_unit():
    series = TimeSeries([10.0, 20.0], times=Times([0.0, 0.5], unit="s")).to("ms")

    assert series.unit == "ms"
    assert np.array_equal(np.asarray(series.times), [0.0, 500.0])
    assert series(250.0) == 10.0
    assert series(500.0) == 20.0  # the last observation, in the new unit


def test_evaluate_reads_times_in_the_given_unit():
    series = TimeSeries([10.0, 20.0], times=Times([0.0, 500.0], unit="ms"))

    assert series(250000.0, "us") == 10.0
    assert series(500000.0, "us") == 20.0
    assert series.unit == "ms"


def test_evaluate_rejects_an_unknown_unit():
    series = TimeSeries([10.0, 20.0], times=Times([0.0, 0.5], unit="s"))

    with pytest.raises(ValueError, match="Unknown time unit"):
        series(0.25, "fortnight")


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
    series = TimeSeries(values, times=times, snap_tolerance=0.25)

    loaded = round_trip(series)

    assert isinstance(loaded, TimeSeries)
    assert loaded.dtype is series.dtype
    assert loaded.shape == series.shape
    assert loaded.times == series.times
    assert loaded.unit == "ms"
    assert loaded.snap_tolerance == 0.25
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
    return TimeSeries([0.0, 10.0], times=Times([0.0, 1.0], unit="ms"), snap_tolerance=0.25)


def test_series_array_equal_an_equal_series():
    assert _array_equal_series().array_equal(_array_equal_series())


@pytest.mark.parametrize("other", [
    pytest.param(TimeSeries([0.0, 20.0], times=Times([0.0, 1.0], unit="ms"), snap_tolerance=0.25),
                 id="values"),
    pytest.param(TimeSeries([0.0, 10.0], times=Times([0.0, 2.0], unit="ms"), snap_tolerance=0.25),
                 id="times"),
    pytest.param(TimeSeries([0.0, 10.0], times=Times([0.0, 1.0], unit="s"), snap_tolerance=0.25),
                 id="unit"),
    pytest.param(TimeSeries([0.0, 10.0], times=Times([0.0, 1.0], unit="ms"), snap_tolerance=0.5),
                 id="snap-tolerance"),
    pytest.param(TimeSeries([[0.0], [10.0]], times=Times([0.0, 1.0], unit="ms"), snap_tolerance=0.25),
                 id="shape"),
    pytest.param(TimeSeries.from_channels([_array_equal_series(), _array_equal_series()]), id="channels"),
    pytest.param(TimeSeries(np.array([0.0, 10.0], dtype=np.float32), times=Times([0.0, 1.0], unit="ms"),
                            snap_tolerance=0.25),
                 id="dtype"),
    pytest.param(sb.FloatTensor(np.array([0.0, 10.0])), id="tensor"),
])
def test_series_array_equal_rejects_a_different_series(other):
    assert not _array_equal_series().array_equal(other)


def _window_at_one(closed):
    # Each observation is 10 * t; the window at 1.0 holds 10.0 at 0 and 15.0
    # at 0.5, and 20.0 at 1 when its end is closed.
    series = TimeSeries([0.0, 5.0, 10.0, 15.0, 20.0], times=[0.0, 0.5, 1.0, 1.5, 2.0])
    return Window(size=1.0, step=1.0, closed=closed, partial=False, anchor="start")(series)[1]


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
    return Window(size=size, step=1.0, closed=closed, partial=False, anchor="start")(series)[1]


def test_series_array_equal_compares_where_the_end_is():
    assert not _gap_window_at_one(2.0, "left").array_equal(_gap_window_at_one(1.5, "left"))


def test_series_array_equal_compares_whether_the_end_is_closed():
    assert not _gap_window_at_one(2.0, "right").array_equal(_gap_window_at_one(2.0, "none"))


def test_series_array_equal_compares_the_channel_order():
    a = _array_equal_series()
    b = TimeSeries([5.0, 15.0], times=Times([0.0, 1.0], unit="ms"), snap_tolerance=0.25)

    assert not TimeSeries.from_channels([a, b]).array_equal(TimeSeries.from_channels([b, a]))


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
@pytest.mark.parametrize("value", [
    # The finest resolutions reach only a short way from 1970-01-01:
    # attoseconds about 9.2 s, femtoseconds 2.6 h, and picoseconds 106 days.
    pytest.param(np.datetime64("1970-01-01T00:00:01.000000000000000001", "as"), id="as"),
    pytest.param(np.datetime64("1970-01-01T01:00:00.000000000000001", "fs"), id="fs"),
    pytest.param(np.datetime64("1970-02-01T00:00:00.000000000001", "ps"), id="ps"),
    pytest.param(np.datetime64("2023-01-01T12:30:00.000000001", "ns"), id="ns"),
    pytest.param(np.datetime64("2023-01-01T12:30:00.000001", "us"), id="us"),
    pytest.param(np.datetime64("2023-01-01T12:30:00.001", "ms"), id="ms"),
    pytest.param(np.datetime64("2023-01-01T12:30:01", "s"), id="s"),
    pytest.param(np.datetime64("2023-01-01T12:30", "m"), id="min"),
    pytest.param(np.datetime64("2023-01-01T12", "h"), id="h"),
    pytest.param(np.datetime64("2023-01-01", "D"), id="d"),
    pytest.param(np.datetime64("1900-01-01", "D"), id="before-1970"),
])
@pytest.mark.parametrize("as_epoch", [
    pytest.param(lambda value: value, id="datetime64"),
    pytest.param(DateTime, id="DateTime"),
])
def test_series_epoch_survives_saving(value, as_epoch, round_trip):
    series = TimeSeries([10.0, 20.0], times=Times([0.0, 1.5], unit="s", epoch=as_epoch(value)))

    loaded = round_trip(series)

    assert loaded.epoch == DateTime(value)
    assert loaded.epoch.to_datetime64() == value
    assert loaded.epoch.to_datetime64().dtype == value.dtype


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
def test_series_without_an_epoch_loads_without_one(round_trip):
    loaded = round_trip(TimeSeries([10.0, 20.0], times=[0.0, 1.5]))

    assert loaded.epoch is None
