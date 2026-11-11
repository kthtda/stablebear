
import datetime
import pickle
import re

import numpy as np
import pytest

import stablebear as sb
from stablebear.timeseries import (DateTime, DateTimeTensor, Domain, TimeDelta, TimeDeltaTensor, TimeSeries,
                                   TimeSeriesDomainError, cut_windows, delay_embed, to_point_clouds)
from stablebear.timeseries.series import TimeSeriesTensor

from _element_values import TWO_ELEMENTS_OF_EVERY_DTYPE
from _round_trip import ROUND_TRIPS


@pytest.mark.parametrize("durations, expected", [
    pytest.param([TimeDelta(0, "s"), TimeDelta(1500, "ms"), TimeDelta(3, "s")],
                 TimeDeltaTensor([0.0, 1500.0, 3000.0], "ms"), id="time_delta"),
    pytest.param([datetime.timedelta(0), datetime.timedelta(seconds=1.5), datetime.timedelta(seconds=3)],
                 TimeDeltaTensor([0.0, 1_500_000.0, 3_000_000.0], "us"), id="timedelta"),
])
def test_series_times_of_durations_are_in_their_finest_unit(durations, expected):
    series = TimeSeries([10.0, 20.0, 30.0], times=durations)

    assert series.times.array_equal(expected)


def test_series_at_a_time_delta_tensor_equals_series_at_its_timedelta64():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor(sb.IntTensor([0, 90]), "min"))

    assert series.array_equal(TimeSeries([10.0, 20.0], times=np.array([0, 90], dtype="timedelta64[m]")))


def test_series_rejects_durations_with_nat():
    with pytest.raises(ValueError, match="NaT"):
        TimeSeries([10.0, 20.0], times=np.array([0, "NaT"], dtype="timedelta64[s]"))


def test_series_takes_durations_as_times():
    series = TimeSeries([10.0, 20.0, 30.0], times=np.array([0, 1, 2], dtype="timedelta64[m]"))

    assert series.times.array_equal(TimeDeltaTensor([0.0, 1.0, 2.0], "min"))

    assert series.epoch is None


@pytest.mark.parametrize("times, unit", [
    ([0.0, 0.0], "s"),  # duplicate timestamps
    ([1.0, 0.0], "s"),  # decreasing timestamps
    ([0.0, np.nan], "s"),
    ([0.0, np.inf], "s"),
    ([[0.0, 1.0]], "s"),  # two-dimensional
    ([0.0, 1.0], "yr"),  # unknown unit
])
def test_series_rejects_invalid_times(times, unit):
    with pytest.raises(ValueError):
        TimeSeries([10.0, 20.0], times=(times, unit))


def test_series_rejects_durations_with_a_unit_as_times():
    with pytest.raises(TypeError, match="own unit"):
        TimeSeries([10.0, 20.0], times=(np.array([0, 1], dtype="timedelta64[h]"), "min"))



def test_series_of_numbers_has_no_unit_and_one_channel():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    assert series.unit is None
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


def test_date_times_at_one_instant_are_equal_at_any_resolution():
    # A string date counts days; a datetime counts microseconds.
    day = DateTime("2023-01-01")
    microseconds = DateTime(datetime.datetime(2023, 1, 1))

    assert day == microseconds
    assert not day != microseconds
    assert hash(day) == hash(microseconds)
    assert DateTime("2023-01-01") != DateTime("2023-01-02")


def test_date_times_before_1678_and_in_ns_are_unequal():
    # 1659 in nanoseconds, where NumPy would compare them, is out of int64.
    assert DateTime("1659-01-01T00:00:00") != DateTime(np.datetime64("1990-01-01T00:00:00", "ns"))


def test_dates_of_a_dated_series_start_at_its_epoch():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.5], "h"), epoch="2023-01-01")

    assert series.datetimes[0] == series.epoch


def test_date_time_survives_pickling():
    date_time = DateTime(np.datetime64("2023-01-01T12:30", "m"))

    assert pickle.loads(pickle.dumps(date_time)) == date_time


def test_date_time_repr_shows_the_date():
    assert repr(DateTime("2023-01-01T12:30")) == "DateTime('2023-01-01T12:30')"


@pytest.mark.parametrize("value", [
    pytest.param(np.datetime64("2023-01-02", "W"), id="weeks"),
    pytest.param(np.datetime64("2023-01", "M"), id="months"),
    pytest.param(np.datetime64("2023", "Y"), id="years"),
    pytest.param(np.datetime64("NaT", "s"), id="nat"),
])
def test_date_time_rejects_values_without_a_fixed_instant_or_resolution(value):
    with pytest.raises(ValueError):
        DateTime(value)


def test_date_time_with_a_timezone_is_its_utc_instant():
    plus_two = datetime.timezone(datetime.timedelta(hours=2))

    assert DateTime(datetime.datetime(2023, 1, 1, 12, tzinfo=plus_two)) == DateTime(datetime.datetime(2023, 1, 1, 10))


def test_date_time_rejects_a_number():
    with pytest.raises(TypeError):
        DateTime(0)


def test_date_time_tensor_of_a_datetime64_array_is_in_its_resolution():
    instants = DateTimeTensor(np.array(["2023-01-01T00:30", "2023-01-01T01:30"], dtype="datetime64[m]"))

    assert instants.unit == "min"
    assert instants.shape == (2,)
    assert isinstance(instants.count, sb.IntTensor)


def test_date_time_tensor_of_strings_is_in_their_finest_resolution():
    instants = DateTimeTensor(["2023-01-01", "2023-01-01T00:30:15"])

    expected = np.array(["2023-01-01T00:00:00", "2023-01-01T00:30:15"], dtype="datetime64[s]")
    assert instants.unit == "s"
    assert np.array_equal(instants.to_datetime64(), expected)


def test_date_time_tensor_indexes_to_date_times():
    instants = DateTimeTensor(["2023-01-01T00:30", "2023-01-01T01:30"])

    assert instants[1] == DateTime("2023-01-01T01:30")
    assert len(instants) == 2


def test_date_time_tensor_compares_each_instant():
    earlier = DateTimeTensor(["2023-01-01T00:30", "2023-01-01T01:30"]) < DateTime("2023-01-01T01:00")

    assert earlier.array_equal(sb.BoolTensor([True, False]))


def test_date_time_tensor_is_unhashable():
    with pytest.raises(TypeError, match="unhashable"):
        hash(DateTimeTensor(["2023-01-01T00:30", "2023-01-01T01:30"]))


def test_date_time_tensor_survives_pickling():
    instants = DateTimeTensor(["2023-01-01T00:30", "2023-01-01T01:30"])

    assert pickle.loads(pickle.dumps(instants)).array_equal(instants)


@pytest.mark.parametrize("value, unit, expected", [
    pytest.param(np.timedelta64(90, "m"), "min", np.timedelta64(90, "m"), id="timedelta64"),
    pytest.param(datetime.timedelta(hours=1), "us", np.timedelta64(3_600_000_000, "us"), id="timedelta"),
])
def test_time_delta_keeps_the_resolution_of_its_value(value, unit, expected):
    time_delta = TimeDelta(value)

    assert time_delta.unit == unit
    assert time_delta.to_timedelta64() == expected
    assert time_delta.to_timedelta64().dtype == expected.dtype


def test_time_delta_from_a_count_and_a_unit():
    time_delta = TimeDelta(3, "h")

    assert time_delta.count == 3
    assert time_delta.unit == "h"
    assert time_delta == TimeDelta(np.timedelta64(3, "h"))


def test_time_deltas_of_one_length_are_equal_in_any_unit():
    assert TimeDelta(1, "h") == TimeDelta(60, "min")
    assert not TimeDelta(1, "h") != TimeDelta(60, "min")
    assert hash(TimeDelta(1, "h")) == hash(TimeDelta(60, "min"))
    assert TimeDelta(1.5, "h") == TimeDelta(90, "min")
    assert hash(TimeDelta(1.5, "h")) == hash(TimeDelta(90, "min"))
    assert TimeDelta(1, "h") != TimeDelta(61, "min")


def test_time_delta_equals_the_timedelta64_of_its_length():
    assert TimeDelta(1, "h") == np.timedelta64(1, "h")
    assert TimeDelta(1, "h") == np.timedelta64(60, "m")
    assert TimeDelta(1, "h") == datetime.timedelta(hours=1)


def test_time_delta_is_unequal_to_nat_and_to_other_values():
    assert TimeDelta(1, "h") != np.timedelta64("NaT")
    assert not TimeDelta(1, "h") == np.timedelta64("NaT")
    assert TimeDelta(1, "h") != 1
    assert DateTime("2023-01-01") != "2023-01-01"


def test_time_delta_survives_pickling():
    time_delta = TimeDelta(90, "min")

    assert pickle.loads(pickle.dumps(time_delta)) == time_delta


def test_time_delta_repr_shows_count_and_unit():
    assert repr(TimeDelta(90, "min")) == "TimeDelta(90, 'min')"


@pytest.mark.parametrize("value", [
    pytest.param(np.timedelta64(1, "W"), id="weeks"),
    pytest.param(np.timedelta64(1, "M"), id="months"),
    pytest.param(np.timedelta64(1, "Y"), id="years"),
    pytest.param(np.timedelta64("NaT", "s"), id="nat"),
])
def test_time_delta_rejects_values_without_a_fixed_length(value):
    with pytest.raises(ValueError):
        TimeDelta(value)


def test_time_delta_keeps_a_fractional_count():
    time_delta = TimeDelta(1.5, "h")

    assert time_delta.count == 1.5
    assert time_delta.unit == "h"


def test_time_delta_of_a_whole_float_has_an_integer_count():
    assert repr(TimeDelta(3.0, "h")) == "TimeDelta(3, 'h')"


def test_time_delta_with_a_fractional_count_converts_to_microseconds():
    assert TimeDelta(1.5, "h").to_timedelta64() == np.timedelta64(5_400_000_000, "us")
    assert TimeDelta(1.5, "h").to_timedelta64().dtype == np.dtype("timedelta64[us]")


def test_time_delta_converts_to_a_given_resolution():
    assert TimeDelta(1, "h").to_timedelta64("s") == np.timedelta64(3600, "s")
    # A third of a second, to the nearest millisecond
    assert TimeDelta(1 / 3, "s").to_timedelta64("ms") == np.timedelta64(333, "ms")


@pytest.mark.parametrize("count", [
    pytest.param(np.nan, id="nan"),
    pytest.param(np.inf, id="inf"),
])
def test_time_delta_rejects_a_count_that_is_not_finite(count):
    with pytest.raises(ValueError):
        TimeDelta(count, "h")


def test_time_delta_rejects_an_unknown_unit():
    # NumPy's "m" is minutes, but it could be read as months.
    with pytest.raises(ValueError, match="Unknown time unit"):
        TimeDelta(3, "m")


@pytest.mark.parametrize("alias, unit", [
    pytest.param("hours", "h", id="hours"),
    pytest.param("minutes", "min", id="minutes"),
    pytest.param("\u00b5s", "us", id="micro-sign"),
])
@pytest.mark.parametrize("make", [
    pytest.param(lambda unit: TimeDelta(90, unit), id="TimeDelta"),
    pytest.param(lambda unit: TimeDeltaTensor([30, 90], unit), id="TimeDeltaTensor"),
])
def test_time_deltas_in_a_unit_alias_are_in_its_short_name(make, alias, unit):
    assert make(alias).unit == unit


def test_time_delta_rejects_a_number_without_a_unit():
    with pytest.raises(TypeError):
        TimeDelta(3)


def test_time_delta_rejects_a_timedelta64_with_a_unit():
    with pytest.raises(TypeError, match="own unit"):
        TimeDelta(np.timedelta64(5, "ns"), "s")


def test_time_delta_rejects_a_boolean_with_a_unit():
    with pytest.raises(TypeError):
        TimeDelta(True, "s")


def test_time_delta_tensor_of_integer_counts_keeps_them_as_an_int_tensor():
    time_delta = TimeDeltaTensor([[30, 90],
                            [240, 60]], "min")

    assert isinstance(time_delta.count, sb.IntTensor)
    assert time_delta.count.array_equal(sb.IntTensor([[30, 90],
                                                      [240, 60]]))
    assert time_delta.unit == "min"
    assert time_delta.shape == (2, 2)


def test_time_delta_tensor_of_float_counts_keeps_them_as_a_float_tensor():
    time_delta = TimeDeltaTensor(sb.FloatTensor([0.5, 1.5]), "h")

    assert time_delta.count.array_equal(sb.FloatTensor([0.5, 1.5]))


def test_time_delta_tensor_of_a_timedelta64_array_is_in_its_resolution():
    time_delta = TimeDeltaTensor(np.array([30, 90], dtype="timedelta64[m]"))

    assert time_delta.array_equal(TimeDeltaTensor([30, 90], "min"))


def test_time_delta_tensor_of_time_deltas_equals_one_of_their_timedelta64_array():
    time_delta = TimeDeltaTensor([TimeDelta(30, "min"), TimeDelta(2, "h")])

    assert time_delta.array_equal(TimeDeltaTensor(np.array([30, 120], dtype="timedelta64[m]")))


def test_time_delta_tensor_indexes_to_time_deltas():
    time_delta = TimeDeltaTensor([[30, 90],
                            [240, 60]], "min")

    assert time_delta[1, 0] == TimeDelta(240, "min")
    assert time_delta[0].array_equal(TimeDeltaTensor([30, 90], "min"))
    assert len(time_delta) == 2
    assert [element.count for element in time_delta[1]] == [240, 60]


def test_time_delta_tensor_of_float_counts_converts_to_microseconds():
    durations = TimeDeltaTensor([0.5, 1.5], "h").to_timedelta64()

    assert np.array_equal(durations, np.array([1_800_000_000, 5_400_000_000], dtype="timedelta64[us]"))
    assert durations.dtype == np.dtype("timedelta64[us]")


def test_time_delta_tensor_rounds_integer_counts_at_ties_to_even():
    durations = TimeDeltaTensor([500, 1500, 2500, -1500], "ms").to_timedelta64("s")

    assert np.array_equal(durations, np.array([0, 2, 2, -2], dtype="timedelta64[s]"))


def test_time_delta_tensor_rounds_float_counts_by_their_exact_value():
    # Each count is slightly more than a half millisecond in binary, so it
    # rounds up, while its product with 1000 rounds to the tie.
    durations = TimeDeltaTensor([0.0005, 0.0015, 0.0025, 0.0035], "s").to_timedelta64("ms")

    assert np.array_equal(durations, np.array([1, 2, 3, 4], dtype="timedelta64[ms]"))


# Units to finer, to coarser, and to the same resolution.
_CONVERSIONS = [
    pytest.param("h", "us", id="h-to-us"),
    pytest.param("s", "ms", id="s-to-ms"),
    pytest.param("ms", "s", id="ms-to-s"),
    pytest.param("ns", "us", id="ns-to-us"),
    pytest.param("us", "us", id="us-to-us"),
]


@pytest.mark.parametrize("unit, resolution", _CONVERSIONS)
def test_time_delta_tensor_of_integer_counts_rounds_as_each_count_does(unit, resolution):
    rng = np.random.default_rng(0)
    counts = rng.integers(-10**9, 10**9, 1000)

    durations = TimeDeltaTensor(counts, unit).to_timedelta64(resolution)

    assert durations.tolist() == [TimeDelta(int(count), unit).to_timedelta64(resolution) for count in counts]


@pytest.mark.parametrize("unit, resolution", _CONVERSIONS)
def test_time_delta_tensor_of_float_counts_rounds_as_each_count_does(unit, resolution):
    rng = np.random.default_rng(0)
    # Halves are ties in every resolution, and their neighbors are near them.
    halves = rng.integers(-10**6, 10**6, 300) + 0.5
    counts = np.concatenate([rng.normal(0, 1e6, 300), halves, np.nextafter(halves, np.inf), halves / 1000,
                             np.nextafter(halves / 1000, -np.inf)])

    durations = TimeDeltaTensor(sb.FloatTensor(counts), unit).to_timedelta64(resolution)

    assert durations.tolist() == [TimeDelta(float(count), unit).to_timedelta64(resolution) for count in counts]


def test_time_delta_tensor_raises_for_a_count_beyond_its_resolution():
    with pytest.raises(OverflowError):
        TimeDeltaTensor([1, 2**62], "s").to_timedelta64("ns")


def test_time_delta_tensor_is_unhashable():
    with pytest.raises(TypeError, match="unhashable"):
        hash(TimeDeltaTensor([30, 90], "min"))


def test_time_delta_tensor_survives_pickling():
    time_delta = TimeDeltaTensor([0.5, 1.5], "h")

    assert pickle.loads(pickle.dumps(time_delta)).array_equal(time_delta)


def test_time_delta_tensor_repr_shows_counts_and_unit():
    assert repr(TimeDeltaTensor([30, 90], "min")) == "TimeDeltaTensor([30, 90], 'min')"


@pytest.mark.parametrize("value, unit", [
    pytest.param([1.0, np.nan], "h", id="nan"),
    pytest.param(np.array([1, "NaT"], dtype="timedelta64[m]"), None, id="nat"),
])
def test_time_delta_tensor_rejects_counts_that_are_not_finite(value, unit):
    with pytest.raises(ValueError):
        TimeDeltaTensor(value, unit)


def test_time_delta_tensor_rejects_counts_that_are_not_numbers():
    with pytest.raises(TypeError):
        TimeDeltaTensor([True, False], "h")


@pytest.mark.parametrize("durations", [
    pytest.param(np.array([0, 1], dtype="timedelta64[h]"), id="timedelta64 array"),
    pytest.param([np.timedelta64(1, "h"), np.timedelta64(30, "m")], id="list of timedelta64"),
])
def test_time_delta_tensor_rejects_durations_with_a_unit(durations):
    with pytest.raises(TypeError, match="own unit"):
        TimeDeltaTensor(durations, "min")


def test_time_delta_tensor_plus_a_time_delta_is_in_the_finer_unit():
    total = TimeDeltaTensor([30, 90], "min") + TimeDelta(1, "h")

    assert total.array_equal(TimeDeltaTensor([90, 150], "min"))


def test_time_delta_tensor_multiplies_each_count():
    product = TimeDeltaTensor([30, 90], "min") * 0.5

    assert product.array_equal(TimeDeltaTensor([15.0, 45.0], "min"))


def test_time_delta_tensor_compares_each_length():
    shorter = TimeDeltaTensor([30, 90], "min") < TimeDelta(1, "h")

    assert shorter.array_equal(sb.BoolTensor([True, False]))


def test_time_delta_tensors_compare_each_length_for_equality():
    minutes = TimeDeltaTensor([30, 90], "min")

    assert (minutes == TimeDeltaTensor([0.5, 2.0], "h")).array_equal(sb.BoolTensor([True, False]))
    assert (minutes != TimeDeltaTensor([0.5, 2.0], "h")).array_equal(sb.BoolTensor([False, True]))
    assert (minutes == TimeDelta(90, "min")).array_equal(sb.BoolTensor([False, True]))


def test_array_equal_tells_lengths_in_different_units_apart():
    assert not TimeDeltaTensor([1], "h").array_equal(TimeDeltaTensor([60], "min"))
    assert TimeDeltaTensor([1], "h").array_equal(TimeDeltaTensor([1], "h"))


def test_date_time_tensors_compare_each_instant_for_equality():
    days = DateTimeTensor(np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[D]"))
    seconds = DateTimeTensor(np.array(["2023-01-01T00:00:00", "2023-01-02T00:00:01"], dtype="datetime64[s]"))

    assert (days == seconds).array_equal(sb.BoolTensor([True, False]))
    assert not days.array_equal(seconds)


def test_date_time_plus_time_delta_tensor_gives_each_instant():
    instants = DateTime("2023-01-01") + TimeDeltaTensor([30, 90], "min")

    assert instants.array_equal(DateTimeTensor(["2023-01-01T00:30", "2023-01-01T01:30"]))


def test_date_time_tensor_minus_a_date_time_is_a_time_delta_tensor():
    durations = DateTimeTensor(["2023-01-01T00:30", "2023-01-01T01:30"]) - DateTime("2023-01-01")

    assert durations.array_equal(TimeDeltaTensor([30, 90], "min"))


def test_date_time_tensor_minus_a_float_count_is_in_microseconds():
    instants = DateTimeTensor(["2023-01-01T01:00", "2023-01-01T02:00"]) - TimeDelta(0.5, "min")

    assert instants.array_equal(DateTimeTensor(np.array(["2023-01-01T00:59:30", "2023-01-01T01:59:30"],
                                                        dtype="datetime64[us]")))


def test_time_deltas_add_and_subtract_in_the_finer_unit():
    assert TimeDelta(1, "h") + TimeDelta(30, "min") == TimeDelta(90, "min")
    assert TimeDelta(1, "h") - TimeDelta(30, "min") == TimeDelta(30, "min")


def test_time_delta_with_a_float_count_adds_in_the_finer_unit():
    # 1.5 h is 90.0 minutes; 90.0 + 30 is a whole number of minutes
    assert TimeDelta(1.5, "h") + TimeDelta(30, "min") == TimeDelta(120, "min")


def test_time_delta_adds_numpy_and_python_durations():
    assert np.timedelta64(30, "m") + TimeDelta(1, "h") == TimeDelta(90, "min")
    assert TimeDelta(1, "h") + datetime.timedelta(minutes=30) == TimeDelta(5_400_000_000, "us")
    assert TimeDelta(1, "h") - np.timedelta64(30, "m") == TimeDelta(30, "min")
    assert np.timedelta64(90, "m") - TimeDelta(1, "h") == TimeDelta(30, "min")


def test_time_delta_negates_and_multiplies_in_its_unit():
    assert -TimeDelta(90, "min") == TimeDelta(-90, "min")
    assert TimeDelta(30, "min") * 3 == TimeDelta(90, "min")
    assert 3 * TimeDelta(30, "min") == TimeDelta(90, "min")
    assert TimeDelta(3, "h") * 0.5 == TimeDelta(1.5, "h")


def test_time_deltas_compare_lengths():
    assert TimeDelta(1, "h") < TimeDelta(61, "min")
    assert TimeDelta(1, "h") <= TimeDelta(60, "min")
    assert TimeDelta(1, "h") >= TimeDelta(60, "min")
    assert TimeDelta(1.5, "h") > TimeDelta(89, "min")


def test_time_delta_rejects_adding_a_number():
    with pytest.raises(TypeError):
        _ = TimeDelta(1, "h") + 1  # pyright: ignore[reportOperatorIssue]


def test_date_time_plus_time_delta_is_at_the_finer_resolution():
    assert DateTime("2023-01-01T00:00") + TimeDelta(90, "s") == DateTime(np.datetime64("2023-01-01T00:01:30", "s"))
    assert TimeDelta(90, "s") + DateTime("2023-01-01T00:00") == DateTime(np.datetime64("2023-01-01T00:01:30", "s"))
    assert DateTime("2023-01-01T00:00") - TimeDelta(1, "h") == DateTime(np.datetime64("2022-12-31T23:00", "m"))


def test_date_time_plus_time_delta_with_a_float_count_is_in_microseconds():
    assert DateTime("2023-01-01T00:00") + TimeDelta(1.5, "h") == DateTime(np.datetime64("2023-01-01T01:30", "us"))


def test_date_time_plus_numpy_duration():
    assert DateTime("2023-01-01T00:00") + np.timedelta64(90, "m") == DateTime(np.datetime64("2023-01-01T01:30", "m"))


def test_date_time_minus_date_time_is_a_time_delta():
    assert DateTime("2023-01-02") - DateTime("2023-01-01") == TimeDelta(1, "d")
    assert DateTime("2023-01-01T01:30") - DateTime("2023-01-01") == TimeDelta(90, "min")


def test_date_times_compare_instants():
    assert DateTime("2023-01-01") < DateTime("2023-01-01T00:01")
    assert DateTime("2023-01-01") <= DateTime(datetime.datetime(2023, 1, 1))
    assert DateTime("2023-01-01") >= DateTime(datetime.datetime(2023, 1, 1))
    assert DateTime("2023-01-02") > DateTime("2023-01-01T23:59")


def test_date_times_before_1678_and_in_ns_compare_instants():
    # 1659 in nanoseconds, where NumPy would compare them, is out of int64.
    early = DateTime("1659-01-01T00:00:00")
    late = DateTime(np.datetime64("1990-01-01T00:00:00", "ns"))

    assert early < late
    assert late > early
    assert (DateTimeTensor(np.array(["1659-01-01T00:00:00", "2000-01-01T00:00:00"], dtype="datetime64[s]"))
            < late).array_equal(sb.BoolTensor([True, False]))


def test_date_time_plus_a_duration_out_of_int64_at_the_finer_resolution_raises():
    # 1659 is before the range of nanoseconds in int64.
    with pytest.raises(OverflowError, match="ns"):
        _ = DateTime("1659-01-01T00:00:00") + np.timedelta64(1, "ns")


def test_date_time_minus_a_date_time_out_of_int64_at_the_finer_resolution_raises():
    with pytest.raises(OverflowError, match="ns"):
        _ = DateTime(np.datetime64("1990-01-01T00:00:00", "ns")) - DateTime("1659-01-01T00:00:00")


def test_time_delta_tensors_of_350_years_and_of_ns_compare_lengths():
    # 350 years in nanoseconds are out of int64.
    years = TimeDeltaTensor([350 * 365], "d")

    assert (years > TimeDelta(1, "ns")).array_equal(sb.BoolTensor([True]))


def test_time_delta_tensor_plus_a_duration_out_of_int64_at_the_finer_resolution_raises():
    with pytest.raises(OverflowError, match="ns"):
        _ = TimeDeltaTensor([350 * 365], "d") + TimeDelta(1, "ns")


def test_series_rejects_an_invalid_epoch():
    with pytest.raises(ValueError):
        TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.5], "s"), epoch=np.datetime64("2023-01", "M"))


def test_series_at_a_date_time_tensor_equals_series_at_its_datetime64():
    dates = DateTimeTensor(["2023-01-01T06:00", "2023-01-01T07:30"])

    series = TimeSeries([10.0, 20.0], times=dates)

    assert series.array_equal(TimeSeries([10.0, 20.0], times=dates.to_datetime64()))
    assert series.epoch == TimeSeries([10.0, 20.0], times=dates.to_datetime64()).epoch


def test_series_takes_a_date_time_tensor_as_times():
    dates = DateTimeTensor(["2023-01-01T06:00", "2023-01-01T07:30"])

    series = TimeSeries([10.0, 20.0], times=dates)

    assert series.times.array_equal(TimeDeltaTensor([0.0, 90.0], "min"))
    assert series.epoch == DateTime("2023-01-01T06:00")


def test_series_counts_its_dates_from_its_epoch():
    dates = DateTimeTensor(["2023-01-01T06:00", "2023-01-01T07:30"])

    series = TimeSeries([10.0, 20.0], times=dates, epoch="2023-01-01")

    assert series.epoch == DateTime("2023-01-01")
    assert series(TimeDelta(360.0, "min")) == 10.0
    assert series(TimeDelta(450.0, "min")) == 20.0
    assert series.domain == TimeSeries([10.0, 20.0], times=dates).domain


def test_series_of_durations_with_an_epoch_equals_the_series_with_that_epoch():
    times = TimeDeltaTensor([0.0, 1.5], "h")

    series = TimeSeries([10.0, 20.0], times=times, epoch="2023-01-01")

    assert series.array_equal(TimeSeries([10.0, 20.0], times=times).with_epoch("2023-01-01"))
    assert series.epoch == DateTime("2023-01-01")


def test_series_rejects_an_epoch_tensor():
    with pytest.raises(TypeError, match="DateTimeTensor"):
        TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.5], "h"),
                   epoch=DateTimeTensor(["2023-01-01", "2023-01-02"]))


def test_series_keeps_explicit_unit():
    series = TimeSeries([1.0, 2.0], times=TimeDeltaTensor([0.0, 0.5], "ms"))

    assert series.unit == "ms"


@pytest.mark.parametrize("times", [
    TimeDeltaTensor([0.0, 1.0], "s"),
    TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "s"),
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
    series = TimeSeries(values, times=TimeDeltaTensor([0.0, 1.0], "s"))

    assert series.dtype is dtype


def test_series_of_points():
    values = [
        [1.0, -1.0],
        [2.0, -2.0],
        [3.0, -3.0],
    ]
    series = TimeSeries(values, times=TimeDeltaTensor([0.0, 1.0, 2.0], "s"))

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
    series = TimeSeries(values, times=TimeDeltaTensor([0.0, 1.0], "s"))

    assert series.shape == (2, 2)
    assert np.array_equal(np.asarray(series.values), values)



@pytest.mark.parametrize("make_values", TWO_ELEMENTS_OF_EVERY_DTYPE)
def test_series_of_every_element_type(make_values):
    values = make_values()
    series = TimeSeries(values, times=TimeDeltaTensor([0.0, 1.0], "s"))

    assert series.dtype is values.dtype
    assert len(series) == 2
    assert series.values.array_equal(values)


@pytest.mark.parametrize("values", [
    pytest.param(
        [sb.PointCloud(np.zeros((2, 2))), sb.PointCloud(np.ones((2, 2)))],
        id="list of point clouds",
    ),
    pytest.param(np.array(["a", "b"]), id="strings"),
    pytest.param(np.array([1, 2], dtype="timedelta64[s]"), id="durations"),
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
        TimeSeries(values, times=TimeDeltaTensor([0.0, 1.0], "s"))


def test_series_evaluates_to_previous_value():
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.5, 4.0], "s"))

    assert series(TimeDelta(0.0, "s")) == 10.0
    assert series(TimeDelta(1.0, "s")) == 10.0
    assert series(TimeDelta(1.5, "s")) == 20.0
    assert series(TimeDelta(3.75, "s")) == 20.0
    assert series(TimeDelta(4.0, "s")) == 30.0
    assert series(1, "s") == 10.0


@pytest.mark.parametrize("t", [TimeDelta(-0.5, "s"), TimeDelta(4.5, "s")])
def test_series_rejects_times_outside_observations(t):
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.5, 4.0], "s"))

    with pytest.raises(ValueError, match="outside the domain"):
        series(t)


def test_empty_series_cannot_be_evaluated():
    series = TimeSeries(sb.zeros((0,), dtype=sb.pcloud64), times=TimeDeltaTensor([], "s"))

    assert len(series) == 0
    assert series.dtype is sb.pcloud64
    with pytest.raises(ValueError, match="without observations"):
        series(TimeDelta(0.0, "s"))


def test_singleton_series_evaluates_at_its_time():
    series = TimeSeries([7.0], times=TimeDeltaTensor([2.0], "s"))

    assert series(TimeDelta(2.0, "s")) == 7.0


def _hourly_series():
    return TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.5, 4.0], "h"))


@pytest.mark.parametrize("duration", [
    pytest.param(TimeDelta(90, "min"), id="TimeDelta"),
    pytest.param(np.timedelta64(90, "m"), id="timedelta64"),
    pytest.param(datetime.timedelta(minutes=90), id="timedelta"),
])
def test_series_at_a_duration_equals_the_series_at_its_time_in_the_unit(duration):
    series = _hourly_series()

    assert series(duration) == series(TimeDelta(1.5, "h"))


@pytest.mark.parametrize("durations", [
    pytest.param(np.array([[30, 90],
                           [240, 60]], dtype="timedelta64[m]"), id="timedelta64"),
    pytest.param(TimeDeltaTensor(sb.FloatTensor([[30, 90],
                                           [240, 60]]), "min"), id="TimeDeltaTensor"),
])
def test_series_at_an_array_of_durations_equals_the_series_at_their_times(durations):
    series = _hourly_series()

    values = series(durations)

    assert values.shape == (2, 2)
    assert values.array_equal(series(TimeDeltaTensor([[0.5, 1.5], [4.0, 1.0]], "h")))


def test_series_at_a_pair_of_count_and_unit_equals_the_series_at_its_time_delta():
    series = _hourly_series()

    assert series((90, "min")) == series(TimeDelta(90, "min"))
    assert series(([30.0, 90.0], "min")).array_equal(series(TimeDeltaTensor([30.0, 90.0], "min")))


def test_windows_at_a_pair_of_counts_and_unit_equal_the_windows_at_their_time_deltas():
    windows = cut_windows(_hourly_series(), size=TimeDelta(2.0, "h"), step=TimeDelta(1.0, "h"), closed="left",
                          partial=False, anchor="start")

    assert windows(([0.0, 30.0], "min")).array_equal(windows(TimeDeltaTensor([0.0, 30.0], "min")))


def test_series_of_times_as_a_pair_equals_the_series_of_their_time_delta_tensor():
    series = TimeSeries([10.0, 20.0], times=([0.0, 1.5], "h"))

    assert series.array_equal(TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.5], "h")))


def test_series_at_a_tuple_of_two_dates_is_at_both_dates():
    series = _dated_series()

    assert series(("2023-01-01T00:30", "2023-01-01T02:00")).array_equal(
        series(DateTimeTensor(["2023-01-01T00:30", "2023-01-01T02:00"])))


def test_series_at_a_pair_with_a_unit_alias_equals_the_series_at_its_short_name():
    assert _hourly_series()((1.5, "hours")) == _hourly_series()((1.5, "h"))


def test_series_at_a_pair_with_an_unknown_unit_raises():
    with pytest.raises(ValueError, match="Unknown time unit"):
        _hourly_series()((90, "fortnight"))


def test_series_at_a_list_of_time_deltas_equals_the_series_at_their_times():
    series = _hourly_series()

    assert series([TimeDelta(30, "min"), TimeDelta(1.5, "h")]).array_equal(series(TimeDeltaTensor([0.5, 1.5], "h")))


def test_dated_series_at_a_duration_counts_from_its_epoch():
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.5, 4.0], "h"), epoch="2023-01-01")

    assert series(np.timedelta64(2, "h")) == series("2023-01-01T02:00")


def test_series_at_a_duration_takes_no_unit():
    with pytest.raises(TypeError, match="no unit"):
        _hourly_series()(np.timedelta64(1, "h"), "h")


def test_series_at_a_pair_of_durations_and_a_unit_raises():
    with pytest.raises(TypeError, match="own unit"):
        _hourly_series()((np.array([0, 1], dtype="timedelta64[h]"), "min"))


@pytest.mark.parametrize("durations", [
    pytest.param(np.array([1, "NaT"], dtype="timedelta64[h]"), id="NaT"),
    pytest.param(np.array([1], dtype="timedelta64[M]"), id="months"),
])
def test_series_rejects_durations_without_a_fixed_length(durations):
    with pytest.raises(ValueError, match="Durations"):
        _hourly_series()(durations)


def test_series_domain_runs_from_first_to_last_timestamp_closed():
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.5, 1.5, 4.0], "s"))

    assert series.domain == Domain(TimeDelta(0.5, "s"), TimeDelta(4.0, "s"), True, True)


def test_series_domain_is_in_the_query_unit():
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.5, 1.5, 4.0], "s")).to("ms")

    assert series.domain == Domain(TimeDelta(500.0, "ms"), TimeDelta(4000.0, "ms"), True, True)


def test_series_domain_fields_by_name_and_by_unpacking():
    domain = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.5, 4.0], "s")).domain

    start, end, start_closed, end_closed = domain

    assert (domain.start, domain.end) == (TimeDelta(0.5, "s"), TimeDelta(4.0, "s"))
    assert (domain.start_closed, domain.end_closed) == (True, True)
    assert (start, end, start_closed, end_closed) == (TimeDelta(0.5, "s"), TimeDelta(4.0, "s"), True, True)


def test_series_domain_of_several_channels_is_their_intersection():
    a = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.5], "s"))
    b = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([500.0, 3000.0], "ms"))

    series = TimeSeries.from_channels([a, b], unit="s")

    assert series.domain == Domain(TimeDelta(0.5, "s"), TimeDelta(2.5, "s"), True, True)


@pytest.mark.parametrize("closed, expected", [
    pytest.param("left", Domain(TimeDelta(0, "s"), TimeDelta(2, "s"), True, False), id="open-end"),
    pytest.param("right", Domain(TimeDelta(0, "s"), TimeDelta(2, "s"), False, True), id="open-start"),
])
def test_series_domain_of_several_channels_excludes_an_end_any_channel_excludes(closed, expected):
    source = TimeSeries([10.0, 20.0, 30.0, 40.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "s"))
    window = cut_windows(source, size=TimeDelta(2.0, "s"), step=TimeDelta(2.0, "s"), closed=closed, partial=False,
                         anchor="start")[0]
    observed = TimeSeries([-10.0, -20.0, -30.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "s"))

    series = TimeSeries.from_channels([window, observed])

    assert series.domain == expected


def test_dated_series_domain_has_date_time_ends():
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.5, 1.5, 4.0], "h"), epoch="2023-01-01")

    domain = series.domain

    assert domain.start.to_datetime64() == np.datetime64("2023-01-01T00:30")
    assert domain.end.to_datetime64() == np.datetime64("2023-01-01T04:00")
    assert (domain.start_closed, domain.end_closed) == (True, True)


def test_dated_series_domain_of_several_channels_is_the_intersection_of_where_each_lies():
    a = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.5], "h"), epoch="2023-01-01")
    b = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([30.0, 180.0], "min"), epoch="2023-01-01")
    series = TimeSeries.from_channels([a, b], unit="h")

    domain = series.domain

    assert domain.start.to_datetime64() == np.datetime64("2023-01-01T00:30")
    assert domain.end.to_datetime64() == np.datetime64("2023-01-01T02:30")


def test_series_epoch_is_the_epoch_of_its_times():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "h"), epoch="2023-01-01")

    assert series.epoch == DateTime("2023-01-01")
    assert series.to("min").epoch == DateTime("2023-01-01")


def test_series_from_datetimes_counts_their_resolution_from_the_first():
    stamps = np.array(["2023-01-01T06:00", "2023-01-01T06:01", "2023-01-01T07:00"], dtype="datetime64[m]")

    series = TimeSeries([10.0, 20.0, 30.0], times=stamps)

    assert series.unit == "min"
    assert series.epoch == DateTime(np.datetime64("2023-01-01T06:00", "m"))
    assert sb.FloatTensor([0.0, 1.0, 60.0], dtype=sb.float64).array_equal(series.times.count)


@pytest.mark.parametrize("dates", [
    pytest.param([datetime.datetime(2023, 1, 1, 6), datetime.datetime(2023, 1, 1, 7, 30)], id="datetimes"),
    pytest.param([datetime.date(2023, 1, 1), datetime.date(2023, 1, 3)], id="dates"),
    pytest.param([DateTime("2023-01-01T06:00"), DateTime("2023-01-01T07:30")], id="date_times"),
    pytest.param(["2023-01-01T06:00", "2023-01-01T07:30"], id="strings"),
    pytest.param(("2023-01-01T06:00", "2023-01-01T07:30"), id="tuple_of_strings"),
])
def test_series_from_a_list_of_dates_equals_the_series_from_their_tensor(dates):
    series = TimeSeries([10.0, 20.0], times=dates)

    assert series.array_equal(TimeSeries([10.0, 20.0], times=DateTimeTensor(dates)))


def test_series_from_a_list_of_dates_and_durations_raises():
    with pytest.raises(TypeError, match="mix"):
        TimeSeries([10.0, 20.0], times=["2023-01-01T06:00", TimeDelta(1, "h")])


def test_series_from_dates_with_a_unit_raises():
    with pytest.raises(TypeError, match="no unit"):
        TimeSeries([10.0, 20.0], times=(["2023-01-01T06:00", "2023-01-01T07:30"], "h"))


def _dated_series():
    return TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.5, 4.0], "h"), epoch="2023-01-01")


@pytest.mark.parametrize("date, t", [
    pytest.param(np.datetime64("2023-01-01T02:00"), TimeDelta(2.0, "h"), id="datetime64"),
    pytest.param(datetime.datetime(2023, 1, 1, 4), TimeDelta(4.0, "h"), id="datetime"),
    pytest.param(datetime.date(2023, 1, 1), TimeDelta(0.0, "h"), id="date"),
    pytest.param("2023-01-01T02:00", TimeDelta(2.0, "h"), id="str"),
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
    assert values.array_equal(series(TimeDeltaTensor([[0.5, 2.0], [4.0, 1.5]], "h")))


def test_series_at_ns_dates_over_292_years_after_its_epoch_equals_the_series_at_those_days():
    # Each observation is its year; 300 years in nanoseconds exceed int64.
    series = TimeSeries([1700.0, 1850.0, 2000.0],
                        times=np.array(["1700-01-01", "1850-01-01", "2000-01-01"], dtype="datetime64[D]"))
    days = np.array(["1850-01-01", "2000-01-01"], dtype="datetime64[D]")

    assert series(days.astype("datetime64[ns]")).array_equal(series(days))


def test_series_with_an_epoch_before_1678_at_ns_dates_equals_the_series_at_those_days():
    # Each observation is its year. 1500 is before the range of
    # datetime64[ns], and 2090 is 590 years after it.
    series = TimeSeries([1500.0, 1800.0, 2100.0],
                        times=np.array(["1500-01-01", "1800-01-01", "2100-01-01"], dtype="datetime64[D]"))
    days = np.array(["1800-01-01", "2090-01-01"], dtype="datetime64[D]")

    assert series(days.astype("datetime64[ns]")).array_equal(series(days))


# 00:30 and 02:00 on 2023-01-01, times 0.5 h and 2 h of _dated_series.
_DATES_IN_EVERY_FORM = [
    pytest.param(["2023-01-01T00:30", "2023-01-01T02:00"], id="strings"),
    pytest.param(("2023-01-01T00:30", "2023-01-01T02:00"), id="tuple_of_strings"),
    pytest.param([datetime.datetime(2023, 1, 1, 0, 30), datetime.datetime(2023, 1, 1, 2)], id="datetimes"),
    pytest.param([DateTime("2023-01-01T00:30"), DateTime("2023-01-01T02:00")], id="date_times"),
    pytest.param(np.array(["2023-01-01T00:30", "2023-01-01T02:00"], dtype="datetime64[15m]"), id="datetime64_15m"),
]


@pytest.mark.parametrize("dates", _DATES_IN_EVERY_FORM)
def test_series_at_dates_in_any_form_equals_the_series_at_their_times(dates):
    series = _dated_series()

    assert series(dates).array_equal(series(TimeDeltaTensor([0.5, 2.0], "h")))


@pytest.mark.parametrize("dates", _DATES_IN_EVERY_FORM)
def test_tensor_at_dates_in_any_form_equals_the_tensor_at_their_times(dates):
    tensor = sb.tensor([_dated_series(), _dated_series()])

    assert tensor(dates).array_equal(tensor(TimeDeltaTensor([0.5, 2.0], "h")))


def test_series_at_a_date_time_equals_the_series_at_its_datetime64():
    series = _dated_series()

    assert series(DateTime("2023-01-01T02:00")) == series(np.datetime64("2023-01-01T02:00"))


def test_series_at_a_date_time_tensor_equals_the_series_at_its_datetime64_array():
    series = _dated_series()
    dates = DateTimeTensor([["2023-01-01T00:30", "2023-01-01T02:00"],
                            ["2023-01-01T04:00", "2023-01-01T01:30"]])

    values = series(dates)

    assert values.shape == (2, 2)
    assert values.array_equal(series(dates.to_datetime64()))


def test_series_of_several_channels_at_a_date_gives_each_channel():
    a = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "h"), epoch="2023-01-01")
    b = TimeSeries([-10.0, -20.0, -30.0], times=TimeDeltaTensor([0.0, 30.0, 60.0], "min"), epoch="2023-01-01")
    series = TimeSeries.from_channels([a, b], unit="h")

    values = series(np.datetime64("2023-01-01T00:45"))

    assert (float(values[0]), float(values[1])) == (10.0, -20.0)


def test_series_of_several_channels_at_an_array_of_dates_gives_each_channel():
    a = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "h"), epoch="2023-01-01")
    b = TimeSeries([-10.0, -20.0, -30.0], times=TimeDeltaTensor([0.0, 30.0, 60.0], "min"), epoch="2023-01-01")
    series = TimeSeries.from_channels([a, b], unit="h")

    values = series(np.array(["2023-01-01T00:15", "2023-01-01T00:45"], dtype="datetime64[m]"))

    assert values[0].array_equal(sb.FloatTensor([10.0, 10.0], dtype=sb.float64))
    assert values[1].array_equal(sb.FloatTensor([-10.0, -20.0], dtype=sb.float64))


def test_times_of_a_dated_series_are_durations_from_its_epoch():
    series = _dated_series()

    assert series.times.array_equal(TimeDeltaTensor([0.0, 1.5, 4.0], "h"))


def test_datetimes_of_a_dated_series_are_its_epoch_plus_its_times():
    series = _dated_series()

    assert series.datetimes.array_equal(
        DateTimeTensor(np.array(["2023-01-01T00:00", "2023-01-01T01:30", "2023-01-01T04:00"], dtype="datetime64[us]")))


def test_datetimes_of_a_series_without_an_epoch_raise():
    with pytest.raises(ValueError, match="with_epoch"):
        TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "h")).datetimes


def test_datetimes_of_several_channels_raise_naming_the_datetimes_of_each_channel():
    a = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "h"), epoch="2023-01-01")
    b = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([0.0, 60.0], "min"), epoch="2023-01-01")

    with pytest.raises(ValueError, match=re.escape("series[i].datetimes")):
        TimeSeries.from_channels([a, b], unit="h").datetimes


def test_series_without_an_epoch_cannot_be_evaluated_at_a_date():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "s"))

    with pytest.raises(ValueError, match="with_epoch"):
        series(np.datetime64("2023-01-01T00:30"))


def test_series_at_a_date_takes_no_unit():
    with pytest.raises(TypeError):
        _dated_series()(np.datetime64("2023-01-01T00:30"), "h")


def test_series_rejects_a_mix_of_dates_and_numbers():
    with pytest.raises(TypeError, match="mix"):
        _dated_series()([np.datetime64("2023-01-01T00:30"), 0.5])


def test_series_at_a_date_with_a_timezone_is_at_its_utc_instant():
    plus_two = datetime.timezone(datetime.timedelta(hours=2))

    # 02:00 at +02:00 is 00:00 UTC, time 0 of the series
    assert _dated_series()(datetime.datetime(2023, 1, 1, 2, tzinfo=plus_two)) == 10.0


def test_series_at_a_date_outside_its_domain_raises_with_the_date_and_the_domain_as_dates():
    dates = np.array(["2023-01-01T01:00", "2023-01-01T05:00"], dtype="datetime64[m]")

    with pytest.raises(TimeSeriesDomainError) as excinfo:
        _dated_series()(dates)

    assert excinfo.value.time == DateTime("2023-01-01T05:00")
    assert excinfo.value.domain == Domain(DateTime("2023-01-01T00:00"), DateTime("2023-01-01T04:00"), True, True)
    assert excinfo.value.index is None


def test_series_at_a_duration_outside_its_domain_raises_with_the_time_and_the_domain_as_durations():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "s"))

    with pytest.raises(TimeSeriesDomainError) as excinfo:
        series(TimeDelta(3.0, "s"))

    assert excinfo.value.time == TimeDelta(3.0, "s")
    assert excinfo.value.domain == Domain(TimeDelta(0.0, "s"), TimeDelta(1.0, "s"), True, True)


def test_time_series_domain_error_survives_pickling():
    error = TimeSeriesDomainError("Time 3 is outside the domain [0, 1]", 3.0, Domain(0.0, 1.0, True, True), (1,))

    copy = pickle.loads(pickle.dumps(error))

    assert (str(copy), copy.time, copy.domain, copy.index) == (str(error), 3.0, Domain(0.0, 1.0, True, True), (1,))


def _nanosecond_series():
    times = np.array(["2023-01-01T00:00", "2023-01-01T01:00", "2023-01-01T02:00"], dtype="datetime64[ns]")
    return TimeSeries([10.0, 20.0, 30.0], times=DateTimeTensor(times))


# A nanosecond after 05:00, which no precision down to microseconds shows.
_NANOSECOND_DATE = np.datetime64("2023-01-01T05:00:00.000000001", "ns")


def test_series_at_a_date_outside_its_domain_in_nanoseconds_names_the_date():
    with pytest.raises(TimeSeriesDomainError) as excinfo:
        _nanosecond_series()(_NANOSECOND_DATE)
    message = str(excinfo.value)

    assert "2023-01-01T05:00:00.000000001" in message


def test_series_at_a_date_just_after_an_end_between_microseconds_names_the_end():
    # The domain ends 700 ns into the first microsecond.
    times = np.array(["2023-01-01T00:00:00.000000000", "2023-01-01T00:00:00.000000700"], dtype="datetime64[ns]")
    series = TimeSeries([10.0, 20.0], times=DateTimeTensor(times))

    with pytest.raises(TimeSeriesDomainError) as excinfo:
        series(np.datetime64("2023-01-01T00:00:00.000000701", "ns"))

    assert excinfo.value.domain.end == DateTime("2023-01-01T00:00:00.000000700")  # not rounded to a microsecond


def test_point_clouds_at_a_date_outside_the_domain_in_nanoseconds_raise_as_the_series_does():
    with pytest.raises(TimeSeriesDomainError) as points_excinfo:
        to_point_clouds(_nanosecond_series(), np.array([_NANOSECOND_DATE]))
    with pytest.raises(TimeSeriesDomainError) as series_excinfo:
        _nanosecond_series()(_NANOSECOND_DATE)

    assert points_excinfo.value.time == series_excinfo.value.time
    assert points_excinfo.value.domain == series_excinfo.value.domain


def test_mapping_a_series_over_a_date_outside_its_domain_raises():
    # A StopIteration from the call would end map() early instead.
    with pytest.raises(TimeSeriesDomainError):
        list(map(_nanosecond_series(), [_NANOSECOND_DATE]))


def test_window_at_a_date_outside_it_raises_with_its_domain_as_dates():
    window = cut_windows(_dated_series(), size=TimeDelta(2.0, "h"), step=TimeDelta(1.0, "h"), closed="left",
                         partial=False, anchor="start")[1]

    with pytest.raises(TimeSeriesDomainError) as excinfo:
        window("2023-01-01T03:00")

    assert excinfo.value.domain == Domain(DateTime("2023-01-01T01:00"), DateTime("2023-01-01T03:00"), True, False)


def test_a_date_just_below_a_timestamp_snaps_to_it():
    # 0.1 + 0.2 computes as 0.30000000000000004, so 300 ms after the epoch
    # converts to just below it.
    series = TimeSeries([1.0, 2.0, 3.0], times=TimeDeltaTensor([0.0, 0.1, 0.1 + 0.2], "s"), epoch="2023-01-01")

    assert series(np.datetime64("2023-01-01T00:00:00.300")) == 3.0


def test_window_at_a_date_equals_its_series_at_that_date():
    series = _dated_series()
    # The window covers 01:00 to 03:00; the value changes from 10 to 20 at
    # 01:30.
    window = cut_windows(series, size=TimeDelta(2.0, "h"), step=TimeDelta(1.0, "h"), closed="left", partial=False,
                         anchor="start")[1]

    assert window(np.datetime64("2023-01-01T01:15")) == 10.0
    assert window(np.datetime64("2023-01-01T02:30")) == 20.0


def _window_from_00_18():
    """The window from 00:18 to 01:18 of a series observed every 6 minutes
    and stored in hours, holding 10 * t at t hours. Its origin, three steps
    of 0.1 h, is 0.30000000000000004, while 00:18 counts as 0.3 h from the
    epoch, so its start date lands just before its start time."""
    times = np.arange("2023-01-01T00:00", "2023-01-01T02:06", np.timedelta64(6, "m"), dtype="datetime64[m]")
    # 6 minutes is 0.1 h, which is inexact, so the window's origin rounds.
    series = TimeSeries(np.arange(21.0), times=DateTimeTensor(times)).to("h")
    return cut_windows(series, size=TimeDelta(1.0, "h"), step=TimeDelta(6.0, "min"), closed="left", partial=False,
                       anchor="start")[3]


def test_window_at_its_start_date_is_its_first_value():
    window = _window_from_00_18()

    assert window(window.domain.start) == 3.0


def test_tensor_of_a_window_at_its_start_date_is_its_first_value():
    window = _window_from_00_18()

    values = sb.tensor([window])(window.domain.start)

    assert values[0] == 3.0


def test_delay_embedding_of_a_window_at_its_start_date_is_its_first_point():
    # The embedding starts a step after the window, at 00:24.
    embedding = delay_embed(_window_from_00_18(), dimension=2, delay=TimeDelta(6.0, "min"))

    point = embedding(embedding.domain.start)

    assert np.array_equal(np.asarray(point), [3.0, 4.0])  # the window at 00:18 and 00:24


def test_channel_at_the_date_of_its_closed_end_after_from_channels_shifts_it_is_its_last_value():
    early = TimeSeries([0.0, 60.0], times=DateTimeTensor(["2023-01-01T00:00", "2023-01-01T00:06"])).to("h")
    late = TimeSeries([-10.0, -30.0, -60.0],
                      times=DateTimeTensor(["2023-01-01T00:01", "2023-01-01T00:03", "2023-01-01T00:06"])).to("h")
    # Shifted by 1/60 h, the end of late, 5/60 h, lands at
    # 0.09999999999999999, while 00:06 counts as 0.1 h from the epoch.
    series = TimeSeries.from_channels([early, late])

    assert series[1](np.datetime64("2023-01-01T00:06")) == -60.0


def test_series_without_an_epoch_has_none():
    assert TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "s")).epoch is None


def test_series_epoch_is_one_value_for_several_channels():
    a = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "h"), epoch="2023-01-01")
    b = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([0.0, 60.0], "min"), epoch="2023-01-01")

    assert TimeSeries.from_channels([a, b], unit="h").epoch == DateTime("2023-01-01")


def test_series_with_epoch_sets_the_epoch_of_every_channel():
    a = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "h"))
    b = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([0.0, 60.0], "min"))

    series = TimeSeries.from_channels([a, b], unit="h").with_epoch("2023-01-01")

    assert series[0].epoch == DateTime("2023-01-01")
    assert series[1].epoch == DateTime("2023-01-01")


def test_series_with_epoch_keeps_its_times_and_values():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.5], "s"))

    dated = series.with_epoch("2023-01-01")

    assert dated.array_equal(series)
    assert dated(TimeDelta(1.5, "s")) == 20.0


def test_series_with_epoch_none_removes_the_epoch():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.5], "s"), epoch="2023-01-01")

    assert series.with_epoch(None).epoch is None


def test_series_channels_with_and_without_an_epoch_raise():
    a = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "h"), epoch="2023-01-01")
    b = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([0.0, 1.0], "h"))

    with pytest.raises(ValueError, match="different epochs"):
        TimeSeries.from_channels([a, b])


def test_series_channels_of_different_epochs_count_from_the_earliest():
    a = TimeSeries([10.0, 20.0, 30.0], times=DateTimeTensor(["2023-01-01T06:00", "2023-01-01T07:00",
                                                             "2023-01-01T08:00"]))
    b = TimeSeries([-10.0, -20.0], times=DateTimeTensor(["2023-01-01T06:30", "2023-01-01T07:30"]))

    series = TimeSeries.from_channels([a, b])

    assert series.epoch == DateTime("2023-01-01T06:00")
    assert series[1](TimeDelta(30.0, "min")) == -10.0
    assert series[1](TimeDelta(90.0, "min")) == -20.0
    values = series("2023-01-01T07:15")
    assert (float(values[0]), float(values[1])) == (20.0, -10.0)


def test_series_channels_from_1659_in_seconds_and_in_ns_count_from_1659():
    # 1659 is before the range of datetime64[ns].
    a = TimeSeries([10.0, 20.0],
                   times=np.array(["1659-01-01T00:00:00", "1659-01-01T00:00:10"], dtype="datetime64[s]"))
    b = TimeSeries([-10.0, -20.0],
                   times=np.array(["1990-01-01T00:00:00", "1990-01-01T00:00:10"], dtype="datetime64[ns]"))

    series = TimeSeries.from_channels([a, b], unit="s")

    assert series.epoch == DateTime("1659-01-01T00:00:00")
    assert series[1](np.datetime64("1990-01-01T00:00:10", "s")) == -20.0


def test_series_channels_at_epochs_of_different_resolutions_combine():
    # The same instant, counted in days and in microseconds.
    a = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "h"), epoch="2023-01-01")
    b = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([0.0, 1.0], "h"), epoch=datetime.datetime(2023, 1, 1))

    series = TimeSeries.from_channels([a, b])

    assert series.epoch == DateTime("2023-01-01")
    assert series[1](TimeDelta(1.0, "h")) == -20.0


def test_series_embedding_channel_of_a_later_epoch_reads_the_same_points():
    # Each observation is 10 times its hour on 2 January.
    source = TimeSeries([0.0, 10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "h"), epoch="2023-01-02")
    embedding = delay_embed(source, dimension=2, delay=TimeDelta(1.0, "h"))
    earlier = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([0.0, 48.0], "h"), epoch="2023-01-01")

    series = TimeSeries.from_channels([embedding, earlier])

    # (x(01:00), x(02:00)) on 2 January
    assert np.array_equal(np.asarray(series("2023-01-02T02:00")[0]), [10.0, 20.0])


def test_series_window_channel_of_a_later_epoch_keeps_where_it_lies():
    # Each observation is 10 times its hour on 2 January.
    source = TimeSeries([0.0, 10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "h"),
                        epoch="2023-01-02")
    # From 01:00 to 03:00 on 2 January
    window = cut_windows(source, size=TimeDelta(2.0, "h"), step=TimeDelta(1.0, "h"), closed="left", partial=False,
                         anchor="start")[1]
    earlier = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([0.0, 48.0], "h"), epoch="2023-01-01")

    series = TimeSeries.from_channels([window, earlier])

    assert series.epoch == DateTime("2023-01-01")
    assert series[0].domain == window.domain
    # x(02:00) on 2 January
    assert float(series("2023-01-02T02:30")[0]) == 20.0


def test_series_origin_is_zero_for_recorded_observations():
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.5, 1.5, 4.0], "s"))

    assert series.origin == TimeDelta(0, "s")


def test_series_origin_has_one_entry_per_channel():
    a = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "s"))
    b = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([0.0, 1000.0], "ms"))

    origin = TimeSeries.from_channels([a, b], unit="s").origin

    assert origin.array_equal(TimeDeltaTensor([0.0, 0.0], "s"))


def test_series_origin_cannot_be_assigned_to():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "s"))

    with pytest.raises(AttributeError):
        series.origin = TimeDelta(1.0, "s")  # pyright: ignore[reportAttributeAccessIssue]


def test_query_just_below_an_observation_snaps_to_it():
    series = TimeSeries([1.0, 2.0, 3.0], times=TimeDeltaTensor([0.0, 0.25, 0.5], "s"), snap_tolerance=1e-14)

    assert series(TimeDelta(np.nextafter(0.25, 0.0), "s")) == 2.0


def test_query_further_below_an_observation_does_not_snap():
    series = TimeSeries([1.0, 2.0, 3.0], times=TimeDeltaTensor([0.0, 0.25, 0.5], "s"), snap_tolerance=1e-14)

    assert series(TimeDelta(0.24, "s")) == 1.0


@pytest.mark.parametrize("times, t, expected", [
    # Sampled every 0.1 ns, in seconds.
    (TimeDeltaTensor([0.0, 1e-10, 2e-10], "s"), TimeDelta(1.5e-10, "s"), 20.0),
    # Microsecond spacing on epoch-sized timestamps.
    (TimeDeltaTensor([1.7e9, 1.7e9 + 1e-6, 1.7e9 + 2e-6], "s"), TimeDelta(1.7e9 + 1.5e-6, "s"), 20.0),
])
def test_snapping_does_not_cross_a_gap(times, t, expected):
    series = TimeSeries([10.0, 20.0, 30.0], times=times, snap_tolerance=1e-14)

    assert series(t) == expected


def test_a_query_one_float_spacing_below_an_epoch_timestamp_snaps_to_it():
    times = TimeDeltaTensor([1.7e9, 1.7e9 + 1e-6, 1.7e9 + 2e-6], "s")
    series = TimeSeries([10.0, 20.0, 30.0], times=times, snap_tolerance=1e-14)

    assert series(TimeDelta(np.nextafter(times[1].count, 0.0), "s")) == 20.0


def test_the_snap_distance_is_relative_to_the_magnitude_of_the_timestamps():
    # With a tolerance of 2**-20, the snap distance is 2**-10 below 1024 but
    # only 2**-20 below 1.
    large = TimeSeries([1.0, 2.0], times=TimeDeltaTensor([0.0, 1024.0], "s"), snap_tolerance=2.0**-20)
    small = TimeSeries([1.0, 2.0], times=TimeDeltaTensor([0.0, 1.0], "s"), snap_tolerance=2.0**-20)

    assert large(TimeDelta(1024.0 - 2.0**-11, "s")) == 2.0
    assert large(TimeDelta(1024.0 - 2.0**-9, "s")) == 1.0
    assert small(TimeDelta(1.0 - 2.0**-11, "s")) == 1.0


def test_snapping_reaches_at_most_a_quarter_of_the_gap():
    # The tolerance allows 2**-10 below 1024, but the gap is 2**-9, so a
    # quarter of it, 2**-11, is the snap distance.
    series = TimeSeries([1.0, 2.0], times=TimeDeltaTensor([1024.0 - 2.0**-9, 1024.0], "s"),
                        snap_tolerance=2.0**-20)

    assert series(TimeDelta(1024.0 - 2.0**-12, "s")) == 2.0
    assert series(TimeDelta(1024.0 - 3 * 2.0**-12, "s")) == 1.0


def test_snap_tolerance_zero_disables_snapping():
    series = TimeSeries([1.0, 2.0, 3.0], times=TimeDeltaTensor([0.0, 0.25, 0.5], "s"), snap_tolerance=0.0)

    assert series.snap_tolerance == 0.0
    assert series(TimeDelta(np.nextafter(0.25, 0.0), "s")) == 1.0


@pytest.mark.parametrize("tolerance", [-0.1, 1.0, np.nan])
def test_snap_tolerance_outside_unit_interval_raises(tolerance):
    with pytest.raises(ValueError):
        TimeSeries([1.0, 2.0], times=TimeDeltaTensor([0.0, 1.0], "s"), snap_tolerance=tolerance)


def test_to_converts_the_times():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 0.5], "s")).to("ms")

    assert series.unit == "ms"
    assert series.times.array_equal(TimeDeltaTensor([0.0, 500.0], "ms"))
    assert series(TimeDelta(250.0, "ms")) == 10.0
    assert series(TimeDelta(500.0, "ms")) == 20.0


def test_to_converts_the_origin_and_the_end_after_the_last_observation():
    # A window starting at 1 s, with observations at 0 s and 1 s and an end at 2 s.
    series = TimeSeries([0.0, 10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "s"))
    window = cut_windows(series, size=TimeDelta(2, "s"), step=TimeDelta(1, "s"), closed="left", partial=False,
                         anchor="start")[1]

    converted = window.to("ms")

    assert converted.origin == TimeDelta(1000.0, "ms")
    assert converted.domain == Domain(TimeDelta(0.0, "ms"), TimeDelta(2000.0, "ms"), True, False)


def test_to_keeps_how_an_embedding_evaluates():
    series = TimeSeries([0.0, 10.0, 25.0, 40.0], times=TimeDeltaTensor([0.0, 1.0, 2.5, 4.0], "s"))
    embedding = delay_embed(series, dimension=3, delay=TimeDelta(1, "s"))
    queries = TimeDeltaTensor([2.0, 2.5, 3.0, 4.0], "s")

    assert embedding.to("ms")(queries).array_equal(embedding(queries))


def test_from_channels_stores_every_channel_in_its_unit():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="ms")

    assert series[0].times.array_equal(TimeDeltaTensor([0.0, 1000.0, 2500.0], "ms"))
    assert series[1].times.array_equal(TimeDeltaTensor([500.0, 1500.0, 3000.0], "ms"))


def test_evaluate_reads_times_in_the_given_unit():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 500.0], "ms"))

    assert series(250000.0, "us") == 10.0
    assert series(500000.0, "us") == 20.0
    assert series.unit == "ms"


def test_evaluate_rejects_an_unknown_unit():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 0.5], "s"))

    with pytest.raises(ValueError, match="Unknown time unit"):
        series(0.25, "fortnight")


def test_series_at_numbers_of_a_unit_equals_the_series_at_their_durations():
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.5, 4.0], "s"))

    assert series(1500.0, "ms") == series(TimeDelta(1500.0, "ms"))
    assert series([0.0, 1500.0, 3000.0], "ms").array_equal(series(TimeDeltaTensor([0.0, 1500.0, 3000.0], "ms")))


@pytest.mark.parametrize("t", [pytest.param(1.5, id="single"), pytest.param([0.0, 1.5], id="many")])
def test_series_with_a_unit_rejects_numbers_without_unit(t):
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.5, 4.0], "s"))

    with pytest.raises(TypeError, match=re.escape("series(t, 's')")):
        series(t)


def test_series_without_a_unit_takes_numbers():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    assert series(1.5) == 20.0
    assert series([0.0, 1.5, 4.0]).array_equal(sb.FloatTensor([10.0, 20.0, 30.0], dtype=sb.float64))


@pytest.mark.parametrize("t, unit", [
    pytest.param(TimeDelta(1.5, "s"), None, id="duration"),
    pytest.param(TimeDeltaTensor([0.0, 1.5], "s"), None, id="durations"),
    pytest.param("2023-01-01", None, id="date"),
    pytest.param(1.5, "s", id="unit"),
])
def test_series_without_a_unit_rejects_durations_dates_and_units(t, unit):
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    with pytest.raises(TypeError, match="without a unit"):
        series(t, unit)


def _channel_a():
    return TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.5], "s"))


def _channel_b():
    return TimeSeries([-1.0, -2.0, -3.0], times=TimeDeltaTensor([500.0, 1500.0, 3000.0], "ms"))


def test_from_channels_keeps_every_channel():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    assert series.n_channels == 2
    assert series.unit == "s"


def _point_cloud_channel():
    clouds = sb.zeros((2,), dtype=sb.pcloud64)
    return TimeSeries(clouds, times=TimeDeltaTensor([0.0, 1.0], "s"))


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
    TimeDelta(0.25, "s"),  # only channel A covers it
    TimeDelta(2.75, "s"),  # only channel B covers it
])
def test_several_channels_evaluate_only_where_all_are_observed(t):
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    # Both ends of the shared domain, 0.5 (B's first) and 2.5 (A's last), work.
    assert series(TimeDelta(0.5, "s"))[1][()] == -1.0
    assert series(TimeDelta(2.5, "s"))[0][()] == 30.0
    with pytest.raises(ValueError, match="outside the domain"):
        series(t)


def test_times_of_several_channels_are_in_the_query_unit():
    # Channel A is in s and channel B in ms.
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    assert series[0].times.array_equal(TimeDeltaTensor([0.0, 1.0, 2.5], "s"))
    assert series[1].times.array_equal(TimeDeltaTensor([0.5, 1.5, 3.0], "s"))


def test_times_of_several_channels_raise_naming_the_times_of_each_channel():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    with pytest.raises(ValueError, match=re.escape("series[i].times")):
        series.times


@pytest.mark.parametrize("t, expected", [
    (TimeDelta(1.25, "s"), [20.0, -1.0]),  # between B's observations at 500 and 1500 ms
    (TimeDelta(1.5, "s"), [20.0, -2.0]),  # at B's observation at 1500 ms
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
    a = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.5], "s"),
                   snap_tolerance=0.0)
    series = TimeSeries.from_channels([a, _channel_b()], unit="s")

    # Just below channel A's 1.0 s and channel B's 1500 ms.
    assert series(TimeDelta(np.nextafter(1.0, 0.0), "s"))[0][()] == 10.0
    assert series(TimeDelta(np.nextafter(1.5, 0.0), "s"))[1][()] == -2.0


def test_several_point_cloud_channels_evaluate_to_their_clouds():
    first = sb.PointCloudTensor([
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ], dtype=sb.pcloud64)
    second = sb.PointCloudTensor([
        np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]]),
    ], dtype=sb.pcloud64)
    series = TimeSeries.from_channels([
        TimeSeries(first, times=TimeDeltaTensor([0.0, 1.0], "s")),
        TimeSeries(second, times=TimeDeltaTensor([0.0], "s")),
    ])

    values = series(TimeDelta(0.0, "s"))

    assert values.dtype is sb.pcloud64
    assert np.array_equal(np.asarray(values[0][()]), [[0.0, 0.0]])
    assert np.array_equal(np.asarray(values[1][()]), [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])


def test_many_times_keep_the_caller_order_and_shape():
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.5, 4.0], "s"))

    assert np.array_equal(np.asarray(series(TimeDeltaTensor([4.0, 0.0, 1.0, 1.5], "s"))), [30.0, 10.0, 10.0, 20.0])
    assert np.array_equal(np.asarray(series(TimeDeltaTensor([[0.0, 4.0], [1.5, 1.0]], "s"))),
                          [[10.0, 30.0], [20.0, 10.0]])


def test_many_times_on_several_channels():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    values = series(TimeDeltaTensor([1.25, 1.5], "s"))

    assert isinstance(values, sb.NestedTensor)
    assert np.array_equal(np.asarray(values[0]), [20.0, 20.0])
    assert np.array_equal(np.asarray(values[1]), [-1.0, -2.0])


def test_many_times_snap_and_check_the_domain_like_single_times():
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.5, 4.0], "s"))
    queries = TimeDeltaTensor([0.0, np.nextafter(1.5, 0.0)], "s")

    assert np.array_equal(np.asarray(series(queries)), [series(t) for t in queries])
    with pytest.raises(ValueError, match="outside the domain"):
        series(TimeDeltaTensor([0.0, 4.5], "s"))
    with pytest.raises(ValueError, match="outside the domain"):
        TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])([0.0, np.nan, 4.0])


@pytest.mark.parametrize("index", [1, -1])
def test_selecting_one_channel_keeps_the_query_unit(index):
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    selected = series[index]

    assert selected.n_channels == 1
    assert selected.unit == "s"
    assert selected(TimeDelta(1.5, "s")) == -2.0


def test_selecting_several_channels_keeps_their_order():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    values = series[[1, 0]](TimeDelta(1.5, "s"))

    assert [values[0][()], values[1][()]] == [-2.0, 20.0]


def test_selection_keeps_each_channel_snap_settings():
    a = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.5], "s"),
                   snap_tolerance=0.25)
    series = TimeSeries.from_channels([a, _channel_b()], unit="s")

    assert series[0].snap_tolerance == 0.25
    assert series[1].snap_tolerance != series[0].snap_tolerance
    assert np.array_equal(np.asarray(series.snap_tolerance),
                          [series[0].snap_tolerance, series[1].snap_tolerance])


def test_values_are_a_copy():
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "s"))

    values = series.values
    values[1] = 0.0

    assert values[1] == 0.0
    assert series(TimeDelta(1.0, "s")) == 20.0


@pytest.mark.parametrize("make_values", [
    p for p in TWO_ELEMENTS_OF_EVERY_DTYPE
    if p.id in ("pcloud64", "distmat64", "symmat64")
])
@pytest.mark.parametrize("read", [
    pytest.param(lambda values, series, embedding: values[0], id="input"),
    pytest.param(lambda values, series, embedding: series.values[0], id="values"),
    pytest.param(lambda values, series, embedding: series(TimeDelta(0.0, "s")), id="evaluation"),
    pytest.param(lambda values, series, embedding: embedding.values[0, 0], id="embedding values"),
])
def test_mutating_an_observation_leaves_the_series_unchanged(make_values, read):
    values = make_values()
    expected = np.asarray(values[0]).copy()
    series = TimeSeries(values, times=TimeDeltaTensor([0.0, 1.0], "s"))
    embedding = delay_embed(series, dimension=2, delay=TimeDelta(1.0, "s"))

    observation = read(values, series, embedding)
    observation[0, 1] = 99.0

    assert observation[0, 1] == 99.0
    assert np.array_equal(np.asarray(series(TimeDelta(0.0, "s"))), expected)
    assert np.array_equal(np.asarray(embedding.values[0, 0]), expected)


def test_values_of_several_channels_have_one_leaf_per_channel():
    series = TimeSeries.from_channels([_channel_a(), _channel_b()], unit="s")

    values = series.values

    assert isinstance(values, sb.NestedTensor)
    assert np.array_equal(np.asarray(values[0]), [10.0, 20.0, 30.0])
    assert np.array_equal(np.asarray(values[1]), [-1.0, -2.0, -3.0])


def test_shapes_and_counts_of_several_channels():
    points = TimeSeries([[1.0, -1.0], [2.0, -2.0]], times=TimeDeltaTensor([0.0, 1.0], "s"))
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
    times = TimeDeltaTensor(0.25 * np.arange(values.shape[0]), "ms")
    series = TimeSeries(values, times=times, snap_tolerance=0.25)

    loaded = round_trip(series)

    assert isinstance(loaded, TimeSeries)
    assert loaded.dtype is series.dtype
    assert loaded.shape == series.shape
    assert loaded.times.array_equal(series.times)
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
        assert loaded[i].times.array_equal(series[i].times)
    for t in (TimeDelta(0.5, "s"), TimeDelta(1.5, "s")):
        assert [loaded(t)[i][()] for i in range(2)] == [series(t)[i][()] for i in range(2)]


def test_ts_dtype_is_one_object_per_value_type():
    assert sb.ts(sb.float64) is sb.ts(sb.float64)
    assert sb.ts(sb.float64) is not sb.ts(sb.float32)
    assert sb.ts(sb.float64).value_type is sb.float64


@pytest.mark.parametrize("value_type", [sb.float32, sb.float64, sb.pcf32, sb.pcf64])
def test_zeros_of_a_series_dtype_is_a_series_tensor_of_that_shape_and_dtype(value_type):
    tensor = sb.zeros((2, 3), dtype=sb.ts(value_type))

    assert isinstance(tensor, TimeSeriesTensor)
    assert tensor.shape == (2, 3)
    assert tensor.dtype is sb.ts(value_type)


def test_zeros_of_series_takes_an_assigned_series():
    tensor = sb.zeros((2,), dtype=sb.ts(sb.float64))
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "s"))

    tensor[1] = series

    assert tensor[1].array_equal(series)


def test_zeros_of_series_raises_on_evaluating_an_element_left_empty():
    tensor = sb.zeros((2,), dtype=sb.ts(sb.float64))

    with pytest.raises(ValueError):
        tensor[0](TimeDelta(0.0, "s"))


def test_tensor_of_series_of_different_lengths_and_units_keeps_each():
    seconds = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "s"))
    minutes = TimeSeries([40.0, 50.0], times=TimeDeltaTensor([0.0, 3.0], "min"))

    tensor = sb.tensor([seconds, minutes])

    assert isinstance(tensor, TimeSeriesTensor)
    assert tensor.shape == (2,)
    assert tensor.dtype is sb.ts(sb.float64)
    assert tensor[0].array_equal(seconds)
    assert tensor[1].array_equal(minutes)


def test_tensor_of_nested_lists_of_series_is_two_dimensional():
    a = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "s"))
    b = TimeSeries([30.0], times=TimeDeltaTensor([0.0], "s"))
    c = TimeSeries([40.0, 50.0, 60.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "s"))

    tensor = sb.tensor([[a, b, c],
                        [c, b, a]])

    assert tensor.shape == (2, 3)
    assert tensor[0, 0].array_equal(a)
    assert tensor[0, 1].array_equal(b)
    assert tensor[0, 2].array_equal(c)
    assert tensor[1, 0].array_equal(c)
    assert tensor[1, 1].array_equal(b)
    assert tensor[1, 2].array_equal(a)


def test_tensor_of_series_takes_a_matching_dtype():
    series = TimeSeries(np.array([10.0, 20.0], dtype=np.float32), times=TimeDeltaTensor([0.0, 1.0], "s"))

    tensor = sb.tensor([series], dtype=sb.ts(sb.float32))

    assert tensor.dtype is sb.ts(sb.float32)
    assert tensor[0].array_equal(series)


def test_tensor_of_series_of_different_element_types_raises():
    doubles = TimeSeries(np.array([10.0, 20.0], dtype=np.float64), times=TimeDeltaTensor([0.0, 1.0], "s"))
    floats = TimeSeries(np.array([10.0, 20.0], dtype=np.float32), times=TimeDeltaTensor([0.0, 1.0], "s"))

    with pytest.raises(TypeError):
        sb.tensor([doubles, floats])


def test_tensor_of_series_with_a_dtype_that_does_not_match_raises():
    doubles = TimeSeries(np.array([10.0, 20.0], dtype=np.float64), times=TimeDeltaTensor([0.0, 1.0], "s"))

    with pytest.raises(TypeError):
        sb.tensor([doubles], dtype=sb.ts(sb.float32))


def _hourly(values, epoch):
    """``values`` at 0, 1, and 2 h after ``epoch``."""
    return TimeSeries(values, times=TimeDeltaTensor([0.0, 1.0, 2.0], "h")).with_epoch(epoch)


def test_tensor_of_series_with_different_epochs_evaluates_dates_at_the_same_instants():
    early = _hourly([10.0, 20.0, 30.0], "2023-01-01T00:00")
    late = _hourly([40.0, 50.0, 60.0], "2023-01-01T01:00")
    dates = DateTimeTensor(["2023-01-01T01:30", "2023-01-01T02:00"])

    values = sb.tensor([early, late])(dates)

    assert np.array_equal(np.asarray(values[0]), np.asarray(early(dates)))
    assert np.array_equal(np.asarray(values[1]), np.asarray(late(dates)))


def test_tensor_of_a_series_from_1659_at_ns_dates_equals_the_series_at_those_dates():
    # Each observation is its year; 1659 is before the range of datetime64[ns].
    series = TimeSeries([1659.0, 1800.0, 2000.0],
                        times=np.array(["1659-01-01", "1800-01-01", "2000-01-01"], dtype="datetime64[D]"))
    dates = np.array(["1800-01-01", "2000-01-01"], dtype="datetime64[ns]")

    values = sb.tensor([series])(dates)

    assert np.array_equal(np.asarray(values[0]), np.asarray(series(dates)))


def test_tensor_of_a_series_in_attoseconds_agrees_with_the_series_at_its_dates():
    # Observations 10 times their seconds after an epoch with nanoseconds, in
    # attoseconds, where its count does not fit in int64.
    series = TimeSeries([0.0, 15.0, 30.0],
                        times=TimeDeltaTensor(sb.IntTensor([0, 1_500_000_000_000_000_000,
                                                            3_000_000_000_000_000_000]), "as"),
                        epoch="2023-01-01T06:00:00.123456789")
    dates = np.array(["2023-01-01T06:00:00.123456789", "2023-01-01T06:00:01.623456789",
                      "2023-01-01T06:00:03.123456789"], dtype="datetime64[ns]")

    values = sb.tensor([series])(dates)

    assert np.array_equal(np.asarray(values[0]), np.asarray(series(dates)))
    assert np.array_equal(np.asarray(series(dates)), [0.0, 15.0, 30.0])


def test_windows_evaluate_a_date_at_its_instant_in_their_series():
    series = TimeSeries([10.0, 20.0, 30.0, 40.0],
                        times=TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "h")).with_epoch("2023-01-01T00:00")
    # The windows are [0 h, 2 h) and [1 h, 3 h), both holding 1:30.
    windows = cut_windows(series, size=TimeDelta(2, "h"), step=TimeDelta(1, "h"), closed="left", partial=False,
                          anchor="start")
    date = DateTime("2023-01-01T01:30")

    values = windows(date)

    assert np.array_equal(np.asarray(values), [series(date), series(date)])


def test_dates_on_a_tensor_with_a_series_without_an_epoch_raise():
    dated = _hourly([10.0, 20.0, 30.0], "2023-01-01T00:00")
    undated = TimeSeries([40.0, 50.0, 60.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "h"))

    with pytest.raises(ValueError, match="epoch"):
        sb.tensor([dated, undated])(DateTime("2023-01-01T01:00"))


def test_a_date_outside_a_series_of_a_tensor_raises_with_its_index_and_domain():
    early = _hourly([10.0, 20.0, 30.0], "2023-01-01T00:00")
    late = _hourly([40.0, 50.0, 60.0], "2023-01-01T01:00")

    # 00:30 is before the domain of late, which starts at 01:00.
    with pytest.raises(TimeSeriesDomainError) as excinfo:
        sb.tensor([early, late])(DateTime("2023-01-01T00:30"))

    assert excinfo.value.index == (1,)
    assert excinfo.value.time == DateTime("2023-01-01T00:30")
    assert excinfo.value.domain == late.domain


def test_a_duration_outside_a_series_of_a_tensor_raises_with_its_index():
    long = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.0, 2.0], "h"))
    short = TimeSeries([40.0, 50.0], times=TimeDeltaTensor([0.0, 1.0], "h"))

    with pytest.raises(TimeSeriesDomainError) as excinfo:
        sb.tensor([long, short])(TimeDelta(2.0, "h"))

    assert excinfo.value.index == (1,)
    assert excinfo.value.domain == short.domain


def test_a_date_outside_a_channel_of_a_series_of_a_tensor_raises_with_the_domain_of_that_channel():
    long = _hourly([10.0, 20.0, 30.0], "2023-01-01T00:00")
    short = TimeSeries([-10.0, -20.0], times=TimeDeltaTensor([0.0, 1.0], "h")).with_epoch("2023-01-01T00:00")
    series = TimeSeries.from_channels([long, short])

    # 01:30 is in the domain of long, which ends at 02:00, but after short.
    with pytest.raises(TimeSeriesDomainError) as excinfo:
        sb.tensor([series])(DateTime("2023-01-01T01:30"))

    assert excinfo.value.domain == short.domain


def _array_equal_series():
    return TimeSeries([0.0, 10.0], times=TimeDeltaTensor([0.0, 1.0], "ms"), snap_tolerance=0.25)


def test_series_array_equal_an_equal_series():
    assert _array_equal_series().array_equal(_array_equal_series())


@pytest.mark.parametrize("other", [
    pytest.param(TimeSeries([0.0, 20.0], times=TimeDeltaTensor([0.0, 1.0], "ms"), snap_tolerance=0.25),
                 id="values"),
    pytest.param(TimeSeries([0.0, 10.0], times=TimeDeltaTensor([0.0, 2.0], "ms"), snap_tolerance=0.25),
                 id="times"),
    pytest.param(TimeSeries([0.0, 10.0], times=TimeDeltaTensor([0.0, 1.0], "s"), snap_tolerance=0.25),
                 id="unit"),
    pytest.param(TimeSeries([0.0, 10.0], times=TimeDeltaTensor([0.0, 1.0], "ms"), snap_tolerance=0.5),
                 id="snap-tolerance"),
    pytest.param(TimeSeries([[0.0], [10.0]], times=TimeDeltaTensor([0.0, 1.0], "ms"), snap_tolerance=0.25),
                 id="shape"),
    pytest.param(TimeSeries.from_channels([_array_equal_series(), _array_equal_series()]), id="channels"),
    pytest.param(TimeSeries(np.array([0.0, 10.0], dtype=np.float32), times=TimeDeltaTensor([0.0, 1.0], "ms"),
                            snap_tolerance=0.25),
                 id="dtype"),
    pytest.param(sb.FloatTensor(np.array([0.0, 10.0])), id="tensor"),
])
def test_series_array_equal_rejects_a_different_series(other):
    assert not _array_equal_series().array_equal(other)


def _window_at_one(closed):
    # Each observation is 10 * t; the window at 1.0 holds 10.0 at 0 and 15.0
    # at 0.5, and 20.0 at 1 when its end is closed.
    series = TimeSeries([0.0, 5.0, 10.0, 15.0, 20.0], times=TimeDeltaTensor([0.0, 0.5, 1.0, 1.5, 2.0], "s"))
    return cut_windows(series, size=TimeDelta(1.0, "s"), step=TimeDelta(1.0, "s"), closed=closed, partial=False,
                       anchor="start")[1]


def test_series_array_equal_compares_the_start():
    # The windows differ only in whether their start is closed.
    assert not _window_at_one("both").array_equal(_window_at_one("right"))


def test_series_array_equal_compares_the_end():
    # The window [0, 1) can be evaluated after its last observation at 0.5;
    # the series cannot.
    assert not _window_at_one("left").array_equal(TimeSeries([10.0, 15.0], times=TimeDeltaTensor([0.0, 0.5], "s")))


def _gap_window_at_one(size, closed):
    # Each observation is 10 * t, with none at 3.0; the window at 1.0 holds
    # 10.0 at 0 and 20.0 at 1 and ends after its last observation.
    series = TimeSeries([0.0, 10.0, 20.0, 40.0], times=TimeDeltaTensor([0.0, 1.0, 2.0, 4.0], "s"))
    return cut_windows(series, size=size, step=TimeDelta(1.0, "s"), closed=closed, partial=False, anchor="start")[1]


def test_series_array_equal_compares_where_the_end_is():
    assert not _gap_window_at_one(TimeDelta(2.0, "s"), "left").array_equal(
        _gap_window_at_one(TimeDelta(1.5, "s"), "left"))


def test_series_array_equal_compares_whether_the_end_is_closed():
    assert not _gap_window_at_one(TimeDelta(2.0, "s"), "right").array_equal(
        _gap_window_at_one(TimeDelta(2.0, "s"), "none"))


def test_series_array_equal_compares_the_channel_order():
    a = _array_equal_series()
    b = TimeSeries([5.0, 15.0], times=TimeDeltaTensor([0.0, 1.0], "ms"), snap_tolerance=0.25)

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
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.5], "s"), epoch=as_epoch(value))

    loaded = round_trip(series)

    assert loaded.epoch == DateTime(value)
    assert loaded.epoch.to_datetime64() == value
    assert loaded.epoch.to_datetime64().dtype == value.dtype


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
def test_series_without_a_unit_survives_saving(round_trip):
    series = TimeSeries([10.0, 20.0], times=[0.0, 1.5])

    loaded = round_trip(series)

    assert loaded.unit is None
    assert loaded.array_equal(series)


def test_series_without_a_unit_rejects_an_epoch():
    with pytest.raises(ValueError, match="without a unit"):
        TimeSeries([10.0, 20.0], times=[0.0, 1.5], epoch="2023-01-01")


def test_series_without_a_unit_rejects_with_epoch():
    series = TimeSeries([10.0, 20.0], times=[0.0, 1.5])

    with pytest.raises(ValueError, match="without a unit"):
        series.with_epoch("2023-01-01")


def test_series_without_a_unit_gets_the_unit_of_to():
    series = TimeSeries([10.0, 20.0], times=[0.0, 1.5]).to("s")

    assert series.unit == "s"
    assert series.times.array_equal(TimeDeltaTensor([0.0, 1.5], "s"))


@pytest.mark.parametrize("unit", [None, "s"])
def test_from_channels_rejects_channels_with_and_without_a_unit(unit):
    with_unit = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.5], "s"))
    without_unit = TimeSeries([10.0, 20.0], times=[0.0, 1.5])

    with pytest.raises(ValueError, match="without a unit"):
        TimeSeries.from_channels([with_unit, without_unit], unit=unit)


def test_series_without_a_unit_gives_its_times_as_numbers():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    assert series.times.array_equal(sb.FloatTensor([0.0, 1.5, 4.0], dtype=sb.float64))


def test_series_without_a_unit_gives_its_domain_as_numbers():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    assert series.domain == Domain(0.0, 4.0, True, True)


def test_series_without_a_unit_gives_its_origin_as_a_number():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    assert series.origin == 0.0


def test_series_without_a_unit_gives_the_origins_of_its_channels_as_numbers():
    a = TimeSeries([10.0, 20.0], times=[0.0, 1.5])
    b = TimeSeries([-10.0, -20.0], times=[0.5, 2.0])

    series = TimeSeries.from_channels([a, b])

    assert series.origin.array_equal(sb.FloatTensor([0.0, 0.0], dtype=sb.float64))


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
def test_series_without_an_epoch_loads_without_one(round_trip):
    loaded = round_trip(TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.5], "s")))

    assert loaded.epoch is None
