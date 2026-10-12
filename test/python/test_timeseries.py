import numpy as np
import pytest

import stablebear as sb
from stablebear.timeseries import Times, TimeSeries


def test_times_keeps_values_and_unit():
    times = Times([0.0, 1.5, 4.0], unit="ms")

    assert isinstance(times.values, sb.FloatTensor)
    assert times.values.dtype is sb.float64
    assert np.array_equal(np.asarray(times.values), [0.0, 1.5, 4.0])
    assert times.unit == "ms"


@pytest.mark.parametrize("values, unit", [
    ([0.0, 0.0], "s"),  # duplicate timestamps
    ([1.0, 0.0], "s"),  # decreasing timestamps
    ([0.0, np.nan], "s"),
    ([0.0, np.inf], "s"),
    ([[0.0, 1.0]], "s"),  # two-dimensional
    ([0.0, 1.0], "yr"),  # unknown unit
])
def test_times_rejects_invalid_input(values, unit):
    with pytest.raises(ValueError):
        Times(values, unit=unit)



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


def test_changing_times_values_leaves_times_unchanged():
    times = Times([0.0, 1.0], unit="s")
    values = times.values

    values[0] = 5.0

    assert values[0] == 5.0
    assert np.array_equal(np.asarray(times.values), [0.0, 1.0])


@pytest.mark.parametrize("values, unit, target, expected", [
    ([0.0, 1.5], "min", "s", [0.0, 90.0]),
    ([0.0, 1.5], "fs", "as", [0.0, 1500.0]),
    ([0.0, 2.25], "as", "fs", [0.0, 0.00225]),
])
def test_times_converts_units(values, unit, target, expected):
    converted = Times(values, unit=unit).to(target)

    assert np.array_equal(np.asarray(converted.values), expected)
    assert converted.unit == target


def test_series_defaults_to_seconds_and_one_channel():
    # This test checks the default unit and channel name.
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    assert series.unit == "s"
    assert series.channel_names == ("0",)
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
