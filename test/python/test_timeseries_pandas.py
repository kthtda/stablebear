import re
import subprocess
import sys

import numpy as np
import pytest

import stablebear as sb

from stablebear.timeseries import DateTime, DateTimeTensor, TimeDelta, TimeDeltaTensor, TimeSeries, cut_windows, delay_embed, to_point_clouds

pd = pytest.importorskip("pandas")

# Blocks pandas, then builds, windows, and embeds a series of 4 observations.
_WITHOUT_PANDAS = """
import sys
sys.modules["pandas"] = None

import numpy as np
from stablebear.timeseries import TimeDelta, TimeDeltaTensor, TimeSeries, cut_windows, delay_embed

series = TimeSeries(np.array([1.0, 2.0, 3.0, 4.0]), TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "s"))
windows = cut_windows(series, TimeDelta(1.0, "s"), TimeDelta(1.0, "s"), closed="left", partial=False, anchor="start")
embedded = delay_embed(series, 2, TimeDelta(1.0, "s"), direction="backward")
print(len(windows), embedded.n_observations)
"""


def test_timeseries_work_without_pandas():
    result = subprocess.run(
        [sys.executable, "-c", _WITHOUT_PANDAS],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    # Windows [0, 1), [1, 2), [2, 3); embedded points at times 1, 2, 3
    assert result.stdout.split() == ["3", "3"]


def test_datetime_keeps_nanoseconds_of_timestamp():
    date_time = DateTime(pd.Timestamp("2023-01-01T00:00:00.000000001"))

    assert date_time.unit == "ns"
    assert date_time.to_datetime64() == np.datetime64("2023-01-01T00:00:00.000000001")


def test_datetime_of_timestamp_with_a_timezone_is_its_utc_instant():
    date_time = DateTime(pd.Timestamp("2023-01-01T12:00:00.000000001", tz="Europe/Stockholm"))

    # Stockholm is at +01:00 in January
    assert date_time == DateTime(pd.Timestamp("2023-01-01T11:00:00.000000001"))


def test_series_at_an_index_with_a_timezone_equals_series_at_its_utc_index():
    stockholm = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T03:00"]).tz_localize("Europe/Stockholm")
    utc = pd.DatetimeIndex(["2023-01-01T00:00", "2023-01-01T02:00"])

    series = TimeSeries([10.0, 20.0], stockholm)

    assert series.datetimes.array_equal(TimeSeries([10.0, 20.0], utc).datetimes)


def test_series_at_a_series_of_dates_equals_series_at_its_values():
    stockholm = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T03:00"]).tz_localize("Europe/Stockholm")

    series = TimeSeries([10.0, 20.0], pd.Series(stockholm))

    assert series.datetimes.array_equal(TimeSeries([10.0, 20.0], stockholm).datetimes)


def test_series_at_an_index_with_nat_raises():
    with pytest.raises(ValueError, match="NaT"):
        TimeSeries([10.0, 20.0], pd.DatetimeIndex(["2023-01-01T00:00", None]))


def test_series_with_index_with_a_timezone_has_its_utc_epoch():
    stockholm = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T03:00"]).tz_localize("Europe/Stockholm")

    series = TimeSeries(np.array([10.0, 20.0]), stockholm)

    assert series.epoch.to_datetime64() == np.datetime64("2023-01-01T00:00")


def _dated_series():
    return TimeSeries(np.array([10.0, 20.0, 30.0]), TimeDeltaTensor([0.0, 1.5, 4.0], "h"), epoch="2023-01-01")


def test_series_at_index_of_dates_equals_series_at_its_values():
    index = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T02:00", "2023-01-01T04:00"])

    at_index = _dated_series()(index)

    assert at_index.array_equal(_dated_series()(index.to_numpy()))
    assert at_index.array_equal(sb.FloatTensor([10.0, 20.0, 30.0]))


def test_series_at_index_with_a_timezone_equals_series_at_its_utc_index():
    stockholm = pd.DatetimeIndex(["2023-01-01T02:00", "2023-01-01T03:00"]).tz_localize("Europe/Stockholm")
    utc = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T02:00"])

    assert _dated_series()(stockholm).array_equal(_dated_series()(utc))


def test_series_at_timestamp_with_a_timezone_is_at_its_utc_instant():
    # 03:00 in Stockholm is 02:00 UTC, after the observation at 01:30
    assert _dated_series()(pd.Timestamp("2023-01-01T03:00", tz="Europe/Stockholm")) == 20.0


def test_date_time_tensor_of_an_index_with_a_timezone_is_its_utc_instants():
    stockholm = pd.DatetimeIndex(["2023-01-01T02:00", "2023-01-01T03:00"]).tz_localize("Europe/Stockholm")

    assert DateTimeTensor(stockholm).array_equal(DateTimeTensor(pd.DatetimeIndex(["2023-01-01T01:00",
                                                                                  "2023-01-01T02:00"])))


def test_series_at_a_timedelta_equals_series_at_its_time_in_the_unit():
    assert _dated_series()(pd.Timedelta("90min")) == _dated_series()(TimeDelta(1.5, "h"))


def test_series_at_index_of_durations_equals_series_at_its_values():
    index = pd.TimedeltaIndex(["1h", "2h", "4h"])

    assert _dated_series()(index).array_equal(_dated_series()(index.to_numpy()))


def test_from_pandas_with_index_of_dates_counts_from_the_first_date():
    index = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T02:00", "2023-01-01T04:00"])

    series = TimeSeries.from_pandas(pd.Series([10.0, 20.0, 30.0], index=index), unit="h")

    assert series.times.array_equal(TimeDeltaTensor([0.0, 1.0, 3.0], "h"))

    assert series.epoch.to_datetime64() == np.datetime64("2023-01-01T01:00")
    assert series.values.array_equal(sb.FloatTensor([10.0, 20.0, 30.0]))


def test_from_pandas_with_an_index_of_400_years_in_ns_equals_that_index_in_seconds():
    # 400 years in nanoseconds exceed int64.
    index = pd.DatetimeIndex(["1700-01-01", "1900-01-01", "2100-01-01"]).as_unit("ns")
    observations = pd.Series([1700.0, 1900.0, 2100.0], index=index)

    in_ns = TimeSeries.from_pandas(observations, unit="d")
    in_seconds = TimeSeries.from_pandas(observations.set_axis(index.as_unit("s")), unit="d")

    assert in_ns.times.array_equal(in_seconds.times)


def test_from_pandas_with_index_of_durations_has_no_epoch():
    index = pd.to_timedelta([0, 500, 1500], unit="ms")

    series = TimeSeries.from_pandas(pd.Series([10.0, 20.0, 30.0], index=index), unit="ms")

    assert series.times.array_equal(TimeDeltaTensor([0.0, 500.0, 1500.0], "ms"))

    assert series.epoch is None


def test_from_pandas_with_range_index_counts_in_unit():
    series = TimeSeries.from_pandas(pd.Series([10.0, 20.0, 30.0]), unit="min")

    assert series.times.array_equal(TimeDeltaTensor([0.0, 1.0, 2.0], "min"))

    assert series.epoch is None


def test_from_pandas_with_numeric_index_without_unit_has_no_unit():
    series = TimeSeries.from_pandas(pd.Series([10.0, 20.0, 30.0], index=[0.0, 1.5, 4.0]))

    assert series.unit is None
    assert series.times.array_equal(sb.FloatTensor([0.0, 1.5, 4.0], dtype=sb.float64))


@pytest.mark.parametrize("values", [
    pytest.param(pd.Series([10.0, np.nan, 30.0]), id="nan"),
    pytest.param(pd.Series([10.0, None, 30.0], dtype="Float64"), id="na"),
])
def test_from_pandas_keeps_missing_values_as_nan(values):
    series = TimeSeries.from_pandas(values, unit="s")

    assert np.array_equal(np.asarray(series.values), [10.0, np.nan, 30.0], equal_nan=True)


@pytest.mark.parametrize("values, dtype", [
    pytest.param([10, 20, 30], sb.int64, id="int64"),
    pytest.param([True, False, True], sb.boolean, id="bool"),
])
def test_from_pandas_keeps_the_dtype(values, dtype):
    series = TimeSeries.from_pandas(pd.Series(values), unit="s")

    assert series.dtype is dtype


def _frame(columns):
    return pd.DataFrame(columns, index=pd.to_timedelta([0, 500, 1500], unit="ms"))


def test_from_pandas_with_a_data_frame_gives_one_channel_per_column():
    frame = _frame({"a": [10.0, 20.0, 30.0], "b": [-1.0, -2.0, -3.0]})

    series = TimeSeries.from_pandas(frame)

    assert series.n_channels == 2
    for i in range(2):
        assert series[i].array_equal(TimeSeries.from_pandas(frame.iloc[:, i]))


def test_from_pandas_with_int_and_float_columns_gives_float64_channels():
    series = TimeSeries.from_pandas(_frame({"a": [10, 20, 30], "b": [-1.5, -2.5, -3.5]}))

    assert series.dtype is sb.float64
    assert series[0].values.array_equal(sb.FloatTensor([10.0, 20.0, 30.0], dtype=sb.float64))


def test_from_pandas_with_a_data_frame_keeps_pd_na_as_nan():
    series = TimeSeries.from_pandas(_frame({"a": pd.array([10.0, None, 30.0], dtype="Float64"),
                                            "b": [1.0, 2.0, 3.0]}))

    assert np.array_equal(np.asarray(series[0].values), [10.0, np.nan, 30.0], equal_nan=True)


def test_from_pandas_with_a_string_column_raises():
    with pytest.raises(TypeError):
        TimeSeries.from_pandas(_frame({"a": [10.0, 20.0, 30.0], "b": ["x", "y", "z"]}))


def test_from_pandas_with_a_column_of_durations_raises():
    index = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T02:00"])

    with pytest.raises(TypeError, match="Durations cannot be observations"):
        TimeSeries.from_pandas(pd.Series(pd.to_timedelta([1, 2], unit="s"), index=index))


def test_time_delta_tensor_rejects_an_index_of_durations_with_a_unit():
    with pytest.raises(TypeError, match="own unit"):
        TimeDeltaTensor(pd.TimedeltaIndex(["1h", "2h"]), "s")


def test_from_pandas_with_a_data_frame_without_columns_raises():
    with pytest.raises(ValueError, match="column"):
        TimeSeries.from_pandas(_frame({}))


@pytest.mark.parametrize("index, message", [
    pytest.param(pd.DatetimeIndex(["2023-01-01", None]), "notna()", id="NaT"),
    pytest.param(pd.to_timedelta([5, 1], unit="s"), "sort_index()", id="unsorted"),
    pytest.param(pd.to_timedelta([1, 1], unit="s"), "index.duplicated()", id="duplicated"),
])
def test_from_pandas_rejects_an_index_naming_the_fix(index, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        TimeSeries.from_pandas(pd.Series([10.0, 20.0], index=index))


@pytest.mark.parametrize("obj", [
    pytest.param(pd.Series([10.0, 20.0], index=pd.period_range("2023-01", periods=2, freq="M")), id="PeriodIndex"),
    pytest.param(pd.Series([10.0, 20.0], index=pd.MultiIndex.from_tuples([(0, 1), (1, 2)])), id="MultiIndex"),
    pytest.param(np.array([10.0, 20.0]), id="ndarray"),
])
def test_from_pandas_rejects_other_indexes_and_objects(obj):
    with pytest.raises(TypeError, match="Expected"):
        TimeSeries.from_pandas(obj)


def test_time_delta_keeps_nanoseconds_of_timedelta():
    time_delta = TimeDelta(pd.Timedelta(1, "ns"))

    assert time_delta == TimeDelta(1, "ns")

def test_series_at_a_timedelta_index_equals_series_at_its_array():
    index = pd.to_timedelta([0, 500, 1500], unit="ms")

    series = TimeSeries([10.0, 20.0, 30.0], index)

    assert series.array_equal(TimeSeries([10.0, 20.0, 30.0], index.to_numpy()))

def test_window_size_of_one_nanosecond_timedelta():
    series = TimeSeries(np.array([1.0, 2.0, 3.0, 4.0]), TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "ns"))

    windows = cut_windows(series, pd.Timedelta(1, "ns"), pd.Timedelta(1, "ns"), closed="left", partial=False,
                          anchor="start")

    # Windows [0, 1), [1, 2), [2, 3) ns
    assert len(windows) == 3


def test_windows_of_a_pandas_timedelta_size_and_step_equal_those_of_their_time_deltas():
    series = TimeSeries(np.arange(10.0), times=TimeDeltaTensor(np.arange(10.0), "h"))

    windows = cut_windows(series, size=pd.Timedelta(2, "h"), step=pd.Timedelta(30, "min"), closed="left",
                          partial=False, anchor="start")

    assert windows.array_equal(cut_windows(series, size=TimeDelta(2, "h"), step=TimeDelta(30, "min"),
                                           closed="left", partial=False, anchor="start"))


def test_length_of_a_window_from_pandas_equals_its_size():
    levels = pd.Series(np.arange(10.0), index=pd.date_range("2023-01-01", periods=10, freq="h", unit="ns"))
    window = cut_windows(TimeSeries.from_pandas(levels), size=TimeDelta(2, "h"), step=TimeDelta(1, "h"),
                         closed="left", partial=False, anchor="start")[0]

    assert window.domain.end - window.domain.start == TimeDelta(2, "h")


def test_embedding_delay_timedelta_equals_timedelta64():
    series = TimeSeries(np.array([1.0, 2.0, 3.0, 4.0]), TimeDeltaTensor([0.0, 1.0, 2.0, 3.0], "h"))

    from_pandas = delay_embed(series, 2, pd.Timedelta(3, "h"), direction="backward")
    from_numpy = delay_embed(series, 2, np.timedelta64(3, "h"), direction="backward")

    assert from_pandas.array_equal(from_numpy)


def test_to_pandas_gives_back_a_series_with_dates():
    index = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T02:30", "2023-01-02T00:00"])
    original = pd.Series([1.0, 2.0, 3.0], index=index)

    converted = TimeSeries.from_pandas(original).to_pandas()

    pd.testing.assert_series_equal(converted, original)


def test_to_pandas_of_a_series_without_an_epoch_has_a_timedelta_index():
    series = TimeSeries([10, 20], times=TimeDeltaTensor([0.0, 1.5], "s"))

    converted = series.to_pandas()

    assert converted.index.dtype == np.dtype("timedelta64[us]")
    assert np.array_equal(converted.index.to_numpy(), np.array([0, 1_500_000], dtype="timedelta64[us]"))
    assert converted.dtype == np.int64


def test_to_pandas_of_a_series_without_a_unit_has_a_numeric_index():
    converted = TimeSeries([10.0, 20.0], times=[0.0, 1.5]).to_pandas()

    assert converted.index.equals(pd.Index([0.0, 1.5]))


def test_to_pandas_of_picoseconds_raises_without_a_resolution():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1500.0], "ps"))

    with pytest.raises(ValueError, match="resolution='ns'"):
        series.to_pandas()


def test_to_pandas_of_picoseconds_rounds_to_nanoseconds_with_a_resolution():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1500.0], "ps"))

    converted = series.to_pandas(resolution="ns")

    assert np.array_equal(converted.index.to_numpy(), np.array([0, 2], dtype="timedelta64[ns]"))


def test_to_pandas_rejects_a_resolution_pandas_does_not_support():
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.5], "h"))

    with pytest.raises(ValueError, match="resolution"):
        series.to_pandas(resolution="h")


def test_to_pandas_of_a_window_starts_at_its_start_date():
    series = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([0.0, 1.5, 4.0], "h"), epoch="2023-01-01")
    window = cut_windows(series, size=TimeDelta(2, "h"), step=TimeDelta(1, "h"), closed="left", partial=False,
                         anchor="start")[1]

    converted = window.to_pandas()

    assert np.array_equal(converted.index.to_numpy(),
                          np.array(["2023-01-01T01:00", "2023-01-01T01:30"], dtype="datetime64[us]"))


def test_to_pandas_of_a_series_of_pcfs_raises():
    series = TimeSeries(sb.zeros((2,), dtype=sb.pcf64), times=TimeDeltaTensor([0.0, 1.0], "s"))

    with pytest.raises(TypeError, match="scalar numbers"):
        series.to_pandas()


def test_to_pandas_without_pandas_says_pandas_is_needed(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    series = TimeSeries([10.0, 20.0], times=TimeDeltaTensor([0.0, 1.5], "s"))

    with pytest.raises(ImportError, match="needs the pandas package"):
        series.to_pandas()


def test_to_pandas_of_several_channels_gives_back_a_data_frame():
    index = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T02:30", "2023-01-02T00:00"])
    original = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}, index=index)

    converted = TimeSeries.from_pandas(original).to_pandas()

    pd.testing.assert_frame_equal(converted, original)


def test_to_pandas_of_channels_with_different_times_holds_each_value_in_its_domain():
    # Channel a can be evaluated from 0 s to 2 s, channel b from 0.5 s to 3 s.
    a = TimeSeries([1.0, 2.0], times=TimeDeltaTensor([0.0, 2.0], "s"))
    b = TimeSeries([10.0, 20.0, 30.0], times=TimeDeltaTensor([500.0, 1000.0, 3000.0], "ms"))

    converted = TimeSeries.from_channels([a, b], unit="s").to_pandas()

    expected = pd.DataFrame({0: [1.0, 1.0, 1.0, 2.0, np.nan], 1: [np.nan, 10.0, 20.0, 20.0, 30.0]},
                            index=pd.to_timedelta([0, 500, 1000, 2000, 3000], unit="ms").as_unit("us"))
    pd.testing.assert_frame_equal(converted, expected)


def test_to_pandas_of_integer_channels_in_their_domains_keeps_integers():
    # Channel b has an observation at 1 s, where channel a holds 1.
    a = TimeSeries([1, 2], times=TimeDeltaTensor([0.0, 2.0], "s"))
    b = TimeSeries([10, 20, 30], times=TimeDeltaTensor([0.0, 1.0, 2.0], "s"))

    converted = TimeSeries.from_channels([a, b]).to_pandas()

    assert converted[0].tolist() == [1, 1, 2]
    assert converted.dtypes.tolist() == [np.int64, np.int64]


def _hours():
    return pd.date_range("2023-01-01", periods=10, freq="h")


def test_windows_of_a_pandas_series_equal_those_of_its_time_series():
    hourly = pd.Series(np.arange(10.0), index=_hours())
    windows = cut_windows(hourly, size=TimeDelta(3, "h"), step=TimeDelta(1, "h"), closed="left", partial=False,
                          anchor="start")
    expected = cut_windows(TimeSeries.from_pandas(hourly), size=TimeDelta(3, "h"), step=TimeDelta(1, "h"),
                           closed="left", partial=False, anchor="start")

    assert windows.array_equal(expected)
    assert [w.domain for w in windows] == [w.domain for w in expected]


def test_embedding_of_a_data_frame_equals_that_of_its_time_series():
    frame = pd.DataFrame({0: np.arange(10.0), 1: -np.arange(10.0)}, index=_hours())
    embedding = delay_embed(frame, dimension=2, delay=TimeDelta(2, "h"), direction="backward")

    assert embedding.array_equal(delay_embed(TimeSeries.from_pandas(frame), dimension=2, delay=TimeDelta(2, "h"),
                                             direction="backward"))


def test_point_cloud_of_a_pandas_series_equals_that_of_its_time_series():
    hourly = pd.Series(10.0 * np.arange(10.0), index=_hours())

    cloud = to_point_clouds(hourly)

    assert np.array_equal(np.asarray(cloud), np.asarray(to_point_clouds(TimeSeries.from_pandas(hourly))))
