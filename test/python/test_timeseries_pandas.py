import subprocess
import sys

import numpy as np
import pytest

import stablebear as sb

from stablebear.timeseries import DateTime, TimeDelayEmbedding, Times, TimeSeries, Window

pd = pytest.importorskip("pandas")

# Blocks pandas, then builds, windows, and embeds a series of 4 observations.
_WITHOUT_PANDAS = """
import sys
sys.modules["pandas"] = None

import numpy as np
from stablebear.timeseries import TimeDelayEmbedding, TimeSeries, Window

series = TimeSeries(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0, 3.0]))
windows = Window(1.0, 1.0, closed="left", partial=False, anchor="start")(series)
embedded = TimeDelayEmbedding(2, 1.0, direction="backward")(series)
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


def test_times_of_index_with_a_timezone_equal_times_of_its_utc_index():
    stockholm = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T03:00"]).tz_localize("Europe/Stockholm")
    utc = pd.DatetimeIndex(["2023-01-01T00:00", "2023-01-01T02:00"])

    assert Times(stockholm, unit="h") == Times(utc, unit="h")


def test_times_of_series_of_dates_equal_times_of_its_values():
    stockholm = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T03:00"]).tz_localize("Europe/Stockholm")

    assert Times(pd.Series(stockholm), unit="h") == Times(stockholm, unit="h")


def test_times_of_index_with_nat_raises():
    with pytest.raises(ValueError, match="NaT"):
        Times(pd.DatetimeIndex(["2023-01-01T00:00", None]), unit="h")


def test_series_with_index_with_a_timezone_has_its_utc_epoch():
    stockholm = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T03:00"]).tz_localize("Europe/Stockholm")

    series = TimeSeries(np.array([10.0, 20.0]), stockholm)

    assert series.epoch.to_datetime64() == np.datetime64("2023-01-01T00:00")


def _dated_series():
    return TimeSeries(np.array([10.0, 20.0, 30.0]), Times([0.0, 1.5, 4.0], unit="h", epoch="2023-01-01"))


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


def test_from_pandas_with_index_of_dates_counts_from_the_first_date():
    index = pd.DatetimeIndex(["2023-01-01T01:00", "2023-01-01T02:00", "2023-01-01T04:00"])

    series = TimeSeries.from_pandas(pd.Series([10.0, 20.0, 30.0], index=index), unit="h")

    assert series.times == Times([0.0, 1.0, 3.0], unit="h", epoch=np.datetime64("2023-01-01T01:00", "ns"))
    assert series.values.array_equal(sb.FloatTensor([10.0, 20.0, 30.0]))


def test_from_pandas_with_index_of_durations_has_no_epoch():
    index = pd.to_timedelta([0, 500, 1500], unit="ms")

    series = TimeSeries.from_pandas(pd.Series([10.0, 20.0, 30.0], index=index), unit="ms")

    assert series.times == Times([0.0, 500.0, 1500.0], unit="ms")


def test_from_pandas_with_range_index_counts_in_unit():
    series = TimeSeries.from_pandas(pd.Series([10.0, 20.0, 30.0]), unit="min")

    assert series.times == Times([0.0, 1.0, 2.0], unit="min")


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


def test_window_size_of_one_nanosecond_timedelta():
    series = TimeSeries(np.array([1.0, 2.0, 3.0, 4.0]), Times([0.0, 1.0, 2.0, 3.0], unit="ns"))

    windows = Window(pd.Timedelta(1, "ns"), pd.Timedelta(1, "ns"), closed="left", partial=False,
                     anchor="start")(series)

    # Windows [0, 1), [1, 2), [2, 3) ns
    assert len(windows) == 3


def test_embedding_delay_timedelta_equals_timedelta64():
    series = TimeSeries(np.array([1.0, 2.0, 3.0, 4.0]), Times([0.0, 1.0, 2.0, 3.0], unit="h"))

    from_pandas = TimeDelayEmbedding(2, pd.Timedelta(3, "h"), direction="backward")(series)
    from_numpy = TimeDelayEmbedding(2, np.timedelta64(3, "h"), direction="backward")(series)

    assert from_pandas.array_equal(from_numpy)
