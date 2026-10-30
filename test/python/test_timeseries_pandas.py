import subprocess
import sys

import numpy as np
import pytest

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
