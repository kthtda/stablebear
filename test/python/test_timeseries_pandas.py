import subprocess
import sys

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
