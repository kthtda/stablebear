"""Generate a pinned V3 time-series corpus using the installed current build."""

import numpy as np
import stablebear as sb
from stablebear.persistence import Barcode
from stablebear.timeseries import Times, TimeSeries, Window

from pinned_corpus import write_pinned_corpus


INF = float("inf")


def _array(dtype, shape, values):
    return {"dtype": dtype, "shape": shape, "values": values}


def _floats(values):
    return {
        "accepted_types": ["FloatTensor"], "comparison": "array",
        "dtype": "float64", "shape": [len(values)],
        "array": _array("float64", [len(values)], values),
    }


def _channel(times, values, evaluations):
    """A channel's times and values, and its value at each listed time; None
    marks a time outside its domain."""
    return {
        "times": _array("float64", [len(times)], times),
        "values": values,
        "evaluations": [
            {"t": t, "outside": True} if value is None
            else {"t": t, "value": _array("float64", [], value)}
            for t, value in evaluations
        ],
    }


def _series(dtype, unit, channels):
    return {
        "accepted_types": ["TimeSeries"], "comparison": "series",
        "dtype": dtype, "unit": unit, "channels": channels,
    }


def _barcode_series():
    # Item 28's result for three clouds: two points 1 apart, a unit square,
    # and points at 0, 2, and 5 on a line, in degrees 0 and 1.
    bars = [
        [[[0, INF], [0, 1]], []],
        [[[0, INF], [0, 1], [0, 1], [0, 1]], [[1, 1.4142135381698608]]],
        [[[0, INF], [0, 2], [0, 3]], []],
    ]
    barcodes = sb.zeros((3, 2), dtype=sb.barcode32)
    elements = []
    for i, observation in enumerate(bars):
        for j, values in enumerate(observation):
            barcodes[i, j] = Barcode(np.array(values, dtype=np.float32).reshape(-1, 2))
            elements.append({"index": [i, j], "array": _array("float32", [len(values), 2], values)})
    series = TimeSeries(barcodes, times=Times([0.0, 1.3, 4.0], unit="ms"))
    expected = _series("barcode32", "ms", [_channel(
        [0.0, 1.3, 4.0],
        {
            "accepted_types": ["BarcodeTensor"], "comparison": "elements",
            "dtype": "barcode32", "shape": [3, 2], "elements": elements,
        },
        [],
    )])
    return series, expected


def _channels_series():
    # Channel B is stored in milliseconds and read in seconds.
    a = TimeSeries([10.0, 20.0, 30.0], times=Times([0.0, 1.0, 2.5], unit="s"))
    b = TimeSeries([-1.0, -2.0, -3.0], times=Times([500.0, 1500.0, 3000.0], unit="ms"))
    series = TimeSeries.from_channels([a, b], unit="s")
    expected = _series("float64", "s", [
        _channel([0.0, 1.0, 2.5], _floats([10.0, 20.0, 30.0]),
                 [(0.0, 10.0), (1.75, 20.0), (2.5, 30.0), (3.0, None)]),
        _channel([0.5, 1.5, 3.0], _floats([-1.0, -2.0, -3.0]),
                 [(0.0, None), (1.0, -1.0), (3.0, -3.0)]),
    ])
    return series, expected


def _windows(closed, windows):
    # Each observation is 10 * t, with none at 3.0, so the window at 1.0 ends
    # after its last observation.
    series = TimeSeries([0.0, 10.0, 20.0, 40.0], times=[0.0, 1.0, 2.0, 4.0])
    tensor = Window(size=2.0, step=1.0, closed=closed, partial=False, anchor="start")(series)
    expected = {
        "accepted_types": ["TimeSeriesTensor"], "comparison": "series_elements",
        "dtype": "series(float64)", "shape": [len(windows)],
        "elements": [
            {"index": [i], "series": _series("float64", "s", [_channel(times, _floats(values), evaluations)])}
            for i, (times, values, evaluations) in enumerate(windows)
        ],
    }
    return tensor, expected


def _open_windows():
    # (0, 2), (1, 3), and (2, 4): open starts, and open ends after the last
    # observation.
    return _windows("none", [
        ([0.0, 1.0], [0.0, 10.0], [(0.0, None), (0.5, 0.0), (1.5, 10.0), (2.0, None)]),
        ([0.0, 1.0], [10.0, 20.0], [(0.0, None), (0.5, 10.0), (1.5, 20.0), (2.0, None)]),
        ([0.0], [20.0], [(0.0, None), (1.5, 20.0), (2.0, None)]),
    ])


def _closed_windows():
    # [0, 2], [1, 3], and [2, 4]: closed starts, and a closed end after the
    # last observation for the window at 1.0.
    return _windows("both", [
        ([0.0, 1.0, 2.0], [0.0, 10.0, 20.0], [(0.0, 0.0), (1.5, 10.0), (2.0, 20.0)]),
        ([0.0, 1.0], [10.0, 20.0], [(0.0, 10.0), (1.5, 20.0), (2.0, 20.0)]),
        ([0.0, 2.0], [20.0, 40.0], [(0.0, 20.0), (1.5, 20.0), (2.0, 40.0)]),
    ])


def _cases():
    for name, description, make in (
        ("series_barcode32", "Series of barcodes of three point clouds", _barcode_series),
        ("series_channels_float64", "Series of two channels in different units", _channels_series),
        ("windows_open_float64", "Windows with open starts and ends", _open_windows),
        ("windows_closed_float64", "Windows with closed starts and ends", _closed_windows),
    ):
        obj, expected = make()
        yield name, description, obj, expected


if __name__ == "__main__":
    write_pinned_corpus("0.5-pre-timeseries", _cases())
