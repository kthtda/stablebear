import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import Barcode, BarcodeTensor
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



def _pcfs(np_dtype):
    return sb.tensor([
        sb.Pcf(np.array([[0, 1], [1, 0]], dtype=np_dtype)),
        sb.Pcf(np.array([[0, 2], [1, 1], [2, 0]], dtype=np_dtype)),
    ])


def _point_clouds(dtype, np_dtype):
    # Clouds of 1 and 2 points.
    return sb.PointCloudTensor([
        np.array([[0.0, 0.0]], dtype=np_dtype),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np_dtype),
    ], dtype=dtype)


def _distance_matrices(dtype, np_dtype):
    tensor = sb.zeros((2,), dtype=dtype)
    tensor[0] = sb.DistanceMatrix(np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=np_dtype))
    tensor[1] = sb.DistanceMatrix(np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 3.0],
        [2.0, 3.0, 0.0],
    ], dtype=np_dtype))
    return tensor


def _symmetric_matrices(dtype, np_dtype):
    tensor = sb.zeros((2,), dtype=dtype)
    tensor[0] = sb.SymmetricMatrix(np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ], dtype=np_dtype))
    tensor[1] = sb.SymmetricMatrix(np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0],
    ], dtype=np_dtype))
    return tensor


def _barcodes(np_dtype):
    # Barcodes with 1 and 2 bars.
    return BarcodeTensor([
        Barcode(np.array([[0.0, 1.0]], dtype=np_dtype)),
        Barcode(np.array([[0.0, 2.0], [0.5, 1.5]], dtype=np_dtype)),
    ])


@pytest.mark.parametrize("make_values", [
    pytest.param(lambda: sb.tensor(np.array([True, False])), id="boolean"),
    pytest.param(lambda: sb.tensor(np.array([1.0, 2.0], dtype=np.float32)), id="float32"),
    pytest.param(lambda: sb.tensor(np.array([1.0, 2.0], dtype=np.float64)), id="float64"),
    pytest.param(lambda: sb.tensor(np.array([1, 2], dtype=np.int32)), id="int32"),
    pytest.param(lambda: sb.tensor(np.array([1, 2], dtype=np.int64)), id="int64"),
    pytest.param(lambda: sb.tensor(np.array([1, 2], dtype=np.uint32)), id="uint32"),
    pytest.param(lambda: sb.tensor(np.array([1, 2], dtype=np.uint64)), id="uint64"),
    pytest.param(lambda: _pcfs(np.float32), id="pcf32"),
    pytest.param(lambda: _pcfs(np.float64), id="pcf64"),
    pytest.param(lambda: _pcfs(np.int32), id="pcf32i"),
    pytest.param(lambda: _pcfs(np.int64), id="pcf64i"),
    pytest.param(lambda: _point_clouds(sb.pcloud32, np.float32), id="pcloud32"),
    pytest.param(lambda: _point_clouds(sb.pcloud64, np.float64), id="pcloud64"),
    pytest.param(lambda: _distance_matrices(sb.distmat32, np.float32), id="distmat32"),
    pytest.param(lambda: _distance_matrices(sb.distmat64, np.float64), id="distmat64"),
    pytest.param(lambda: _symmetric_matrices(sb.symmat32, np.float32), id="symmat32"),
    pytest.param(lambda: _symmetric_matrices(sb.symmat64, np.float64), id="symmat64"),
    pytest.param(lambda: _barcodes(np.float32), id="barcode32"),
    pytest.param(lambda: _barcodes(np.float64), id="barcode64"),
])
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


def test_observation_by_index():
    values = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
    series = TimeSeries(values, times=[0.0, 1.0, 2.0])

    assert np.array_equal(np.asarray(series.observation(1)), [3.0, 4.0])
    assert np.array_equal(np.asarray(series.observation(-1)), [5.0, 6.0])
    with pytest.raises(IndexError):
        series.observation(3)


def test_series_evaluates_to_previous_value():
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    assert series(0.0) == 10.0
    assert series(1.0) == 10.0
    assert series(1.5) == 20.0
    assert series(3.75) == 20.0
    assert series(4.0) == 30.0
    assert series(1) == 10.0


@pytest.mark.parametrize("t", [-0.5, 4.5])
def test_series_rejects_times_outside_domain(t):
    series = TimeSeries([10.0, 20.0, 30.0], times=[0.0, 1.5, 4.0])

    assert series.domain == (0.0, 4.0)
    with pytest.raises(ValueError):
        series(t)
