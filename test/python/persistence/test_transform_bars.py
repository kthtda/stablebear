import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import Barcode, BarcodeTensor, transform_bars
from stablebear.timeseries import TimeDeltaTensor, TimeSeries


def _f():
    return sb.Plf([[0.0, 0.0], [1.0, 2.0], [3.0, 3.0]])


def test_endpoints_map_through_plf():
    bc = Barcode([[0.0, 1.0], [0.5, 2.0], [1.0, 5.0]])
    out = transform_bars(bc, _f())
    assert isinstance(out, Barcode)
    np.testing.assert_allclose(out.to_numpy(), [[0.0, 2.0], [1.0, 2.5], [2.0, 4.0]])


@pytest.mark.parametrize("points, expected", [
    ([[0.0, 0.0], [1.0, 2.0]], np.inf),
    ([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0]], 2.0),
])
def test_infinite_death_maps_to_limit(points, expected):
    out = transform_bars(Barcode([[0.0, np.inf]]), sb.Plf(points))
    assert out.to_numpy()[0, 1] == expected


def test_bars_with_birth_at_or_after_death_are_removed():
    f = sb.Plf([[0.0, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 1.0]])
    bc = Barcode([[0.0, 1.0], [1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [0.5, 2.5]])
    np.testing.assert_allclose(transform_bars(bc, f).to_numpy(), [[0.0, 2.0], [1.0, 1.5]])


def test_empty_barcode():
    assert len(transform_bars(Barcode(np.zeros((0, 2))), _f())) == 0


def test_tensor_keeps_shape_and_transforms_each_barcode():
    bcs = [Barcode([[0.0, 1.0]]), Barcode([[0.5, 2.0], [1.0, 3.0]]), Barcode([[2.0, 3.0]])]
    out = transform_bars(BarcodeTensor(bcs).reshape((1, 3)), _f())
    assert out.shape == (1, 3)
    for i, bc in enumerate(bcs):
        np.testing.assert_allclose(out[0, i].to_numpy(), _f()(bc.to_numpy()))


@pytest.mark.parametrize("points", [[[0.0, 3.0], [1.0, 2.0]], [[0.0, 3.0]]])
def test_infinite_bar_removed_when_limit_is_not_above_birth(points):
    assert len(transform_bars(Barcode([[0.0, np.inf]]), sb.Plf(points))) == 0


def test_float32_barcode_stays_float32():
    out = transform_bars(Barcode(np.array([[0.0, 1.0]], dtype=np.float32)), _f())
    assert out.dtype == sb.barcode32
    np.testing.assert_allclose(out.to_numpy(), [[0.0, 2.0]])


def test_series_of_barcodes():
    bcs = BarcodeTensor([Barcode([[0.0, 1.0]]), Barcode([[0.5, 2.0]])])
    series = TimeSeries(bcs, times=TimeDeltaTensor([0.0, 1.0], "s"))
    out = transform_bars(series, _f())
    assert isinstance(out, TimeSeries)
    np.testing.assert_allclose(out.values[1].to_numpy(), [[1.0, 2.5]])


def test_negative_endpoint_raises():
    with pytest.raises(ValueError, match="before time 0"):
        transform_bars(Barcode([[-1.0, 1.0]]), _f())


def test_non_plf_raises():
    with pytest.raises(TypeError, match="Plf"):
        transform_bars(Barcode([[0.0, 1.0]]), sb.Pcf(np.array([[0.0, 1.0]])))  # pyright: ignore[reportCallIssue, reportArgumentType]
