import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import Barcode, BarcodeTensor


def _make_pcf(val, dtype=np.float32):
    return sb.Pcf(np.array([[0.0, val], [1.0, val + 1]], dtype=dtype))


def _make_int_pcf(val, dtype=np.int32):
    return sb.Pcf(np.array([[0, val], [1, val + 1]], dtype=dtype))


def _make_barcode(offset=0.0, dtype=np.float64):
    return Barcode(np.array([[offset, offset + 1.0]], dtype=dtype))


# --- PcfTensor ---


def test_pcf_tensor_1d():
    fs = [_make_pcf(i) for i in range(3)]
    t = sb.PcfTensor(fs)
    assert t.shape == (3,)
    assert t.dtype == sb.pcf32


def test_pcf_tensor_2d():
    fs = [[_make_pcf(i * 3 + j) for j in range(3)] for i in range(2)]
    t = sb.PcfTensor(fs)
    assert t.shape == (2, 3)
    assert t.dtype == sb.pcf32


def test_pcf_tensor_3d():
    fs = [[[_make_pcf(i * 6 + j * 3 + k) for k in range(3)]
           for j in range(2)] for i in range(2)]
    t = sb.PcfTensor(fs)
    assert t.shape == (2, 2, 3)


def test_pcf_tensor_f64():
    fs = [_make_pcf(i, np.float64) for i in range(4)]
    t = sb.PcfTensor(fs)
    assert t.shape == (4,)
    assert t.dtype == sb.pcf64


def test_pcf_tensor_tuple():
    fs = (_make_pcf(0), _make_pcf(1))
    t = sb.PcfTensor(fs)
    assert t.shape == (2,)


def test_pcf_tensor_roundtrip_elements():
    fs = [_make_pcf(i) for i in range(4)]
    t = sb.PcfTensor(fs)
    for i, f in enumerate(fs):
        assert f == t[i]


def test_pcf_tensor_2d_roundtrip_elements():
    fs = [[_make_pcf(i * 2 + j) for j in range(2)] for i in range(3)]
    t = sb.PcfTensor(fs)
    for i in range(3):
        for j in range(2):
            assert fs[i][j] == t[i, j]


# --- IntPcfTensor ---


def test_int_pcf_tensor_1d():
    fs = [_make_int_pcf(i) for i in range(3)]
    t = sb.IntPcfTensor(fs)
    assert t.shape == (3,)
    assert t.dtype == sb.pcf32i


def test_int_pcf_tensor_2d():
    fs = [[_make_int_pcf(i * 2 + j) for j in range(2)] for i in range(2)]
    t = sb.IntPcfTensor(fs)
    assert t.shape == (2, 2)
    assert t.dtype == sb.pcf32i


def test_int_pcf_tensor_i64():
    fs = [_make_int_pcf(i, np.int64) for i in range(2)]
    t = sb.IntPcfTensor(fs)
    assert t.dtype == sb.pcf64i


# --- BarcodeTensor ---


def test_barcode_tensor_1d():
    bcs = [_make_barcode(i) for i in range(3)]
    t = BarcodeTensor(bcs)
    assert t.shape == (3,)
    assert t.dtype == sb.barcode64


def test_barcode_tensor_2d():
    bcs = [[_make_barcode(i * 2 + j) for j in range(2)] for i in range(2)]
    t = BarcodeTensor(bcs)
    assert t.shape == (2, 2)
    assert t.dtype == sb.barcode64


def test_barcode_tensor_f32():
    bcs = [_make_barcode(i, np.float32) for i in range(2)]
    t = BarcodeTensor(bcs)
    assert t.dtype == sb.barcode32


# --- Error cases ---


def test_empty_list():
    t = sb.PcfTensor([])
    assert t.shape == (0,)


def test_ragged_list_raises():
    with pytest.raises(ValueError):
        sb.PcfTensor([[_make_pcf(0), _make_pcf(1)], [_make_pcf(2)]])
