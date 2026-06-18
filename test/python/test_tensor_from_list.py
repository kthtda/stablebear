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


# ---------------------------------------------------------------------------
# Bug #4: ragged nested lists were silently accepted (and elements placed at
# wrong positions) whenever the total element count happened to equal a
# rectangular product, because validation only inspected the first sub-list at
# each depth. Every branch is now validated.
# ---------------------------------------------------------------------------


def test_ragged_with_matching_total_raises():
    """Rows of length 2, 3, 1 (total 6 == 3*2) was silently accepted as (3, 2)."""
    data = [[_make_pcf(0), _make_pcf(1)],
            [_make_pcf(2), _make_pcf(3), _make_pcf(4)],
            [_make_pcf(5)]]
    with pytest.raises(ValueError, match="Ragged nested list"):
        sb.PcfTensor(data)


def test_rank3_ragged_raises():
    """Non-first-branch raggedness at depth 2 must be caught."""
    data = [[[_make_pcf(0), _make_pcf(1)], [_make_pcf(2), _make_pcf(3)]],
            [[_make_pcf(4)], [_make_pcf(5), _make_pcf(6), _make_pcf(7)]]]
    with pytest.raises(ValueError, match="Ragged nested list"):
        sb.PcfTensor(data)


def test_ragged_intpcf_raises():
    data = [[_make_int_pcf(0), _make_int_pcf(1)],
            [_make_int_pcf(2), _make_int_pcf(3), _make_int_pcf(4)],
            [_make_int_pcf(5)]]
    with pytest.raises(ValueError, match="Ragged nested list"):
        sb.IntPcfTensor(data)


def test_mixed_depth_scalar_among_lists_raises():
    """A non-sequence sibling where a sub-list is expected is also ragged."""
    with pytest.raises(ValueError, match="Ragged nested list"):
        sb.PcfTensor([[_make_pcf(0), _make_pcf(1)], _make_pcf(2)])
