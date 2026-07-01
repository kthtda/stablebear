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


# ---------------------------------------------------------------------------
# Construction error quality (issues #54, #55, #80) and building a
# BarcodeTensor directly from raw barcode ndarrays (issue #83).
# ---------------------------------------------------------------------------


def test_float_tensor_explicit_numpy_dtype():
    # FloatTensor used to silently ignore the requested dtype (issue #54).
    assert sb.FloatTensor([1.0], dtype=np.float32).dtype == sb.float32
    assert sb.FloatTensor([1.0], dtype=np.float64).dtype == sb.float64


def test_float_tensor_bad_dtype_raises():
    with pytest.raises(TypeError):
        sb.FloatTensor([1.0], dtype=sb.int32)


def test_pcf_tensor_plain_element_raises():
    # A non-element item used to leak an AttributeError (issue #80).
    with pytest.raises(TypeError, match="float"):
        sb.PcfTensor([_make_pcf(0), 3.0])


def test_pcf_tensor_mixed_precision_raises():
    # A mixed-precision list used to dump the raw pybind overloads (issue #80).
    with pytest.raises(TypeError, match="same dtype"):
        sb.PcfTensor([_make_pcf(0, np.float32), _make_pcf(1, np.float64)])


def test_l2_kernel_mixed_precision_raises():
    with pytest.raises(TypeError, match="same dtype"):
        sb.l2_kernel([_make_pcf(0, np.float32), _make_pcf(1, np.float64)])


def test_barcode_tensor_mixed_precision_raises():
    # BarcodeTensor now routes through _tensor_from_nested, so a mixed-precision
    # list raises the same "same dtype" TypeError as PcfTensor (issue #80).
    with pytest.raises(TypeError, match="same dtype"):
        BarcodeTensor([_make_barcode(0, np.float32), _make_barcode(1, np.float64)])


def test_barcode_tensor_from_ndarray_list_1d():
    # BarcodeTensor from raw (n, 2) ndarrays (issue #83).
    bcs = [np.array([[0.0, 1.0]], dtype=np.float32),
           np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)]
    t = BarcodeTensor(bcs)
    assert t.shape == (2,)
    assert t.dtype == sb.barcode32


def test_barcode_tensor_from_ndarray_list_2d():
    bcs = [[np.array([[0.0, 1.0]], dtype=np.float64)],
           [np.array([[0.0, 2.0]], dtype=np.float64)]]
    t = BarcodeTensor(bcs)
    assert t.shape == (2, 1)
    assert t.dtype == sb.barcode64


def test_barcode_tensor_invalid_leaf_dtype_raises_typeerror():
    # An unsupported leaf (e.g. an int-dtype barcode array) must surface the
    # Barcode constructor's TypeError, not an AttributeError from a
    # half-initialized Barcode.
    with pytest.raises(TypeError, match="Barcode cannot be constructed"):
        BarcodeTensor([np.array([[0, 1]], dtype=np.int64)])


def test_barcode_invalid_input_raises_typeerror():
    with pytest.raises(TypeError, match="Barcode cannot be constructed"):
        Barcode(np.array([[0, 1]], dtype=np.int64))
    with pytest.raises(TypeError, match="Barcode cannot be constructed"):
        Barcode([[0.0, 1.0]])
