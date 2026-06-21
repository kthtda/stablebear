import numpy as np
import pytest

import stablebear as sb


_NUMERIC_TYPES = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
    pytest.param(sb.IntTensor, np.int32, id="int32"),
    pytest.param(sb.IntTensor, np.int64, id="int64"),
]


def _assert_reshape(np_arr, new_shape, TensorType):
    """Assert that stablebear reshape matches NumPy."""
    t = TensorType(np_arr)
    result = np.asarray(t.reshape(new_shape))
    expected = np_arr.reshape(new_shape)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


def test_tensor2d_flatten():
    np_arr = np.array([[0, 1, 2], [10, 11, 12]], dtype=np.float32)
    t = sb.FloatTensor(np_arr)
    np_flat = np_arr.flatten()
    flat = t.flatten()
    assert flat.shape == np_flat.shape
    for i in range(flat.shape[0]):
        assert flat[i] == np_flat[i]


def test_flatten_numpy_view_is_row_major():
    """flatten() must expose row-major data through numpy/buffer, not stride-0.

    Regression for #15: flatten() used to set the flattened-axis stride to 0,
    leaving m_isContiguous=true, so np.asarray()/print() read element 0 repeated
    while element indexing (a different C++ path) stayed correct.
    """
    t = sb.FloatTensor(np.arange(6, dtype=np.float64).reshape(2, 3))
    f = t.flatten()

    expected = np.arange(6, dtype=np.float64).reshape(2, 3).flatten()

    # The numpy/buffer view must match the canonical row-major flatten...
    assert np.array_equal(np.asarray(f), expected)
    # ...and must agree with the element-access path (which is already correct).
    assert [f[i] for i in range(6)] == expected.tolist()
    # The contiguous flattened axis must have stride 1, never 0.
    assert tuple(f.strides) == (1,)


def test_flatten_noncontiguous_slice_numpy_view():
    """flatten() of a non-contiguous slice copies, then exposes a correct view."""
    big = sb.FloatTensor(np.arange(12, dtype=np.float64).reshape(3, 4))
    f = big[:, 1:3].flatten()
    expected = np.arange(12, dtype=np.float64).reshape(3, 4)[:, 1:3].flatten()
    assert np.array_equal(np.asarray(f), expected)
    assert tuple(f.strides) == (1,)


def test_flatten_expand_dims_numpy_view():
    """expand_dims after flatten must still produce a correct numpy view."""
    t = sb.FloatTensor(np.arange(6, dtype=np.float64).reshape(2, 3))
    f = t.flatten().expand_dims(0)
    expected = np.arange(6, dtype=np.float64).reshape(2, 3).flatten()[np.newaxis, :]
    assert np.array_equal(np.asarray(f), expected)


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestReshape:
    def test_2d_to_1d(self, TensorType, np_dtype):
        _assert_reshape(np.arange(12, dtype=np_dtype).reshape(3, 4), (12,), TensorType)

    def test_1d_to_2d(self, TensorType, np_dtype):
        _assert_reshape(np.arange(12, dtype=np_dtype), (3, 4), TensorType)

    def test_2d_to_3d(self, TensorType, np_dtype):
        _assert_reshape(np.arange(24, dtype=np_dtype).reshape(4, 6), (2, 3, 4), TensorType)

    def test_infer_dimension(self, TensorType, np_dtype):
        _assert_reshape(np.arange(24, dtype=np_dtype), (6, -1), TensorType)

    def test_infer_first_dimension(self, TensorType, np_dtype):
        _assert_reshape(np.arange(24, dtype=np_dtype), (-1, 4), TensorType)

    def test_same_shape(self, TensorType, np_dtype):
        _assert_reshape(np.arange(12, dtype=np_dtype).reshape(3, 4), (3, 4), TensorType)

    def test_non_contiguous(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype).reshape(3, 4)
        t = TensorType(np_arr)
        view = t[::2, :]  # non-contiguous
        _assert_reshape(np.asarray(view), (8,), TensorType)

    def test_contiguous_is_view(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype)
        t = TensorType(np_arr)
        reshaped = t.reshape((3, 4))
        reshaped[0, 0] = 99
        assert t[0] == 99

    def test_non_contiguous_is_copy(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype).reshape(3, 4)
        t = TensorType(np_arr)
        view = t[::2, :]
        reshaped = view.reshape((8,))
        reshaped[0] = 99
        assert t[0, 0] == 0

    def test_incompatible_shape_raises(self, TensorType, np_dtype):
        t = TensorType(np.arange(12, dtype=np_dtype))
        with pytest.raises(ValueError):
            t.reshape((5, 5))

    def test_multiple_infer_raises(self, TensorType, np_dtype):
        t = TensorType(np.arange(12, dtype=np_dtype))
        with pytest.raises(ValueError):
            t.reshape((-1, -1))


# --- transpose ---


def _assert_transpose(np_arr, axes, TensorType):
    """Assert that stablebear transpose matches NumPy."""
    t = TensorType(np_arr)
    if axes is None:
        result = np.asarray(t.T)
        expected = np_arr.T
    else:
        result = np.asarray(t.transpose(axes))
        expected = np_arr.transpose(axes)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestTranspose:
    def test_2d_T(self, TensorType, np_dtype):
        _assert_transpose(np.arange(12, dtype=np_dtype).reshape(3, 4), None, TensorType)

    def test_2d_transpose_axes(self, TensorType, np_dtype):
        _assert_transpose(np.arange(12, dtype=np_dtype).reshape(3, 4), (1, 0), TensorType)

    def test_3d_transpose_default(self, TensorType, np_dtype):
        _assert_transpose(np.arange(24, dtype=np_dtype).reshape(2, 3, 4), None, TensorType)

    def test_3d_transpose_axes(self, TensorType, np_dtype):
        _assert_transpose(np.arange(24, dtype=np_dtype).reshape(2, 3, 4), (2, 0, 1), TensorType)

    def test_3d_transpose_negative_axes(self, TensorType, np_dtype):
        _assert_transpose(
            np.arange(24, dtype=np_dtype).reshape(2, 3, 4),
            (-1, -2, -3),
            TensorType,
        )

    def test_3d_transpose_mixed_negative_axes(self, TensorType, np_dtype):
        _assert_transpose(
            np.arange(24, dtype=np_dtype).reshape(2, 3, 4),
            (2, 1, -3),
            TensorType,
        )

    def test_1d_T_is_noop(self, TensorType, np_dtype):
        np_arr = np.arange(5, dtype=np_dtype)
        t = TensorType(np_arr)
        np.testing.assert_array_equal(np.asarray(t.T), np_arr)

    def test_transpose_is_view(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype).reshape(3, 4)
        t = TensorType(np_arr)
        tr = t.T
        tr[0, 0] = 99
        assert t[0, 0] == 99

    def test_transpose_then_slice(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype).reshape(3, 4)
        t = TensorType(np_arr)
        result = np.asarray(t.T[1:3, :])
        expected = np_arr.T[1:3, :]
        np.testing.assert_array_equal(result, expected)

    def test_wrong_number_of_axes_raises(self, TensorType, np_dtype):
        t = TensorType(np.arange(12, dtype=np_dtype).reshape(3, 4))
        with pytest.raises((ValueError, RuntimeError)):
            t.transpose((0,))

    def test_repeated_axis_raises(self, TensorType, np_dtype):
        t = TensorType(np.arange(24, dtype=np_dtype).reshape(2, 3, 4))
        with pytest.raises((ValueError, RuntimeError)):
            t.transpose((2, 2, 0))


# --- swapaxes ---


def _assert_swapaxes(np_arr, axis1, axis2, TensorType):
    """Assert that stablebear swapaxes matches NumPy."""
    t = TensorType(np_arr)
    result = np.asarray(t.swapaxes(axis1, axis2))
    expected = np_arr.swapaxes(axis1, axis2)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestSwapaxes:
    def test_2d(self, TensorType, np_dtype):
        _assert_swapaxes(np.arange(12, dtype=np_dtype).reshape(3, 4), 0, 1, TensorType)

    def test_3d_01(self, TensorType, np_dtype):
        _assert_swapaxes(np.arange(24, dtype=np_dtype).reshape(2, 3, 4), 0, 1, TensorType)

    def test_3d_02(self, TensorType, np_dtype):
        _assert_swapaxes(np.arange(24, dtype=np_dtype).reshape(2, 3, 4), 0, 2, TensorType)

    def test_3d_12(self, TensorType, np_dtype):
        _assert_swapaxes(np.arange(24, dtype=np_dtype).reshape(2, 3, 4), 1, 2, TensorType)

    def test_same_axis(self, TensorType, np_dtype):
        _assert_swapaxes(np.arange(12, dtype=np_dtype).reshape(3, 4), 0, 0, TensorType)

    def test_negative_axes(self, TensorType, np_dtype):
        _assert_swapaxes(np.arange(24, dtype=np_dtype).reshape(2, 3, 4), -1, -3, TensorType)

    def test_is_view(self, TensorType, np_dtype):
        np_arr = np.arange(12, dtype=np_dtype).reshape(3, 4)
        t = TensorType(np_arr)
        swapped = t.swapaxes(0, 1)
        swapped[0, 0] = 99
        assert t[0, 0] == 99

    def test_out_of_range_raises(self, TensorType, np_dtype):
        t = TensorType(np.arange(12, dtype=np_dtype).reshape(3, 4))
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            t.swapaxes(0, 3)

    def test_negative_out_of_range_raises(self, TensorType, np_dtype):
        t = TensorType(np.arange(12, dtype=np_dtype).reshape(3, 4))
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            t.swapaxes(-3, 0)


# --- squeeze ---


def _assert_squeeze(np_arr, axis, TensorType):
    """Assert that stablebear squeeze matches NumPy."""
    t = TensorType(np_arr)
    if axis is None:
        result = np.asarray(t.squeeze())
        expected = np_arr.squeeze()
    else:
        result = np.asarray(t.squeeze(axis))
        expected = np_arr.squeeze(axis=axis)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestSqueeze:
    def test_squeeze_all(self, TensorType, np_dtype):
        _assert_squeeze(np.arange(6, dtype=np_dtype).reshape(1, 6, 1), None, TensorType)

    def test_squeeze_specific_axis(self, TensorType, np_dtype):
        _assert_squeeze(np.arange(6, dtype=np_dtype).reshape(1, 6, 1), 0, TensorType)

    def test_squeeze_last_axis(self, TensorType, np_dtype):
        _assert_squeeze(np.arange(6, dtype=np_dtype).reshape(1, 6, 1), 2, TensorType)

    def test_squeeze_negative_axes(self, TensorType, np_dtype):
        _assert_squeeze(np.arange(6, dtype=np_dtype).reshape(1, 6, 1), -1, TensorType)
        _assert_squeeze(np.arange(6, dtype=np_dtype).reshape(1, 6, 1), -3, TensorType)

    def test_squeeze_no_size1_dims(self, TensorType, np_dtype):
        _assert_squeeze(np.arange(12, dtype=np_dtype).reshape(3, 4), None, TensorType)

    def test_squeeze_is_view(self, TensorType, np_dtype):
        np_arr = np.arange(6, dtype=np_dtype).reshape(1, 6)
        t = TensorType(np_arr)
        sq = t.squeeze()
        sq[0] = 99
        assert t[0, 0] == 99

    def test_squeeze_non_size1_raises(self, TensorType, np_dtype):
        t = TensorType(np.arange(12, dtype=np_dtype).reshape(3, 4))
        with pytest.raises((ValueError, RuntimeError)):
            t.squeeze(0)


# --- expand_dims ---


def _assert_expand_dims(np_arr, axis, TensorType):
    """Assert that stablebear expand_dims matches NumPy."""
    t = TensorType(np_arr)
    result = np.asarray(t.expand_dims(axis))
    expected = np.expand_dims(np_arr, axis)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestExpandDims:
    def test_expand_first(self, TensorType, np_dtype):
        _assert_expand_dims(np.arange(6, dtype=np_dtype), 0, TensorType)

    def test_expand_last(self, TensorType, np_dtype):
        _assert_expand_dims(np.arange(6, dtype=np_dtype), -1, TensorType)

    def test_expand_middle(self, TensorType, np_dtype):
        _assert_expand_dims(np.arange(12, dtype=np_dtype).reshape(3, 4), 1, TensorType)

    def test_expand_2d_first(self, TensorType, np_dtype):
        _assert_expand_dims(np.arange(12, dtype=np_dtype).reshape(3, 4), 0, TensorType)

    def test_expand_2d_last(self, TensorType, np_dtype):
        _assert_expand_dims(np.arange(12, dtype=np_dtype).reshape(3, 4), 2, TensorType)

    def test_expand_negative_axis(self, TensorType, np_dtype):
        _assert_expand_dims(np.arange(12, dtype=np_dtype).reshape(3, 4), -2, TensorType)

    def test_expand_negative_axis_before_first(self, TensorType, np_dtype):
        _assert_expand_dims(np.arange(12, dtype=np_dtype).reshape(3, 4), -3, TensorType)

    def test_expand_is_view(self, TensorType, np_dtype):
        np_arr = np.arange(6, dtype=np_dtype)
        t = TensorType(np_arr)
        expanded = t.expand_dims(0)
        expanded[0, 0] = 99
        assert t[0] == 99

    def test_expand_out_of_range_raises(self, TensorType, np_dtype):
        t = TensorType(np.arange(6, dtype=np_dtype))
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            t.expand_dims(3)
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            t.expand_dims(-3)


# --- astype ---


def _assert_astype(tensor, target_dtype, expected_type):
    """Assert that astype produces the right type/dtype and preserves shape."""
    result = tensor.astype(target_dtype)
    assert isinstance(result, expected_type)
    assert result.dtype == target_dtype
    assert result.shape == tensor.shape


# Numeric precision changes
_NUMERIC_ASTYPE_CASES = [
    pytest.param(sb.float32, sb.float64, sb.FloatTensor, id="f32→f64"),
    pytest.param(sb.float64, sb.float32, sb.FloatTensor, id="f64→f32"),
    pytest.param(sb.int32, sb.int64, sb.IntTensor, id="i32→i64"),
    pytest.param(sb.int64, sb.int32, sb.IntTensor, id="i64→i32"),
    pytest.param(sb.uint32, sb.uint64, sb.IntTensor, id="u32→u64"),
    pytest.param(sb.int32, sb.uint32, sb.IntTensor, id="i32→u32"),
]

# Cross-family numeric changes
_CROSS_FAMILY_CASES = [
    pytest.param(sb.float64, sb.int32, sb.IntTensor, id="f64→i32"),
    pytest.param(sb.int32, sb.float64, sb.FloatTensor, id="i32→f64"),
]


@pytest.mark.parametrize("src_dtype, target_dtype, expected_type", _NUMERIC_ASTYPE_CASES + _CROSS_FAMILY_CASES)
def test_astype_numeric(src_dtype, target_dtype, expected_type):
    t = sb.zeros((3, 4), dtype=src_dtype)
    _assert_astype(t, target_dtype, expected_type)


# PCF precision changes
_PCF_ASTYPE_CASES = [
    pytest.param(sb.pcf32, sb.pcf64, sb.PcfTensor, id="pcf32→pcf64"),
    pytest.param(sb.pcf64, sb.pcf32, sb.PcfTensor, id="pcf64→pcf32"),
    pytest.param(sb.pcf32i, sb.pcf64i, sb.IntPcfTensor, id="pcf32i→pcf64i"),
    pytest.param(sb.pcf64i, sb.pcf32i, sb.IntPcfTensor, id="pcf64i→pcf32i"),
]


@pytest.mark.parametrize("src_dtype, target_dtype, expected_type", _PCF_ASTYPE_CASES)
def test_astype_pcf(src_dtype, target_dtype, expected_type):
    t = sb.zeros((3,), dtype=src_dtype)
    _assert_astype(t, target_dtype, expected_type)


# Point cloud precision changes
def test_astype_pcloud32_to_64():
    t = sb.zeros((3,), dtype=sb.pcloud32)
    _assert_astype(t, sb.pcloud64, sb.PointCloudTensor)


def test_astype_pcloud64_to_32():
    t = sb.zeros((3,), dtype=sb.pcloud64)
    _assert_astype(t, sb.pcloud32, sb.PointCloudTensor)


# Same dtype always copies
def test_astype_same_dtype_copies():
    arr = np.array([1.0, 2.0, 3.0])
    t = sb.FloatTensor(arr)
    t2 = t.astype(sb.float64)
    t2[0] = 99.0
    assert t[0] == 1.0


# Cross-family errors
def test_astype_float_to_pcf_raises():
    t = sb.FloatTensor(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises((ValueError, TypeError)):
        t.astype(sb.pcf32)
