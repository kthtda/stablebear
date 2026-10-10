import io
import pickle

import numpy as np
import pytest

import stablebear as sb


def _roundtrip(tensor):
    buf = io.BytesIO()
    sb.save(tensor, buf)
    buf.seek(0)
    return sb.load(buf)


def _assert_roundtrip(original):
    restored = _roundtrip(original)
    assert type(restored) is type(original)
    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    assert original.array_equal(restored)


def _pickle_roundtrip(value):
    return pickle.loads(pickle.dumps(value))


_NESTED_SCALAR_DTYPES = [
    pytest.param(sb.float32, np.float32, id="float32"),
    pytest.param(sb.float64, np.float64, id="float64"),
    pytest.param(sb.int32, np.int32, id="int32"),
    pytest.param(sb.int64, np.int64, id="int64"),
    pytest.param(sb.uint32, np.uint32, id="uint32"),
    pytest.param(sb.uint64, np.uint64, id="uint64"),
]


@pytest.mark.parametrize("roundtrip", [_roundtrip, _pickle_roundtrip])
@pytest.mark.parametrize("sb_dtype,np_dtype", _NESTED_SCALAR_DTYPES)
def test_nonzero_scalar_value_survives_at_every_nesting_level(
    sb_dtype, np_dtype, roundtrip
):
    scalar = np.array(42, dtype=np_dtype)

    ordinary = roundtrip(sb.tensor(scalar, dtype=sb_dtype))
    assert np.asarray(ordinary)[()] == 42

    depth_three = sb.NestedTensor([
        sb.NestedTensor([scalar], dtype=sb_dtype)
    ])
    restored = roundtrip(depth_three)
    assert restored.depth == 3
    assert restored[0][0][()] == 42


@pytest.mark.parametrize("roundtrip", [_roundtrip, _pickle_roundtrip])
def test_true_scalar_bool_survives_roundtrip(roundtrip):
    restored = roundtrip(sb.tensor(np.array(True), dtype=sb.boolean))
    assert np.asarray(restored)[()]


@pytest.mark.parametrize("roundtrip", [_roundtrip, _pickle_roundtrip])
@pytest.mark.parametrize(
    "pcloud_dtype,np_dtype",
    [
        pytest.param(sb.pcloud32, np.float32, id="pcloud32"),
        pytest.param(sb.pcloud64, np.float64, id="pcloud64"),
    ],
)
def test_scalar_point_cloud_tensor_preserves_cloud(
    pcloud_dtype, np_dtype, roundtrip
):
    coordinates = np.array([[1, 2], [3, 4]], dtype=np_dtype)
    restored = roundtrip(sb.PointCloudTensor(coordinates, dtype=pcloud_dtype))
    assert restored.shape == ()
    np.testing.assert_array_equal(np.asarray(restored[()]), coordinates)


# --- Float tensors ---


def test_float32_tensor_roundtrip():
    _assert_roundtrip(sb.FloatTensor(np.array([[1.0, 2.5], [3.0, -4.5]], dtype=np.float32)))


def test_float64_tensor_roundtrip():
    _assert_roundtrip(sb.FloatTensor(np.array([[1.0, 2.5], [3.0, -4.5]], dtype=np.float64)))


def test_float32_tensor_1d():
    _assert_roundtrip(sb.FloatTensor(np.array([1.5, 2.5, 3.5], dtype=np.float32)))


def test_float64_tensor_3d():
    _assert_roundtrip(sb.FloatTensor(np.random.randn(2, 3, 4).astype(np.float64)))


def test_float32_tensor_scalar():
    _assert_roundtrip(sb.FloatTensor(np.array([42.0], dtype=np.float32)))


# --- PCF tensors ---


def test_pcf32_tensor_roundtrip():
    _assert_roundtrip(sb.random.noisy_sin((3, 4), dtype=sb.pcf32))


def test_pcf64_tensor_roundtrip():
    _assert_roundtrip(sb.random.noisy_sin((3, 4), dtype=sb.pcf64))


def test_pcf32_tensor_1d():
    _assert_roundtrip(sb.random.noisy_cos((5,), dtype=sb.pcf32))


def test_pcf32_tensor_zeros():
    _assert_roundtrip(sb.zeros((2, 3), dtype=sb.pcf32))


def test_pcf64_tensor_zeros():
    _assert_roundtrip(sb.zeros((2, 3), dtype=sb.pcf64))


# --- PointCloud tensors ---


def test_point_cloud32_tensor_roundtrip():
    original = sb.zeros((3,), dtype=sb.pcloud32)
    original[0] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    original[1] = np.array([[5.0, 6.0]], dtype=np.float32)
    _assert_roundtrip(original)


def test_point_cloud64_tensor_roundtrip():
    original = sb.zeros((3,), dtype=sb.pcloud64)
    original[0] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    original[1] = np.array([[5.0, 6.0]], dtype=np.float64)
    _assert_roundtrip(original)


def test_point_cloud32_tensor_empty():
    _assert_roundtrip(sb.zeros((2,), dtype=sb.pcloud32))


# --- Barcode tensors ---


def test_barcode32_tensor_roundtrip():
    original = sb.zeros((2,), dtype=sb.barcode32)
    original[0] = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float32)
    _assert_roundtrip(original)


def test_barcode64_tensor_roundtrip():
    original = sb.zeros((2,), dtype=sb.barcode64)
    original[0] = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64)
    _assert_roundtrip(original)


def test_barcode32_tensor_empty():
    _assert_roundtrip(sb.zeros((3,), dtype=sb.barcode32))
