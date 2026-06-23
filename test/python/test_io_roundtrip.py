import io

import numpy as np

import stablebear as sb
from stablebear.sampling import Gaussian, subsample


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


# --- Indexed point clouds (subsamples) ---


def test_indexed_subsamples_roundtrip():
    R = np.random.default_rng(0).standard_normal((150, 6))
    X = np.random.default_rng(1).standard_normal((3, 6))
    subs = subsample(R, X, sample_size=12, n_instances=7,
                     distribution=Gaussian(0.0, 1.0), generator=sb.random.Generator(0))

    before = [[np.asarray(subs[i, j]) for j in range(7)] for i in range(3)]

    loaded = _roundtrip(subs)

    assert isinstance(loaded, sb.PointCloudTensor)
    assert loaded.shape == (3, 7)
    # Indexed structure survives the round trip (sources shared, not re-stored).
    assert loaded[0, 0].is_indexed
    for i in range(3):
        for j in range(7):
            assert np.array_equal(np.asarray(loaded[i, j]), before[i][j])
