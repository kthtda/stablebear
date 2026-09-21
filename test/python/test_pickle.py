import io
import pickle

import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import Barcode, BarcodeTensor


# --- Helpers ---


def _pickle_roundtrip(obj):
    reducer, args = obj.__reduce__()
    assert reducer.__module__ == "stablebear.io"
    assert reducer.__name__ == "_unpickle"
    assert len(args) == 1
    data = pickle.dumps(obj)
    return pickle.loads(data)


def _assert_tensor_roundtrip(t):
    restored = _pickle_roundtrip(t)
    assert type(restored) is type(t)
    assert restored.dtype == t.dtype
    assert restored.shape == t.shape
    assert t.array_equal(restored)


# --- Float tensors ---


def test_pickle_float32_tensor():
    _assert_tensor_roundtrip(
        sb.FloatTensor(np.array([1.0, 2.0, 3.0], dtype=np.float32)))


def test_pickle_float64_tensor():
    _assert_tensor_roundtrip(
        sb.FloatTensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)))


def test_pickle_float_tensor_3d():
    _assert_tensor_roundtrip(
        sb.FloatTensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4)))


# --- Int tensors ---


def test_pickle_int32_tensor():
    _assert_tensor_roundtrip(
        sb.IntTensor(np.array([10, 20, 30], dtype=np.int32)))


def test_pickle_int64_tensor():
    _assert_tensor_roundtrip(
        sb.IntTensor(np.array([[1, 2], [3, 4]], dtype=np.int64)))


def test_pickle_unsigned_integer_tensors():
    _assert_tensor_roundtrip(
        sb.IntTensor(np.array([1, 2, 3], dtype=np.uint32)))
    _assert_tensor_roundtrip(
        sb.IntTensor(np.array([4, 5, 6], dtype=np.uint64)))


# --- Bool tensor ---


def test_pickle_bool_tensor():
    _assert_tensor_roundtrip(
        sb.BoolTensor(np.array([[True, False], [False, True]])))


# --- PCF tensors ---


def test_pickle_pcf32_tensor():
    f = sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32))
    g = sb.Pcf(np.array([[0.0, 3.0], [2.0, 4.0]], dtype=np.float32))
    _assert_tensor_roundtrip(sb.PcfTensor([f, g]))


def test_pickle_pcf64_tensor():
    f = sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64))
    _assert_tensor_roundtrip(sb.PcfTensor([f]))


def test_pickle_pcf_tensor_2d():
    fs = [sb.Pcf(np.array([[0, float(i)], [1, float(i + 1)]], dtype=np.float32))
          for i in range(6)]
    t = sb.PcfTensor(fs).reshape((2, 3))
    _assert_tensor_roundtrip(t)


def test_pickle_integer_pcf_tensors():
    _assert_tensor_roundtrip(sb.IntPcfTensor([
        sb.Pcf(np.array([[0, 1], [2, 3]], dtype=np.int32))
    ]))
    _assert_tensor_roundtrip(sb.IntPcfTensor([
        sb.Pcf(np.array([[0, 1], [2, 3]], dtype=np.int64))
    ]))


def test_pickle_nested_tensors_all_leaf_dtypes():
    cases = (
        (sb.float32, np.float32),
        (sb.float64, np.float64),
        (sb.int32, np.int32),
        (sb.int64, np.int64),
        (sb.uint32, np.uint32),
        (sb.uint64, np.uint64),
    )
    for sb_dtype, np_dtype in cases:
        nested = sb.NestedTensor([
            sb.tensor(np.array([1, 2], dtype=np_dtype), dtype=sb_dtype),
            sb.tensor(np.array([3], dtype=np_dtype), dtype=sb_dtype),
        ])
        _assert_tensor_roundtrip(nested)


def test_pickle_remaining_tensor_families():
    for np_dtype in (np.float32, np.float64):
        _assert_tensor_roundtrip(sb.PointCloudTensor(np.array(
            [[[[1, 2], [3, 4]]]], dtype=np_dtype
        )))
        _assert_tensor_roundtrip(BarcodeTensor([
            Barcode(np.array([[0, 1], [0.5, 2]], dtype=np_dtype))
        ]))

        symmetric = np.array([[[1, 2], [2, 3]]], dtype=np_dtype)
        _assert_tensor_roundtrip(sb.SymmetricMatrixTensor(symmetric))
        distance = np.array([[[0, 2], [2, 0]]], dtype=np_dtype)
        _assert_tensor_roundtrip(sb.DistanceMatrixTensor(distance))


def test_pickle_supported_protocols_embed_binary_payload():
    value = sb.FloatTensor(np.array([1, 2], dtype=np.float32))
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(value, protocol=protocol))
        assert restored.array_equal(value)


# --- Standalone Pcf ---


def test_pickle_pcf_f32():
    f = sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.5]], dtype=np.float32))
    restored = _pickle_roundtrip(f)
    assert isinstance(restored, sb.Pcf)
    assert f == restored


def test_pickle_pcf_f64():
    f = sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64))
    restored = _pickle_roundtrip(f)
    assert isinstance(restored, sb.Pcf)
    assert f == restored


def test_pickle_pcf_i32():
    f = sb.Pcf(np.array([[0, 1], [2, 3]], dtype=np.int32))
    restored = _pickle_roundtrip(f)
    assert isinstance(restored, sb.Pcf)
    assert f == restored


def test_pickle_pcf_i64():
    f = sb.Pcf(np.array([[0, 1], [2, 3]], dtype=np.int64))
    restored = _pickle_roundtrip(f)
    assert isinstance(restored, sb.Pcf)
    assert f == restored


# --- Barcode ---


def test_pickle_barcode_f64():
    bc = Barcode(np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64))
    restored = _pickle_roundtrip(bc)
    assert isinstance(restored, Barcode)
    assert np.array_equal(np.asarray(bc), np.asarray(restored))


def test_pickle_barcode_f32():
    bc = Barcode(np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float32))
    restored = _pickle_roundtrip(bc)
    assert isinstance(restored, Barcode)
    assert np.array_equal(np.asarray(bc), np.asarray(restored))


# --- DistanceMatrix ---


def test_pickle_distance_matrix():
    dm = sb.DistanceMatrix(3, dtype=sb.float64)
    dm[0, 1] = 1.0
    dm[0, 2] = 2.0
    dm[1, 2] = 3.0
    restored = _pickle_roundtrip(dm)
    assert isinstance(restored, sb.DistanceMatrix)
    assert restored.size == dm.size
    assert restored[0, 1] == 1.0
    assert restored[0, 2] == 2.0
    assert restored[1, 2] == 3.0


# --- SymmetricMatrix ---


def test_pickle_symmetric_matrix():
    sm = sb.SymmetricMatrix(3, dtype=sb.float64)
    sm[0, 0] = 1.0
    sm[0, 1] = 2.0
    sm[1, 1] = 3.0
    restored = _pickle_roundtrip(sm)
    assert isinstance(restored, sb.SymmetricMatrix)
    assert restored.size == sm.size
    assert restored[0, 0] == 1.0
    assert restored[0, 1] == 2.0
    assert restored[1, 1] == 3.0


# --- Standalone PointCloud ---


def test_pickle_and_public_io_point_cloud_both_precisions():
    for np_dtype, sb_dtype in ((np.float32, sb.float32), (np.float64, sb.float64)):
        cloud = sb.PointCloud(np.array([[1, 2], [3, 4]], dtype=np_dtype))
        restored = _pickle_roundtrip(cloud)
        assert type(restored) is sb.PointCloud
        assert restored.dtype is sb_dtype
        assert np.array_equal(np.asarray(restored), np.asarray(cloud))

        binary = io.BytesIO()
        sb.save(cloud, binary)
        restored = sb.load(io.BytesIO(binary.getvalue()))
        assert type(restored) is sb.PointCloud
        assert restored.dtype is sb_dtype
        assert np.array_equal(np.asarray(restored), np.asarray(cloud))


@pytest.mark.parametrize(
    ("np_dtype", "sb_dtype"),
    ((np.float32, sb.float32), (np.float64, sb.float64)),
)
def test_owner_backed_point_cloud_pickle_is_standalone_logical_value(
    np_dtype, sb_dtype
):
    owner = sb.PointCloudTensor(np.array([
        [[1, 2], [3, 4]],
        [[90, 80], [70, 60]],
    ], dtype=np_dtype))
    restored = _pickle_roundtrip(owner[0])
    del owner

    assert restored._owner is None
    assert restored.dtype is sb_dtype
    assert np.array_equal(
        np.asarray(restored), np.array([[1, 2], [3, 4]], dtype=np_dtype)
    )


def test_binary_magic_prevents_corrupt_payload_from_using_legacy_fallback():
    from stablebear.io import _BINARY_MAGIC, _unpickle

    fallback_calls = []

    def fallback(data):
        fallback_calls.append(data)
        return "legacy"

    assert _unpickle(b"legacy bytes", fallback) == "legacy"
    with pytest.raises(RuntimeError):
        _unpickle(_BINARY_MAGIC + b"corrupt", fallback)
    assert fallback_calls == [b"legacy bytes"]


def test_pickle_inventory_non_data_objects():
    # dtype is immutable singleton metadata and deliberately retains its
    # global-name reduction. Generator and Rectangle wrap non-pickleable
    # execution/backend state and remain explicitly unsupported.
    assert pickle.loads(pickle.dumps(sb.float32)) is sb.float32
    with pytest.raises(TypeError):
        pickle.dumps(sb.random.Generator(1))
    f = sb.Pcf(np.array([[0.0, 1.0]], dtype=np.float64))
    rectangle = sb.iterate_rectangles(f, f)[0]
    with pytest.raises(TypeError):
        pickle.dumps(rectangle)


def test_binary_io_mixin_requires_explicit_backend_data_hook():
    from stablebear._binary_io import _BinaryIoMixin

    class IncompleteBinaryObject(_BinaryIoMixin):
        pass

    with pytest.raises(TypeError, match="abstract method _binary_io_data"):
        IncompleteBinaryObject()
