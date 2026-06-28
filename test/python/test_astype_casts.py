"""astype cast-matrix completion (#50, #98, #99, #95, #114).

Covers float<->uint, bool<->numeric, distmat/symmat/barcode precision casts.
"""
import numpy as np
import pytest

import stablebear as sb
import stablebear.persistence as mp


# ---------------------------------------------------------------------------
# #50 / #98: float <-> uint casts (and the cross-dtype assignment they enable).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("uint_dtype, np_uint", [(sb.uint32, np.uint32),
                                                 (sb.uint64, np.uint64)])
@pytest.mark.parametrize("float_dtype, np_float", [(sb.float32, np.float32),
                                                   (sb.float64, np.float64)])
def test_uint_to_float_and_back(uint_dtype, np_uint, float_dtype, np_float):
    u = sb.IntTensor(np.array([0, 1, 7, 255], dtype=np_uint))
    f = u.astype(float_dtype)
    assert f.dtype is float_dtype
    np.testing.assert_array_equal(
        np.asarray(f), np.array([0, 1, 7, 255], dtype=np_uint).astype(np_float))

    back = sb.FloatTensor(np.array([0.0, 1.9, 7.2], dtype=np_float)).astype(uint_dtype)
    assert back.dtype is uint_dtype
    np.testing.assert_array_equal(
        np.asarray(back), np.array([0.0, 1.9, 7.2], dtype=np_float).astype(np_uint))


def test_uint_into_float_slice_assignment():
    # Bug #50: cross-dtype assignment routed through astype, which lacked the
    # float<->uint cast and raised "Cannot cast from uint32 to float64".
    f = sb.FloatTensor(np.zeros(3, dtype=np.float64))
    u = sb.IntTensor(np.array([1, 2, 3], dtype=np.uint32))
    f[:] = u
    np.testing.assert_array_equal(np.asarray(f), np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# #99: bool <-> numeric casts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_dtype, np_num", [
    (sb.int32, np.int32), (sb.int64, np.int64),
    (sb.uint32, np.uint32), (sb.uint64, np.uint64),
    (sb.float32, np.float32), (sb.float64, np.float64),
])
def test_bool_to_numeric(num_dtype, np_num):
    b = sb.BoolTensor([True, False, True])
    out = b.astype(num_dtype)
    assert out.dtype is num_dtype
    np.testing.assert_array_equal(np.asarray(out), np.array([1, 0, 1], dtype=np_num))


@pytest.mark.parametrize("ctor, data", [
    (sb.IntTensor, np.array([0, 2, 0, -1], dtype=np.int64)),
    (sb.FloatTensor, np.array([0.0, 1.5, 0.0, -3.0], dtype=np.float64)),
])
def test_numeric_to_bool(ctor, data):
    out = ctor(data).astype(sb.boolean)
    assert out.dtype is sb.boolean
    np.testing.assert_array_equal(np.asarray(out), data.astype(bool))


# ---------------------------------------------------------------------------
# #95: DistanceMatrix / SymmetricMatrix precision casts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "TensorCls, d32, d64, symmetric_only",
    [
        (sb.DistanceMatrixTensor, sb.distmat32, sb.distmat64, False),
        (sb.SymmetricMatrixTensor, sb.symmat32, sb.symmat64, True),
    ],
)
def test_matrix_precision_roundtrip(TensorCls, d32, d64, symmetric_only):
    base = np.array([[[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]])
    t64 = TensorCls.from_numpy(base)
    t32 = t64.astype(d32)
    assert t32.dtype is d32
    np.testing.assert_allclose(t32[0].to_dense(), base[0], rtol=1e-6)
    t64b = t32.astype(d64)
    assert t64b.dtype is d64
    np.testing.assert_allclose(t64b[0].to_dense(), base[0])


# ---------------------------------------------------------------------------
# #114: barcode precision casts.
# ---------------------------------------------------------------------------


def test_barcode_precision_roundtrip():
    pc = sb.zeros((2,), dtype=sb.pcloud64)
    pc[0] = np.random.RandomState(0).randn(6, 2)
    pc[1] = np.random.RandomState(1).randn(6, 2)
    bt = mp.compute_persistent_homology(pc, max_dim=1)
    assert bt.dtype is sb.barcode64

    bt32 = bt.astype(sb.barcode32)
    assert bt32.dtype is sb.barcode32
    assert tuple(bt32.shape) == tuple(bt.shape)
    np.testing.assert_allclose(
        bt[0, 0].to_numpy(), bt32[0, 0].to_numpy(), rtol=1e-5)

    bt64 = bt32.astype(sb.barcode64)
    assert bt64.dtype is sb.barcode64


def test_astype_to_same_dtype_is_copy():
    t = sb.FloatTensor([1.0, 2.0, 3.0])
    assert t.astype(sb.float64).dtype is sb.float64


# ---------------------------------------------------------------------------
# Genuinely undefined casts still raise a clear TypeError.
#
# astype builds a C++ binding name cast_{src_tag}_{dst_tag}; when no such
# binding exists it raises TypeError (_tensor_base.py:796-798). Casts between
# unrelated families were never registered, so they remain undefined.
# ---------------------------------------------------------------------------


def test_numeric_to_pcf_cast_is_undefined():
    t = sb.FloatTensor(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    with pytest.raises(TypeError):
        t.astype(sb.pcf32)


def test_distmat_to_symmat_cast_is_undefined():
    base = np.array([[[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]])
    t = sb.DistanceMatrixTensor.from_numpy(base)
    with pytest.raises(TypeError):
        t.astype(sb.symmat32)
