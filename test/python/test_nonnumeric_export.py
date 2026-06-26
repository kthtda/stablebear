"""Tests for exporting the non-numeric tensor families back to NumPy / dense
arrays (issues #85, #93, #94). Previously the only way to read these tensors
out was a per-element loop; now they expose ``to_dense``/``to_numpy`` (and work
with ``np.asarray``), and ``BarcodeTensor`` exposes ``to_numpy``/``tolist``."""

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb
import stablebear.persistence as mp
from stablebear.persistence import Barcode


def _symmetric_zero_diag_batch(N, n, seed=0):
    rng = np.random.RandomState(seed)
    out = np.zeros((N, n, n))
    for k in range(N):
        m = rng.rand(n, n)
        m = (m + m.T) / 2.0
        np.fill_diagonal(m, 0.0)
        out[k] = m
    return out


def _symmetric_batch(shape, n, seed=0):
    rng = np.random.RandomState(seed)
    out = np.zeros(shape + (n, n))
    for idx in np.ndindex(*shape):
        m = rng.rand(n, n)
        out[idx] = (m + m.T) / 2.0
    return out


# --- PointCloudTensor (#93) ---


@pytest.mark.parametrize("shape", [(3, 5, 2), (2, 3, 4, 2), (6, 4, 2)])
def test_pointcloud_to_dense_roundtrip(shape):
    arr = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    pc = sb.PointCloudTensor(arr)
    dense = pc.to_dense()
    assert dense.shape == arr.shape
    assert dense.dtype != object
    npt.assert_allclose(dense, arr)
    # to_numpy is an alias
    npt.assert_allclose(pc.to_numpy(), dense)


@pytest.mark.parametrize("convert", [np.asarray, np.array])
def test_pointcloud_asarray_is_dense(convert):
    arr = np.random.RandomState(0).rand(3, 5, 2)
    pc = sb.PointCloudTensor(arr)
    out = convert(pc)
    assert out.dtype != object
    assert out.shape == (3, 5, 2)
    npt.assert_allclose(out, arr)


def test_pointcloud_asarray_dtype_argument():
    arr = np.random.RandomState(0).rand(4, 3, 2)
    pc = sb.PointCloudTensor(arr)
    out = np.asarray(pc, dtype=np.float32)
    assert out.dtype == np.float32
    npt.assert_allclose(out, arr, atol=1e-6)


def test_pointcloud_single_cloud_to_dense():
    cloud = np.random.RandomState(1).rand(7, 2)
    pc = sb.PointCloudTensor(cloud)  # 0-d tensor (single cloud)
    assert pc.ndim == 0
    npt.assert_allclose(pc.to_dense(), cloud)
    npt.assert_allclose(np.asarray(pc), cloud)


def test_pointcloud_ragged_to_dense_raises():
    pc = sb.PointCloudTensor([np.random.rand(3, 2), np.random.rand(5, 2)])
    with pytest.raises(ValueError, match="ragged"):
        pc.to_dense()
    with pytest.raises(ValueError, match="ragged"):
        np.asarray(pc)


def test_pointcloud_empty_axis_to_dense_raises():
    pc = sb.zeros((0,), dtype=sb.pcloud64)
    with pytest.raises(ValueError, match="empty axis"):
        pc.to_dense()


# --- DistanceMatrixTensor / SymmetricMatrixTensor (#94) ---


@pytest.mark.parametrize("build", ["from_numpy", "from_dense"])
def test_distance_matrix_tensor_to_dense_roundtrip(build):
    stack = _symmetric_zero_diag_batch(4, 5)
    dt = getattr(sb.DistanceMatrixTensor, build)(stack)
    dense = dt.to_dense()
    assert dense.shape == (4, 5, 5)
    npt.assert_allclose(dense, stack, atol=1e-6)
    npt.assert_allclose(dt.to_numpy(), dense)
    npt.assert_allclose(np.asarray(dt), dense)


def test_distance_matrix_tensor_to_dense_multidim():
    stack = _symmetric_zero_diag_batch(6, 3).reshape(2, 3, 3, 3)
    dt = sb.DistanceMatrixTensor.from_numpy(stack)
    dense = dt.to_dense()
    assert dense.shape == (2, 3, 3, 3)
    npt.assert_allclose(dense, stack, atol=1e-6)


@pytest.mark.parametrize("build", ["from_numpy", "from_dense"])
def test_symmetric_matrix_tensor_to_dense_roundtrip(build):
    stack = _symmetric_batch((3,), 4)
    sm = getattr(sb.SymmetricMatrixTensor, build)(stack)
    dense = sm.to_dense()
    assert dense.shape == (3, 4, 4)
    npt.assert_allclose(dense, stack, atol=1e-6)
    npt.assert_allclose(sm.to_numpy(), dense)
    npt.assert_allclose(np.asarray(sm), dense)


def test_symmetric_matrix_tensor_to_dense_multidim():
    stack = _symmetric_batch((2, 3), 4)
    sm = sb.SymmetricMatrixTensor.from_numpy(stack)
    dense = sm.to_dense()
    assert dense.shape == (2, 3, 4, 4)
    npt.assert_allclose(dense, stack, atol=1e-6)


def test_matrix_tensor_asarray_dtype_argument():
    stack = _symmetric_zero_diag_batch(2, 4)
    dt = sb.DistanceMatrixTensor.from_numpy(stack)
    out = np.asarray(dt, dtype=np.float32)
    assert out.dtype == np.float32
    npt.assert_allclose(out, stack, atol=1e-6)


def test_distance_matrix_tensor_empty_to_dense_raises():
    dt = sb.zeros((0,), dtype=sb.distmat64)
    with pytest.raises(ValueError, match="empty axis"):
        dt.to_dense()


# --- BarcodeTensor (#85) ---


def test_barcode_tensor_to_numpy_object_array():
    b0 = Barcode(np.array([[0.0, 1.0], [0.0, 2.0]]))
    b1 = Barcode(np.array([[0.0, 3.0]]))
    bt = mp.BarcodeTensor([b0, b1])
    out = bt.to_numpy()
    assert isinstance(out, np.ndarray)
    assert out.dtype == object
    assert out.shape == (2,)
    assert out[0].shape == (2, 2)
    assert out[1].shape == (1, 2)
    npt.assert_allclose(out[0], b0.to_numpy())
    npt.assert_allclose(out[1], b1.to_numpy())


def test_barcode_tensor_tolist():
    b0 = Barcode(np.array([[0.0, 1.0], [0.0, 2.0]]))
    b1 = Barcode(np.array([[0.0, 3.0]]))
    bt = mp.BarcodeTensor([b0, b1])
    lst = bt.tolist()
    assert isinstance(lst, list)
    assert len(lst) == 2
    npt.assert_allclose(lst[0], b0.to_numpy())
    npt.assert_allclose(lst[1], b1.to_numpy())


def test_barcode_tensor_from_homology_to_numpy():
    clouds = sb.PointCloudTensor(np.random.RandomState(0).rand(2, 8, 2))
    bcs = mp.compute_persistent_homology(clouds, max_dim=1, verbose=False)
    out = bcs.to_numpy()
    assert out.dtype == object
    # shape is (n_clouds, max_dim + 1) == (2, 2)
    assert out.shape == tuple(bcs.shape)
    for idx in np.ndindex(*out.shape):
        entry = out[idx]
        assert entry.ndim == 2 and entry.shape[1] == 2
    # tolist mirrors the shape as nested lists
    lst = bcs.tolist()
    assert len(lst) == out.shape[0]
    assert len(lst[0]) == out.shape[1]


def test_barcode_tensor_no_array_protocol():
    # Barcodes are ragged: BarcodeTensor must NOT pretend to be a dense array.
    b0 = Barcode(np.array([[0.0, 1.0]]))
    bt = mp.BarcodeTensor([b0])
    with pytest.raises(TypeError):
        np.asarray(bt)
