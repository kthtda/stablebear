#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for numpy/list constructors of the non-numeric tensor families
(issues #53, #92, #84). Previously the only way to populate a
PointCloud/DistanceMatrix/SymmetricMatrix tensor was zeros() plus a Python
element-assignment loop."""

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


def _symmetric_zero_diag_batch(N, n, seed=0):
    rng = np.random.RandomState(seed)
    out = np.zeros((N, n, n))
    for k in range(N):
        m = rng.rand(n, n)
        m = (m + m.T) / 2.0
        np.fill_diagonal(m, 0.0)
        out[k] = m
    return out


# --- PointCloudTensor (#92, #84, #53) ---


def test_pointcloud_from_4d_array():
    arr = np.arange(3 * 5 * 4 * 2, dtype=np.float64).reshape(3, 5, 4, 2)
    pc = sb.PointCloudTensor(arr)
    assert isinstance(pc, sb.PointCloudTensor)
    assert pc.shape == (3, 5)
    assert pc.dtype == sb.pcloud64
    for i in range(3):
        for j in range(5):
            npt.assert_allclose(np.asarray(pc[i, j]), arr[i, j])


def test_pointcloud_from_3d_batch():
    arr = np.random.RandomState(0).rand(6, 4, 2)
    pc = sb.PointCloudTensor(arr)
    assert pc.shape == (6,)
    assert pc.dtype == sb.pcloud64
    for i in range(6):
        npt.assert_allclose(np.asarray(pc[i]), arr[i])


def test_pointcloud_dtype_inference():
    arr32 = np.zeros((2, 3, 2), dtype=np.float32)
    assert sb.PointCloudTensor(arr32).dtype == sb.pcloud32
    arr64 = np.zeros((2, 3, 2), dtype=np.float64)
    assert sb.PointCloudTensor(arr64).dtype == sb.pcloud64


def test_pointcloud_explicit_dtype():
    arr = np.zeros((2, 3, 2), dtype=np.float64)
    pc = sb.PointCloudTensor(arr, dtype=sb.pcloud32)
    assert pc.dtype == sb.pcloud32


def test_pointcloud_from_ragged_list():
    clouds = [
        np.random.RandomState(1).rand(3, 2),
        np.random.RandomState(2).rand(5, 2),
    ]
    pc = sb.PointCloudTensor(clouds)
    assert pc.shape == (2,)
    npt.assert_allclose(np.asarray(pc[0]), clouds[0])
    npt.assert_allclose(np.asarray(pc[1]), clouds[1])


def test_pointcloud_from_ragged_tuple():
    clouds = (
        np.random.RandomState(1).rand(3, 2),
        np.random.RandomState(2).rand(5, 2),
    )
    pc = sb.PointCloudTensor(clouds)
    assert pc.shape == (2,)
    npt.assert_allclose(np.asarray(pc[0]), clouds[0])
    npt.assert_allclose(np.asarray(pc[1]), clouds[1])


def test_pointcloud_from_ragged_list_dtype_inference():
    clouds32 = [
        np.random.RandomState(1).rand(3, 2).astype(np.float32),
        np.random.RandomState(2).rand(5, 2).astype(np.float32),
    ]
    assert sb.PointCloudTensor(clouds32).dtype == sb.pcloud32
    clouds64 = [c.astype(np.float64) for c in clouds32]
    assert sb.PointCloudTensor(clouds64).dtype == sb.pcloud64


def test_pointcloud_from_ragged_list_explicit_dtype():
    clouds = [
        np.random.RandomState(1).rand(3, 2).astype(np.float64),
        np.random.RandomState(2).rand(5, 2).astype(np.float64),
    ]
    pc = sb.PointCloudTensor(clouds, dtype=sb.pcloud32)
    assert pc.dtype == sb.pcloud32
    npt.assert_allclose(np.asarray(pc[0]), clouds[0], atol=1e-6)


def test_pointcloud_from_empty_list_raises():
    with pytest.raises(ValueError):
        sb.PointCloudTensor([])


def test_pointcloud_ragged_list_mixed_dtype_raises():
    mixed = [
        np.random.RandomState(1).rand(3, 2).astype(np.float32),
        np.random.RandomState(2).rand(5, 2).astype(np.float64),
    ]
    with pytest.raises(TypeError, match="differing dtypes"):
        sb.PointCloudTensor(mixed)


def test_pointcloud_ragged_list_mixed_dtype_explicit_ok():
    mixed = [
        np.random.RandomState(1).rand(3, 2).astype(np.float32),
        np.random.RandomState(2).rand(5, 2).astype(np.float64),
    ]
    pc = sb.PointCloudTensor(mixed, dtype=sb.pcloud64)
    assert pc.dtype == sb.pcloud64
    npt.assert_allclose(np.asarray(pc[0]), mixed[0], atol=1e-6)
    npt.assert_allclose(np.asarray(pc[1]), mixed[1])


def test_pointcloud_bad_dtype_raises():
    with pytest.raises(TypeError):
        sb.PointCloudTensor(np.zeros((2, 3, 2)), dtype=sb.float64)


# --- DistanceMatrixTensor (#84, #53) ---


@pytest.mark.parametrize("build", ["from_numpy", "ctor"])
def test_distance_matrix_tensor_from_numpy(build):
    batch = _symmetric_zero_diag_batch(4, 5)
    if build == "from_numpy":
        dt = sb.DistanceMatrixTensor.from_numpy(batch)
    else:
        dt = sb.DistanceMatrixTensor(batch)
    assert dt.shape == (4,)
    assert dt.dtype == sb.distmat64
    for k in range(4):
        npt.assert_allclose(dt[k].to_dense(), batch[k], atol=1e-6)


def test_distance_matrix_tensor_multidim():
    batch = _symmetric_zero_diag_batch(6, 3).reshape(2, 3, 3, 3)
    dt = sb.DistanceMatrixTensor.from_numpy(batch)
    assert dt.shape == (2, 3)


def test_distance_matrix_tensor_non_square_raises():
    with pytest.raises(ValueError):
        sb.DistanceMatrixTensor.from_numpy(np.zeros((4, 3, 5)))


def test_distance_matrix_tensor_non_symmetric_raises():
    batch = np.array([[[0.0, 2.0], [9.0, 0.0]]])  # asymmetric off-diagonal
    with pytest.raises(ValueError, match="symmetric"):
        sb.DistanceMatrixTensor.from_numpy(batch)


def test_distance_matrix_tensor_nonzero_diagonal_raises():
    batch = np.array([[[5.0, 2.0], [2.0, 5.0]]])  # symmetric but nonzero diagonal
    with pytest.raises(ValueError, match="[Dd]iagonal"):
        sb.DistanceMatrixTensor.from_numpy(batch)


def test_distance_matrix_tensor_negative_entry_raises():
    batch = np.array([[[0.0, -2.0], [-2.0, 0.0]]])  # symmetric, zero diag, but negative
    with pytest.raises(ValueError, match="nonnegative"):
        sb.DistanceMatrixTensor.from_numpy(batch)


# --- SymmetricMatrixTensor (#53) ---


def test_symmetric_matrix_tensor_from_numpy():
    rng = np.random.RandomState(3)
    batch = np.zeros((3, 4, 4))
    for k in range(3):
        m = rng.rand(4, 4)
        batch[k] = (m + m.T) / 2.0
    sm = sb.SymmetricMatrixTensor.from_numpy(batch)
    assert sm.shape == (3,)
    assert sm.dtype == sb.symmat64
    for k in range(3):
        npt.assert_allclose(sm[k].to_dense(), batch[k], atol=1e-6)


def test_symmetric_matrix_tensor_non_square_raises():
    with pytest.raises(ValueError):
        sb.SymmetricMatrixTensor.from_numpy(np.zeros((4, 3, 5)))


def test_symmetric_matrix_tensor_non_symmetric_raises():
    batch = np.array([[[1.0, 2.0], [9.0, 1.0]]])  # asymmetric off-diagonal
    with pytest.raises(ValueError, match="symmetric"):
        sb.SymmetricMatrixTensor.from_numpy(batch)


# --- tensor() factory (#53) ---


def test_tensor_factory_numeric_inference():
    npt.assert_allclose(np.asarray(sb.tensor([1.0, 2.0, 3.0])), np.array([1.0, 2.0, 3.0]))
    assert isinstance(sb.tensor([1, 2, 3]), sb.IntTensor)
    assert isinstance(sb.tensor([True, False]), sb.BoolTensor)
    assert isinstance(sb.tensor([1.0, 2.0]), sb.FloatTensor)


def test_tensor_factory_pointcloud():
    arr = np.zeros((2, 3, 2))
    pc = sb.tensor(arr, dtype=sb.pcloud64)
    assert isinstance(pc, sb.PointCloudTensor)
    assert pc.shape == (2,)


def test_tensor_factory_distmat():
    batch = _symmetric_zero_diag_batch(3, 4)
    dt = sb.tensor(batch, dtype=sb.distmat64)
    assert isinstance(dt, sb.DistanceMatrixTensor)
    assert dt.shape == (3,)


def test_tensor_factory_unknown_dtype_raises():
    with pytest.raises(TypeError):
        sb.tensor(np.array(["a", "b"]))
