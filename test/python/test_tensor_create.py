"""Tests for the :func:`stablebear.tensor` factory."""

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


def test_tensor_factory_numeric_inference():
    npt.assert_allclose(np.asarray(sb.tensor([1.0, 2.0, 3.0])), np.array([1.0, 2.0, 3.0]))
    assert isinstance(sb.tensor([1, 2, 3]), sb.IntTensor)
    assert isinstance(sb.tensor([True, False]), sb.BoolTensor)
    assert isinstance(sb.tensor([1.0, 2.0]), sb.FloatTensor)


@pytest.mark.parametrize(
    ("np_dtype", "tensor_type", "sb_dtype"),
    [
        pytest.param(np.float32, sb.PcfTensor, sb.pcf32, id="pcf32"),
        pytest.param(np.float64, sb.PcfTensor, sb.pcf64, id="pcf64"),
        pytest.param(np.int32, sb.IntPcfTensor, sb.pcf32i, id="pcf32i"),
        pytest.param(np.int64, sb.IntPcfTensor, sb.pcf64i, id="pcf64i"),
    ],
)
class TestTensorFactoryPcfInference:
    def test_equal_length(self, np_dtype, tensor_type, sb_dtype):
        arrays = [
            np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np_dtype),
            np.array([[0.0, 3.0], [2.0, 4.0]], dtype=np_dtype),
        ]
        tensor = sb.tensor([sb.Pcf(array) for array in arrays])

        assert isinstance(tensor, tensor_type)
        assert tensor.shape == (2,)
        assert tensor.dtype == sb_dtype
        for i, expected in enumerate(arrays):
            npt.assert_array_equal(tensor[i].to_numpy(), expected)

    def test_different_length(self, np_dtype, tensor_type, sb_dtype):
        arrays = [
            np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np_dtype),
            np.array([[0.0, 3.0], [2.0, 4.0], [5.0, 0.0]], dtype=np_dtype),
        ]
        tensor = sb.tensor([sb.Pcf(array) for array in arrays])

        assert isinstance(tensor, tensor_type)
        assert tensor.shape == (2,)
        assert tensor.dtype == sb_dtype
        for i, expected in enumerate(arrays):
            npt.assert_array_equal(tensor[i].to_numpy(), expected)


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
