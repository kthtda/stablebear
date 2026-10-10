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


_DTYPE_INFO = {
    np.bool_: (sb.BoolTensor, sb.boolean),
    np.int32: (sb.IntTensor, sb.int32),
    np.int64: (sb.IntTensor, sb.int64),
    np.uint32: (sb.IntTensor, sb.uint32),
    np.uint64: (sb.IntTensor, sb.uint64),
    np.float32: (sb.FloatTensor, sb.float32),
    np.float64: (sb.FloatTensor, sb.float64),
}


def _assert_numeric_tensor_inference(values, np_dtype):
    expected = np.asarray(values, dtype=np_dtype)
    tensor = sb.tensor(values)
    tensor_type, sb_dtype = _DTYPE_INFO[np_dtype]

    assert isinstance(tensor, tensor_type)
    assert tensor.shape == expected.shape
    assert tensor.dtype == sb_dtype
    npt.assert_array_equal(np.asarray(tensor), expected)


def test_tensor_factory_numpy_bool_array_inference():
    values = np.array([[True, False], [False, True]], dtype=np.bool_)
    _assert_numeric_tensor_inference(values, np.bool_)


@pytest.mark.parametrize(
    "np_dtype",
    [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64],
)
def test_tensor_factory_numpy_array_inference(np_dtype):
    values = np.array([[0, 1], [2, 3]], dtype=np_dtype)
    _assert_numeric_tensor_inference(values, np_dtype)


def test_tensor_factory_bool_inference():
    _assert_numeric_tensor_inference([True, False], np.bool_)


@pytest.mark.parametrize(
    "np_dtype",
    [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64],
)
def test_tensor_factory_numeric_inference(np_dtype):
    values = [np_dtype(value) for value in [0, 1, 2]]
    _assert_numeric_tensor_inference(values, np_dtype)


def test_tensor_factory_nested_bool_inference():
    _assert_numeric_tensor_inference([[True, False], [False, True]], np.bool_)


@pytest.mark.parametrize(
    "np_dtype",
    [np.int32, np.int64, np.uint32, np.uint64, np.float32, np.float64],
)
def test_tensor_factory_nested_numeric_inference(np_dtype):
    values = [[np_dtype(value) for value in row] for row in [[0, 1], [2, 3]]]
    _assert_numeric_tensor_inference(values, np_dtype)


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

    def test_nested_shape(self, np_dtype, tensor_type, sb_dtype):
        arrays = [
            [
                np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np_dtype),
                np.array([[0.0, 3.0], [2.0, 4.0]], dtype=np_dtype),
            ],
            [
                np.array([[0.0, 5.0], [3.0, 6.0]], dtype=np_dtype),
                np.array([[0.0, 7.0], [4.0, 8.0]], dtype=np_dtype),
            ],
        ]
        tensor = sb.tensor([
            [sb.Pcf(array) for array in row]
            for row in arrays
        ])

        assert isinstance(tensor, tensor_type)
        assert tensor.shape == (2, 2)
        assert tensor.dtype == sb_dtype
        for i, row in enumerate(arrays):
            for j, expected in enumerate(row):
                npt.assert_array_equal(tensor[i, j].to_numpy(), expected)

    def test_single_pcf(self, np_dtype, tensor_type, sb_dtype):
        array = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np_dtype)
        tensor = sb.tensor(sb.Pcf(array))

        assert isinstance(tensor, tensor_type)
        assert tensor.shape == (1,)
        assert tensor.dtype == sb_dtype
        npt.assert_array_equal(tensor[0].to_numpy(), array)


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


@pytest.mark.parametrize("values", [[3, 1, 3], np.array([3, 1, 3])])
def test_indices_is_uint64_tensor_factory(values):
    actual = sb.indices(values)

    assert isinstance(actual, sb.IntTensor)
    assert actual.dtype == sb.uint64
    npt.assert_array_equal(np.asarray(actual), [3, 1, 3])
