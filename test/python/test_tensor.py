import pytest

import stablebear as sb
import stablebear._sb_cpp as cpp


def test_shape_dunder():
    shp = sb.Shape((2, 3, 5))

    str_ = str(shp)
    repr_ = repr(shp)

    assert str_ == "Shape(2, 3, 5)"
    assert repr_ == "(2, 3, 5)"

    assert len(shp) == 3
    assert shp[0] == 2
    assert shp[1] == 3
    assert shp[2] == 5

    with pytest.raises(IndexError):
        print(shp[3])


def test_basic_shape_strides():
    X = sb.zeros((7, 3, 5, 6), dtype=sb.float64)

    assert X.shape == sb.Shape((7, 3, 5, 6))
    assert X.shape != sb.Shape((1, 2, 3))

    assert X.strides[0] == 3 * 5 * 6 * 1  # 90
    assert X.strides[1] == 5 * 6 * 1  # 30
    assert X.strides[2] == 6 * 1  #  6
    assert X.strides[3] == 1  #  1


def test_construct_1d_tensor():
    s = sb.Shape(10)
    X = sb.zeros(s)

    assert X.shape == s


def test_dtype_results_in_correct_type():
    X32 = sb.zeros((3, 2), dtype=sb.pcf32)
    assert isinstance(X32, sb.PcfTensor)
    assert X32.dtype == sb.pcf32

    assert isinstance(X32[0, 0], sb.Pcf)
    assert isinstance(X32[0, 0]._data, cpp.Pcf_f32_f32)

    X64 = sb.zeros((3, 2), dtype=sb.pcf64)
    assert isinstance(X64, sb.PcfTensor)
    assert X64.dtype == sb.pcf64

    assert isinstance(X64[0, 0], sb.Pcf)
    assert isinstance(X64[0, 0]._data, cpp.Pcf_f64_f64)


def test_tensor_copy_does_not_modify_original():
    X = sb.zeros((10, 10), dtype=sb.float64)

    X[2, 3] = 1.5

    Y = X.copy()

    Y[2, 3] = 0.5

    assert Y[2, 3] != X[2, 3]

    X[2, 3] = 4.5

    assert Y[2, 3] != X[2, 3]


ALL_DTYPES = [
    sb.pcf32, sb.pcf64,
    sb.pcf32i, sb.pcf64i,
    sb.float32, sb.float64,
    sb.boolean,
    sb.pcloud32, sb.pcloud64,
    sb.barcode32, sb.barcode64,
    sb.distmat32, sb.distmat64,
    sb.symmat32, sb.symmat64,
]


@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=[d.__name__ for d in ALL_DTYPES])
def test_copy_preserves_dtype(dtype):
    X = sb.zeros((3,), dtype=dtype)
    Y = X.copy()
    assert type(Y) is type(X)
    assert Y.dtype == X.dtype
    assert Y.shape == X.shape


@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=[d.__name__ for d in ALL_DTYPES])
def test_slice_preserves_dtype(dtype):
    X = sb.zeros((5,), dtype=dtype)
    Y = X[1:4]
    assert type(Y) is type(X)
    assert Y.dtype == X.dtype


@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=[d.__name__ for d in ALL_DTYPES])
def test_flatten_preserves_dtype(dtype):
    X = sb.zeros((2, 3), dtype=dtype)
    Y = X.flatten()
    assert type(Y) is type(X)
    assert Y.dtype == X.dtype
    assert Y.shape == (6,)


@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=[d.__name__ for d in ALL_DTYPES])
def test_broadcast_to_preserves_dtype(dtype):
    X = sb.zeros((1,), dtype=dtype)
    Y = X.broadcast_to((5,))
    assert type(Y) is type(X)
    assert Y.dtype == X.dtype
    assert Y.shape == (5,)


import numpy as np
from stablebear.base_tensor import BoolTensor


@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=[d.__name__ for d in ALL_DTYPES])
def test_masked_select_preserves_dtype(dtype):
    X = sb.zeros((3,), dtype=dtype)
    mask = BoolTensor(np.array([True, False, True]))
    Y = X[mask]
    assert type(Y) is type(X)
    assert Y.dtype == X.dtype
    assert Y.shape == (2,)


# Arithmetic ops only apply to numeric and PCF tensors.
ARITHMETIC_DTYPES = [
    sb.pcf32, sb.pcf64,
    sb.pcf32i, sb.pcf64i,
    sb.float32, sb.float64,
]


@pytest.mark.parametrize("dtype", ARITHMETIC_DTYPES, ids=[d.__name__ for d in ARITHMETIC_DTYPES])
def test_arithmetic_preserves_dtype(dtype):
    X = sb.zeros((3,), dtype=dtype)
    Y = X + X
    assert type(Y) is type(X)
    assert Y.dtype == X.dtype

    Z = -X
    assert type(Z) is type(X)
    assert Z.dtype == X.dtype
