import numpy as np
import pytest

import stablebear as sb
from stablebear.distance_matrix import DistanceMatrix
from stablebear.symmetric_matrix import SymmetricMatrix


# ---------------------------------------------------------------------------
# Bug #41: PointCloudTensor advertised float/int as valid setitem dtypes, so a
# scalar assignment passed validation and then raised a misleading FloatTensor
# constructor error ("Cannot create FloatTensor from <class 'float'>"). A cloud
# cell is an (n_points, dim) array, so a scalar is not a meaningful RHS and must
# be rejected up front with a clear message.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [sb.pcloud32, sb.pcloud64])
@pytest.mark.parametrize("scalar", [3.14, 7, np.float64(3.0)])
def test_pcloud_scalar_assignment_raises_clean_error(dtype, scalar):
    t = sb.zeros((2,), dtype=dtype)
    with pytest.raises(TypeError) as excinfo:
        t[0] = scalar
    msg = str(excinfo.value)
    assert "Cannot create FloatTensor" not in msg
    assert "accepted" in msg


@pytest.mark.parametrize("dtype", [sb.pcloud32, sb.pcloud64])
def test_pcloud_array_assignment_still_works(dtype):
    t = sb.zeros((2,), dtype=dtype)
    cloud = np.arange(8.0).reshape(4, 2)
    t[0] = cloud
    np.testing.assert_array_equal(np.asarray(t[0]), cloud)


# ---------------------------------------------------------------------------
# Bug #42: assigning a DistanceMatrix / SymmetricMatrix (or whole tensor) of the
# wrong 32/64-bit precision into a matrix tensor leaked the raw pybind
# "_set_element(): incompatible function arguments" / "__setitem__(): ..." dump.
# It must raise a clear TypeError naming the dtypes; matching precision still
# works.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tensor_dtype, elem_dtype, tensor_cls, elem_cls",
    [
        (sb.distmat64, sb.float32, sb.DistanceMatrixTensor, DistanceMatrix),
        (sb.distmat32, sb.float64, sb.DistanceMatrixTensor, DistanceMatrix),
        (sb.symmat64, sb.float32, sb.SymmetricMatrixTensor, SymmetricMatrix),
        (sb.symmat32, sb.float64, sb.SymmetricMatrixTensor, SymmetricMatrix),
    ],
)
def test_matrix_element_assign_dtype_mismatch_clean(
    tensor_dtype, elem_dtype, tensor_cls, elem_cls
):
    t = sb.zeros((2,), dtype=tensor_dtype)
    elem = elem_cls(3, dtype=elem_dtype)
    with pytest.raises(TypeError) as excinfo:
        t[0] = elem
    msg = str(excinfo.value)
    assert "incompatible function arguments" not in msg
    assert "_set_element" not in msg
    assert elem_dtype.name in msg and tensor_dtype.name in msg


@pytest.mark.parametrize(
    "dst_dtype, src_dtype",
    [
        (sb.distmat64, sb.distmat32),
        (sb.symmat64, sb.symmat32),
    ],
)
def test_matrix_tensor_assign_dtype_mismatch_clean(dst_dtype, src_dtype):
    dst = sb.zeros((2,), dtype=dst_dtype)
    src = sb.zeros((2,), dtype=src_dtype)
    with pytest.raises(TypeError) as excinfo:
        dst[:] = src
    msg = str(excinfo.value)
    assert "incompatible function arguments" not in msg
    assert "__setitem__" not in msg
    assert dst_dtype.name in msg and src_dtype.name in msg


@pytest.mark.parametrize(
    "dst_dtype, src_dtype",
    [
        (sb.distmat32, sb.distmat64),
        (sb.symmat32, sb.symmat64),
    ],
)
def test_matrix_tensor_assign_dtype_mismatch_clean_reverse(dst_dtype, src_dtype):
    dst = sb.zeros((2,), dtype=dst_dtype)
    src = sb.zeros((2,), dtype=src_dtype)
    with pytest.raises(TypeError) as excinfo:
        dst[:] = src
    msg = str(excinfo.value)
    assert "incompatible function arguments" not in msg
    assert "__setitem__" not in msg
    assert dst_dtype.name in msg and src_dtype.name in msg


@pytest.mark.parametrize(
    "tensor_dtype, elem_dtype, elem_cls",
    [
        (sb.distmat64, sb.float64, DistanceMatrix),
        (sb.distmat32, sb.float32, DistanceMatrix),
        (sb.symmat64, sb.float64, SymmetricMatrix),
        (sb.symmat32, sb.float32, SymmetricMatrix),
    ],
)
def test_matrix_matching_precision_assignment_still_works(
    tensor_dtype, elem_dtype, elem_cls
):
    t = sb.zeros((2,), dtype=tensor_dtype)
    elem = elem_cls(3, dtype=elem_dtype)
    elem[0, 1] = 2.0
    t[0] = elem
    assert t[0][0, 1] == 2.0
    # whole-tensor assignment of matching precision also works
    t[:] = sb.zeros((2,), dtype=tensor_dtype)
