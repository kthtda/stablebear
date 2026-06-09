import operator

import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


def _make_pcf32(vals):
    return sb.Pcf(np.array(vals, dtype=np.float32))


def _make_pcf64(vals):
    return sb.Pcf(np.array(vals, dtype=np.float64))


def _assert_scalar_op(np_arr, scalar, op, TensorType=sb.FloatTensor):
    """Assert that a scalar op on a tensor matches numpy."""
    result = np.asarray(op(TensorType(np_arr), scalar))
    expected = op(np_arr, scalar)
    npt.assert_array_equal(result, expected)


def _assert_scalar_iop(np_arr, scalar, iop, TensorType=sb.FloatTensor):
    """Assert that an in-place scalar op on a tensor matches numpy."""
    np_copy = np_arr.copy()
    iop(np_copy, scalar)
    t = TensorType(np_arr.copy())
    iop(t, scalar)
    npt.assert_array_equal(np.asarray(t), np_copy)


# --- Numeric tensors (parameterized across float and int) ---


_NUMERIC_TYPES = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
    pytest.param(sb.IntTensor, np.int32, id="int32"),
    pytest.param(sb.IntTensor, np.int64, id="int64"),
]


def _assert_binary_op(np_a, np_b, op, TensorType):
    """Assert that a binary op on tensors matches numpy."""
    result = np.asarray(op(TensorType(np_a), TensorType(np_b)))
    expected = op(np_a, np_b)
    npt.assert_array_equal(result, expected)


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestNumericScalarArithmetic:
    def test_add(self, TensorType, np_dtype):
        a = np.array([1, 2, 3], dtype=np_dtype)
        _assert_scalar_op(a, np_dtype(10), lambda x, y: x + y, TensorType)

    def test_iadd(self, TensorType, np_dtype):
        a = np.array([1, 2, 3], dtype=np_dtype)
        _assert_scalar_iop(a, np_dtype(10), operator.iadd, TensorType)

    def test_sub(self, TensorType, np_dtype):
        a = np.array([10, 20, 30], dtype=np_dtype)
        _assert_scalar_op(a, np_dtype(1), lambda x, y: x - y, TensorType)

    def test_isub(self, TensorType, np_dtype):
        a = np.array([10, 20, 30], dtype=np_dtype)
        _assert_scalar_iop(a, np_dtype(1), operator.isub, TensorType)

    def test_mul(self, TensorType, np_dtype):
        a = np.array([1, 2, 3], dtype=np_dtype)
        _assert_scalar_op(a, np_dtype(5), lambda x, y: x * y, TensorType)

    def test_imul(self, TensorType, np_dtype):
        a = np.array([1, 2, 3], dtype=np_dtype)
        _assert_scalar_iop(a, np_dtype(5), operator.imul, TensorType)

    def test_radd(self, TensorType, np_dtype):
        a = np.array([1, 2, 3], dtype=np_dtype)
        _assert_scalar_op(a, np_dtype(10), lambda x, y: y + x, TensorType)

    def test_rsub(self, TensorType, np_dtype):
        a = np.array([1, 2, 3], dtype=np_dtype)
        _assert_scalar_op(a, np_dtype(10), lambda x, y: y - x, TensorType)

    def test_rmul(self, TensorType, np_dtype):
        a = np.array([1, 2, 3], dtype=np_dtype)
        _assert_scalar_op(a, np_dtype(5), lambda x, y: y * x, TensorType)

    def test_add_does_not_modify_original(self, TensorType, np_dtype):
        a = np.array([1, 2, 3], dtype=np_dtype)
        X = TensorType(a)
        _ = X + np_dtype(10)
        npt.assert_array_equal(np.asarray(X), a)

    def test_2d(self, TensorType, np_dtype):
        a = np.array([[1, 2], [3, 4]], dtype=np_dtype)
        result = np.asarray(TensorType(a) * np_dtype(2))
        npt.assert_array_equal(result, a * np_dtype(2))


@pytest.mark.parametrize("TensorType, np_dtype", _NUMERIC_TYPES)
class TestNumericTensorTensorArithmetic:
    def test_add(self, TensorType, np_dtype):
        a = np.array([1, 2, 3], dtype=np_dtype)
        b = np.array([10, 20, 30], dtype=np_dtype)
        _assert_binary_op(a, b, lambda x, y: x + y, TensorType)

    def test_sub(self, TensorType, np_dtype):
        a = np.array([10, 20, 30], dtype=np_dtype)
        b = np.array([1, 2, 3], dtype=np_dtype)
        _assert_binary_op(a, b, lambda x, y: x - y, TensorType)

    def test_mul(self, TensorType, np_dtype):
        a = np.array([2, 3, 4], dtype=np_dtype)
        b = np.array([5, 6, 7], dtype=np_dtype)
        _assert_binary_op(a, b, lambda x, y: x * y, TensorType)


# --- Float-only: division (float semantics differ from int truncation) ---


class TestFloatDivision:
    def test_truediv_float64(self):
        a = np.array([10.0, 20.0, 30.0])
        _assert_scalar_op(a, 2.0, lambda x, y: x / y)

    def test_itruediv_float64(self):
        a = np.array([10.0, 20.0, 30.0])
        _assert_scalar_iop(a, 2.0, operator.itruediv)

    def test_truediv_float32(self):
        a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        _assert_scalar_op(a, np.float32(2.0), lambda x, y: x / y)


# --- Floor division ---


_FLOAT_TYPES = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
]


@pytest.mark.parametrize("TensorType, np_dtype", _FLOAT_TYPES)
class TestFloatFloorDiv:
    def test_floordiv_scalar(self, TensorType, np_dtype):
        a = np.array([10.5, -7.3, 21.0], dtype=np_dtype)
        result = np.asarray(TensorType(a) // np_dtype(3))
        npt.assert_array_almost_equal(result, a // np_dtype(3))

    def test_floordiv_tensor(self, TensorType, np_dtype):
        a = np.array([10.5, -7.3, 21.0], dtype=np_dtype)
        b = np.array([3.0, 2.0, 4.0], dtype=np_dtype)
        result = np.asarray(TensorType(a) // TensorType(b))
        npt.assert_array_almost_equal(result, a // b)

    def test_rfloordiv_scalar(self, TensorType, np_dtype):
        a = np.array([3.0, 7.0, 4.0], dtype=np_dtype)
        result = np.asarray(np_dtype(10) // TensorType(a))
        npt.assert_array_almost_equal(result, np_dtype(10) // a)

    def test_ifloordiv_scalar(self, TensorType, np_dtype):
        a = np.array([10.5, -7.3, 21.0], dtype=np_dtype)
        t = TensorType(a.copy())
        t //= np_dtype(3)
        npt.assert_array_almost_equal(np.asarray(t), a // np_dtype(3))


# --- PCF tensors (parametrized across all PCF dtypes) ---


_PCF_DTYPES = [
    pytest.param(sb.pcf32, np.float32, sb.PcfTensor, id="pcf32"),
    pytest.param(sb.pcf64, np.float64, sb.PcfTensor, id="pcf64"),
    pytest.param(sb.pcf32i, np.int32, sb.IntPcfTensor, id="pcf32i"),
    pytest.param(sb.pcf64i, np.int64, sb.IntPcfTensor, id="pcf64i"),
]


def _make_pcf(np_dtype, vals):
    return sb.Pcf(np.array(vals, dtype=np_dtype))


@pytest.mark.parametrize("pcf_dtype, np_dtype, tensor_cls", _PCF_DTYPES)
class TestPcfTensorArithmetic:
    def _make_tensor(self, pcf_dtype, np_dtype):
        X = sb.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 1], [2, 3]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        return X

    def test_add_pcf(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        pcf = _make_pcf(np_dtype, [[0, 100], [2, 200]])
        Y = X + pcf
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(101)
        assert Y[1].to_numpy()[0, 1] == np_dtype(110)

    def test_iadd_pcf(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        pcf = _make_pcf(np_dtype, [[0, 100], [2, 200]])
        X += pcf
        assert X[0].to_numpy()[0, 1] == np_dtype(101)

    def test_sub_pcf(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        pcf = _make_pcf(np_dtype, [[0, 1], [2, 1]])
        Y = X - pcf
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(0)
        assert Y[1].to_numpy()[0, 1] == np_dtype(9)

    def test_isub_pcf(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        pcf = _make_pcf(np_dtype, [[0, 1], [2, 1]])
        X -= pcf
        assert X[0].to_numpy()[0, 1] == np_dtype(0)

    def test_add_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        Y = X + np_dtype(5)
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(6)
        assert Y[1].to_numpy()[0, 1] == np_dtype(15)

    def test_iadd_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        X += np_dtype(5)
        assert X[0].to_numpy()[0, 1] == np_dtype(6)

    def test_radd_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        Y = np_dtype(5) + X
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(6)

    def test_sub_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        Y = X - np_dtype(1)
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(0)
        assert Y[1].to_numpy()[0, 1] == np_dtype(9)

    def test_rsub_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        Y = np_dtype(100) - X
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(99)
        assert Y[1].to_numpy()[0, 1] == np_dtype(90)

    def test_isub_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        X -= np_dtype(1)
        assert X[0].to_numpy()[0, 1] == np_dtype(0)

    def test_mul_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        Y = X * np_dtype(2)
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(2)
        assert Y[1].to_numpy()[0, 1] == np_dtype(20)

    def test_imul_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        X *= np_dtype(2)
        assert X[0].to_numpy()[0, 1] == np_dtype(2)

    def test_truediv_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = sb.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 4], [2, 8]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        Y = X / np_dtype(2)
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(2)
        assert Y[1].to_numpy()[0, 1] == np_dtype(5)

    def test_rtruediv_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = sb.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 2], [2, 4]])
        X[1] = _make_pcf(np_dtype, [[0, 5], [2, 10]])
        Y = np_dtype(20) / X
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(10)
        assert Y[1].to_numpy()[0, 1] == np_dtype(4)

    def test_itruediv_scalar(self, pcf_dtype, np_dtype, tensor_cls):
        X = sb.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 4], [2, 8]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        X /= np_dtype(2)
        assert X[0].to_numpy()[0, 1] == np_dtype(2)

    def test_add_does_not_modify_original(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        orig = X[0].to_numpy().copy()
        _ = X + _make_pcf(np_dtype, [[0, 100], [2, 200]])
        npt.assert_array_equal(X[0].to_numpy(), orig)

    def test_mul_does_not_modify_original(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        orig = X[0].to_numpy().copy()
        _ = X * np_dtype(10)
        npt.assert_array_equal(X[0].to_numpy(), orig)

    def test_neg(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        Y = -X
        assert isinstance(Y, tensor_cls)
        assert Y[0].to_numpy()[0, 1] == np_dtype(-1)
        assert Y[1].to_numpy()[0, 1] == np_dtype(-10)

    def test_neg_does_not_modify_original(self, pcf_dtype, np_dtype, tensor_cls):
        X = self._make_tensor(pcf_dtype, np_dtype)
        orig = X[0].to_numpy().copy()
        _ = -X
        npt.assert_array_equal(X[0].to_numpy(), orig)


@pytest.mark.parametrize("pcf_dtype, np_dtype, tensor_cls", _PCF_DTYPES)
class TestPcfTensorMulDiv:
    def test_mul_tensor_tensor(self, pcf_dtype, np_dtype, tensor_cls):
        X = sb.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 2], [2, 4]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        Y = sb.zeros((2,), dtype=pcf_dtype)
        Y[0] = _make_pcf(np_dtype, [[0, 3], [2, 5]])
        Y[1] = _make_pcf(np_dtype, [[0, 2], [2, 10]])
        Z = X * Y
        assert isinstance(Z, tensor_cls)
        assert Z[0].to_numpy()[0, 1] == np_dtype(6)
        assert Z[1].to_numpy()[0, 1] == np_dtype(20)

    def test_imul_tensor_tensor(self, pcf_dtype, np_dtype, tensor_cls):
        X = sb.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 2], [2, 4]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        Y = sb.zeros((2,), dtype=pcf_dtype)
        Y[0] = _make_pcf(np_dtype, [[0, 3], [2, 5]])
        Y[1] = _make_pcf(np_dtype, [[0, 2], [2, 10]])
        X *= Y
        assert X[0].to_numpy()[0, 1] == np_dtype(6)

    def test_truediv_tensor_tensor(self, pcf_dtype, np_dtype, tensor_cls):
        X = sb.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 6], [2, 8]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        Y = sb.zeros((2,), dtype=pcf_dtype)
        Y[0] = _make_pcf(np_dtype, [[0, 2], [2, 2]])
        Y[1] = _make_pcf(np_dtype, [[0, 5], [2, 4]])
        Z = X / Y
        assert isinstance(Z, tensor_cls)
        assert Z[0].to_numpy()[0, 1] == np_dtype(3)
        assert Z[1].to_numpy()[0, 1] == np_dtype(2)

    def test_itruediv_tensor_tensor(self, pcf_dtype, np_dtype, tensor_cls):
        X = sb.zeros((2,), dtype=pcf_dtype)
        X[0] = _make_pcf(np_dtype, [[0, 6], [2, 8]])
        X[1] = _make_pcf(np_dtype, [[0, 10], [2, 20]])
        Y = sb.zeros((2,), dtype=pcf_dtype)
        Y[0] = _make_pcf(np_dtype, [[0, 2], [2, 2]])
        Y[1] = _make_pcf(np_dtype, [[0, 5], [2, 4]])
        X /= Y
        assert X[0].to_numpy()[0, 1] == np_dtype(3)


# --- Float tensor broadcasting ---


def _check_broadcast_op(np_a, np_b, op, TensorType=sb.FloatTensor):
    """Apply op to both numpy arrays and stablebear tensors, assert results match."""
    X = TensorType(np_a)
    Y = TensorType(np_b)
    expected = op(np_a, np_b)
    result = np.asarray(op(X, Y))
    npt.assert_array_almost_equal(result, expected)


class TestFloat64TensorBroadcast:
    def test_add_same_shape(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        _check_broadcast_op(a, b, lambda x, y: x + y)

    def test_add_broadcast_row_vector(self):
        a = np.array([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]])
        b = np.array([100.0, 200.0, 300.0, 400.0])
        _check_broadcast_op(a, b, lambda x, y: x + y)

    def test_add_broadcast_both_expand(self):
        a = np.array([[1.0], [2.0]])
        b = np.array([[10.0, 20.0, 30.0]])
        _check_broadcast_op(a, b, lambda x, y: x + y)

    def test_sub_broadcast(self):
        a = np.array([[10.0, 20.0], [30.0, 40.0]])
        b = np.array([1.0, 2.0])
        _check_broadcast_op(a, b, lambda x, y: x - y)

    def test_mul_broadcast(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([10.0, 20.0])
        _check_broadcast_op(a, b, lambda x, y: x * y)

    def test_truediv_broadcast(self):
        a = np.array([[10.0, 20.0], [30.0, 40.0]])
        b = np.array([2.0, 5.0])
        _check_broadcast_op(a, b, lambda x, y: x / y)

    def test_iadd_broadcast(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([10.0, 20.0])
        X = sb.FloatTensor(a.copy())
        Y = sb.FloatTensor(b)
        X += Y
        expected = a + b
        npt.assert_array_equal(np.asarray(X), expected)

    def test_iadd_incompatible_raises(self):
        X = sb.FloatTensor(np.array([1.0, 2.0]))
        Y = sb.FloatTensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError):
            X += Y

    def test_incompatible_shapes_raise(self):
        X = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))
        Y = sb.FloatTensor(np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            _ = X + Y

    def test_add_does_not_modify_originals(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        X = sb.FloatTensor(a)
        Y = sb.FloatTensor(b)
        _ = X + Y
        npt.assert_array_equal(np.asarray(X), a)
        npt.assert_array_equal(np.asarray(Y), b)


class TestNumericTensorPow:
    @pytest.fixture(params=[
        pytest.param((sb.FloatTensor, np.float32), id="float32"),
        pytest.param((sb.FloatTensor, np.float64), id="float64"),
    ])
    def tensor_info(self, request):
        return request.param

    def _make(self, tensor_info, vals):
        cls, dtype = tensor_info
        return cls(np.array(vals, dtype=dtype))

    def test_pow_2(self, tensor_info):
        _, dtype = tensor_info
        a = np.array([2.0, 3.0, 4.0], dtype=dtype)
        result = np.asarray(self._make(tensor_info, a) ** 2)
        npt.assert_array_equal(result, a ** 2)

    def test_pow_half(self, tensor_info):
        _, dtype = tensor_info
        a = np.array([4.0, 9.0, 16.0], dtype=dtype)
        result = np.asarray(self._make(tensor_info, a) ** 0.5)
        npt.assert_array_almost_equal(result, a ** 0.5)

    def test_pow_negative_exponent(self, tensor_info):
        _, dtype = tensor_info
        a = np.array([2.0, 4.0, 5.0], dtype=dtype)
        result = np.asarray(self._make(tensor_info, a) ** -1.0)
        npt.assert_array_almost_equal(result, a ** -1.0)

    def test_pow_does_not_mutate(self, tensor_info):
        a = np.array([2.0, 3.0, 4.0])
        X = self._make(tensor_info, a)
        _ = X ** 2
        npt.assert_array_equal(np.asarray(X), a)

    def test_pow_negative_base_fractional_exp_warns(self, tensor_info):
        X = self._make(tensor_info, [-2.0, 4.0])
        with pytest.warns(RuntimeWarning):
            Y = X ** 0.5
        assert np.isnan(np.asarray(Y)[0])
        npt.assert_almost_equal(np.asarray(Y)[1], 2.0)

    def test_pow_zero_base_negative_exp_warns(self, tensor_info):
        X = self._make(tensor_info, [0.0, 2.0])
        with pytest.warns(RuntimeWarning):
            Y = X ** -1.0
        assert np.isinf(np.asarray(Y)[0])
        npt.assert_almost_equal(np.asarray(Y)[1], 0.5)

    def test_ipow(self, tensor_info):
        _, dtype = tensor_info
        a = np.array([2.0, 3.0, 4.0], dtype=dtype)
        X = self._make(tensor_info, a)
        X **= 2
        npt.assert_array_equal(np.asarray(X), a ** 2)


class TestPcfTensorPow:
    @pytest.fixture(params=[
        pytest.param((sb.pcf32, np.float32), id="pcf32"),
        pytest.param((sb.pcf64, np.float64), id="pcf64"),
    ])
    def pcf_info(self, request):
        return request.param

    def _make_tensor(self, pcf_info, vals_list):
        pcf_dtype, np_dtype = pcf_info
        tensor = sb.zeros((len(vals_list),), dtype=pcf_dtype)
        for i, vals in enumerate(vals_list):
            tensor[i] = sb.Pcf(np.array(vals, dtype=np_dtype))
        return tensor, np_dtype

    def test_pow_2(self, pcf_info):
        T, np_dtype = self._make_tensor(pcf_info, [
            [[0.0, 2.0], [1.0, 3.0]],
            [[0.0, 4.0], [1.0, 5.0]],
        ])
        R = T ** 2
        expected_0 = sb.Pcf(np.array([[0.0, 4.0], [1.0, 9.0]], dtype=np_dtype))
        expected_1 = sb.Pcf(np.array([[0.0, 16.0], [1.0, 25.0]], dtype=np_dtype))
        assert R[0] == expected_0
        assert R[1] == expected_1

    def test_pow_does_not_mutate(self, pcf_info):
        T, _ = self._make_tensor(pcf_info, [
            [[0.0, 2.0], [1.0, 3.0]],
        ])
        original = T.copy()
        _ = T ** 2
        assert T.array_equal(original)

    def test_ipow(self, pcf_info):
        T, np_dtype = self._make_tensor(pcf_info, [
            [[0.0, 2.0], [1.0, 3.0]],
            [[0.0, 4.0], [1.0, 5.0]],
        ])
        T **= 2
        expected_0 = sb.Pcf(np.array([[0.0, 4.0], [1.0, 9.0]], dtype=np_dtype))
        expected_1 = sb.Pcf(np.array([[0.0, 16.0], [1.0, 25.0]], dtype=np_dtype))
        assert T[0] == expected_0
        assert T[1] == expected_1

    def test_pow_negative_base_fractional_exp_warns(self, pcf_info):
        T, _ = self._make_tensor(pcf_info, [
            [[0.0, -2.0], [1.0, 4.0]],
        ])
        with pytest.warns(RuntimeWarning):
            _ = T ** 0.5


class TestFloat32TensorBroadcast:
    def test_add_broadcast(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([10.0, 20.0], dtype=np.float32)
        _check_broadcast_op(a, b, lambda x, y: x + y, sb.FloatTensor)
