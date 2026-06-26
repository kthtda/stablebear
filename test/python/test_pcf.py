import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb
from stablebear.functional import Pcf


class TestCopyConstructor:
    def test_copy_from_pcf_f32(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        g = Pcf(f)
        assert g.ttype == sb.float32
        assert g.vtype == sb.float32
        npt.assert_array_equal(f.to_numpy(), g.to_numpy())

    def test_copy_from_pcf_f64(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64))
        g = Pcf(f)
        assert g.ttype == sb.float64
        assert g.vtype == sb.float64
        npt.assert_array_equal(f.to_numpy(), g.to_numpy())


class TestIntArrayInput:
    def test_int64_array(self):
        arr = np.array([[0, 1], [2, 3]], dtype=np.int64)
        f = Pcf(arr)
        assert f.ttype == sb.int64
        assert f.vtype == sb.int64
        assert f.size == 2

    def test_int32_array(self):
        arr = np.array([[0, 1], [2, 3]], dtype=np.int32)
        f = Pcf(arr)
        assert f.ttype == sb.int32
        assert f.vtype == sb.int32
        assert f.size == 2

    def test_unsupported_array_dtype_raises(self):
        arr = np.array([[0, 1], [2, 3]], dtype=np.complex128)
        with pytest.raises(ValueError, match="Unsupported array type"):
            Pcf(arr)

    def test_nonzero_first_time_raises(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with pytest.raises(Exception, match="t=0"):
            Pcf(arr)


class TestListInput:
    def test_list_default_dtype(self):
        f = Pcf([[0.0, 1.0], [2.0, 3.0]])
        assert f.ttype == sb.float64
        assert f.vtype == sb.float64
        assert f.size == 2

    def test_list_f32(self):
        f = Pcf([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        assert f.ttype == sb.float32
        assert f.vtype == sb.float32
        npt.assert_array_almost_equal(
            f.to_numpy(), [[0.0, 1.0], [2.0, 3.0]]
        )

    def test_list_f64(self):
        f = Pcf([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        assert f.ttype == sb.float64
        assert f.vtype == sb.float64
        npt.assert_array_almost_equal(
            f.to_numpy(), [[0.0, 1.0], [2.0, 3.0]]
        )

    def test_list_int32(self):
        f = Pcf([[0, 1], [2, 3]], dtype=np.int32)
        assert f.ttype == sb.int32
        assert f.vtype == sb.int32
        npt.assert_array_equal(f.to_numpy(), [[0, 1], [2, 3]])

    def test_list_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            Pcf([[0.0, 1.0]], dtype=np.complex128)


class TestUnsupportedInputType:
    def test_string_raises(self):
        with pytest.raises(ValueError, match="unsupported input data"):
            Pcf("not a valid input")

    def test_dict_raises(self):
        with pytest.raises(ValueError, match="unsupported input data"):
            Pcf({"a": 1})


class TestDtypeConstructor:
    def test_ndarray_with_pcf32_dtype(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        f = Pcf(arr, dtype=sb.pcf32)
        assert f.ttype == sb.float32

    def test_ndarray_with_pcf64_dtype(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        f = Pcf(arr, dtype=sb.pcf64)
        assert f.ttype == sb.float64


class TestInternalMethods:
    def test_get_time_type_f32(self):
        f = Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
        assert isinstance(f._get_time_type(), str)

    def test_get_value_type_f32(self):
        f = Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
        assert isinstance(f._get_value_type(), str)

    def test_get_time_value_type_f32(self):
        f = Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
        result = f._get_time_value_type()
        assert "_" in result


class TestArithmetic:
    def test_add(self):
        a = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        b = Pcf(np.array([[0.0, 10.0], [2.0, 20.0]], dtype=np.float32))
        c = a + b
        npt.assert_array_almost_equal(c.to_numpy()[:, 1], [11.0, 23.0])

    def test_sub(self):
        a = Pcf(np.array([[0.0, 10.0], [2.0, 20.0]], dtype=np.float32))
        b = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        c = a - b
        npt.assert_array_almost_equal(c.to_numpy()[:, 1], [9.0, 17.0])

    def test_mul_pcf(self):
        a = Pcf(np.array([[0.0, 2.0], [2.0, 3.0]], dtype=np.float32))
        b = Pcf(np.array([[0.0, 5.0], [2.0, 4.0]], dtype=np.float32))
        c = a * b
        npt.assert_array_almost_equal(c.to_numpy()[:, 1], [10.0, 12.0])

    def test_add_scalar(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        g = f + 10.0
        npt.assert_array_almost_equal(g.to_numpy()[:, 1], [11.0, 13.0])

    def test_radd_scalar(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        g = 10.0 + f
        npt.assert_array_almost_equal(g.to_numpy()[:, 1], [11.0, 13.0])

    def test_sub_scalar(self):
        f = Pcf(np.array([[0.0, 10.0], [2.0, 20.0]], dtype=np.float32))
        g = f - 1.0
        npt.assert_array_almost_equal(g.to_numpy()[:, 1], [9.0, 19.0])

    def test_rsub_scalar(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        g = 10.0 - f
        npt.assert_array_almost_equal(g.to_numpy()[:, 1], [9.0, 7.0])

    def test_mul_scalar(self):
        f = Pcf(np.array([[0.0, 2.0], [2.0, 3.0]], dtype=np.float32))
        g = f * 3.0
        npt.assert_array_almost_equal(g.to_numpy()[:, 1], [6.0, 9.0])

    def test_rmul_scalar(self):
        f = Pcf(np.array([[0.0, 2.0], [2.0, 3.0]], dtype=np.float32))
        g = 3.0 * f
        npt.assert_array_almost_equal(g.to_numpy()[:, 1], [6.0, 9.0])

    def test_div_pcf(self):
        a = Pcf(np.array([[0.0, 10.0], [2.0, 12.0]], dtype=np.float32))
        b = Pcf(np.array([[0.0, 2.0], [2.0, 3.0]], dtype=np.float32))
        c = a / b
        npt.assert_array_almost_equal(c.to_numpy()[:, 1], [5.0, 4.0])

    def test_div_scalar(self):
        f = Pcf(np.array([[0.0, 4.0], [2.0, 8.0]], dtype=np.float32))
        g = f / 2.0
        npt.assert_array_almost_equal(g.to_numpy()[:, 1], [2.0, 4.0])

    def test_neg(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        g = -f
        npt.assert_array_almost_equal(g.to_numpy()[:, 1], [-1.0, -3.0])

    def test_rdiv_scalar(self):
        f = Pcf(np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32))
        g = 8.0 / f
        npt.assert_array_almost_equal(g.to_numpy()[:, 1], [4.0, 2.0])

    def test_div_scalar_f64(self):
        f = Pcf(np.array([[0.0, 4.0], [2.0, 8.0]], dtype=np.float64))
        g = f / 2.0
        npt.assert_array_almost_equal(g.to_numpy()[:, 1], [2.0, 4.0])

    @pytest.mark.parametrize("op", [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
        lambda a, _: a + 10.0,
        lambda a, _: 10.0 + a,
        lambda a, _: a - 1.0,
        lambda a, _: 10.0 - a,
        lambda a, _: a * 2.0,
        lambda a, _: 2.0 * a,
        lambda a, _: a / 2.0,
        lambda a, _: 10.0 / a,
    ])
    def test_does_not_mutate(self, op):
        a = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        b = Pcf(np.array([[0.0, 10.0], [2.0, 20.0]], dtype=np.float32))
        a_orig = a.to_numpy().copy()
        b_orig = b.to_numpy().copy()
        _ = op(a, b)
        npt.assert_array_equal(a.to_numpy(), a_orig)
        npt.assert_array_equal(b.to_numpy(), b_orig)

    def test_mismatched_types_raises(self):
        a = Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
        b = Pcf(np.array([[0.0, 1.0]], dtype=np.float64))
        with pytest.raises(TypeError):
            a + b
        with pytest.raises(TypeError):
            a - b
        with pytest.raises(TypeError):
            a * b
        with pytest.raises(TypeError):
            a / b


class TestStr:
    def test_str_f32(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        s = str(f)
        assert "PCF" in s
        assert "float32" in s
        assert "size=2" in s

    def test_str_f64(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64))
        s = str(f)
        assert "float64" in s


class TestSize:
    def test_size(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        assert f.size == 3


class TestArrayProtocol:
    def test_array_with_dtype(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        arr = np.array(f, dtype=np.float64)
        assert arr.dtype == np.float64
        npt.assert_array_almost_equal(arr, [[0.0, 1.0], [2.0, 3.0]])

    def test_array_without_dtype(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        arr = np.array(f)
        assert arr.dtype == np.float32
        assert arr.shape == (2, 2)
        npt.assert_array_almost_equal(arr, [[0.0, 1.0], [2.0, 3.0]])


class TestEquality:
    def test_eq_same(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        g = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        assert f == g

    def test_eq_different(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        g = Pcf(np.array([[0.0, 1.0], [2.0, 9.0]], dtype=np.float32))
        assert not (f == g)


class TestRepr:
    def test_repr_matches_str(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        assert repr(f) == str(f)

    def test_repr_contents(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        r = repr(f)
        assert "PCF" in r
        assert "size=2" in r


class TestDtype:
    @pytest.mark.parametrize("np_dtype,expected", [
        (np.float32, sb.pcf32),
        (np.float64, sb.pcf64),
        (np.int32, sb.pcf32i),
        (np.int64, sb.pcf64i),
    ])
    def test_dtype(self, np_dtype, expected):
        f = Pcf(np.array([[0, 1], [2, 3]], dtype=np_dtype))
        assert f.dtype is expected


class TestArrayAccessors:
    def test_times_values_breakpoints(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
        f = Pcf(arr)
        npt.assert_array_equal(f.times, arr[:, 0])
        npt.assert_array_equal(f.values, arr[:, 1])
        npt.assert_array_equal(f.breakpoints, f.to_numpy())

    def test_breakpoints_full_array(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        f = Pcf(arr)
        npt.assert_array_equal(f.breakpoints, arr)

    def test_t_min_t_max(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
        f = Pcf(arr)
        assert f.t_min == arr[0, 0]
        assert f.t_max == arr[-1, 0]


class TestFromArrays:
    def test_basic(self):
        times = [0.0, 1.0, 3.0]
        values = [1.0, 2.0, 0.0]
        f = Pcf.from_arrays(times, values)
        expected = Pcf(np.column_stack((times, values)))
        npt.assert_array_equal(f.to_numpy(), expected.to_numpy())

    def test_dtype_honored(self):
        f = Pcf.from_arrays([0.0, 1.0], [1.0, 2.0], dtype=sb.pcf32)
        assert f.dtype is sb.pcf32

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Pcf.from_arrays([0.0, 1.0, 2.0], [1.0, 2.0])

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            Pcf.from_arrays([[0.0, 1.0]], [1.0, 2.0])

    def test_bad_first_time_raises(self):
        with pytest.raises(ValueError, match="t=0"):
            Pcf.from_arrays([1.0, 2.0], [1.0, 2.0])

