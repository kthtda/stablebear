import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb
from stablebear.functional import Pcf


class TestIntPcfDtypeConstruction:
    def test_from_float_array_with_pcf32i_dtype(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        f = Pcf(arr, dtype=sb.pcf32i)
        assert f.ttype == sb.int32
        assert f.vtype == sb.int32
        npt.assert_array_equal(f.to_numpy(), [[0, 1], [2, 3]])

    def test_from_float_array_with_pcf64i_dtype(self):
        arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        f = Pcf(arr, dtype=sb.pcf64i)
        assert f.ttype == sb.int64
        assert f.vtype == sb.int64

    def test_from_list_pcf32i_dtype(self):
        f = Pcf([[0, 1], [2, 3]], dtype=sb.pcf32i)
        assert f.ttype == sb.int32

    def test_from_list_pcf64i_dtype(self):
        f = Pcf([[0, 1], [2, 3]], dtype=sb.pcf64i)
        assert f.ttype == sb.int64


class TestIntPcfEval:
    def test_scalar_eval_i32(self):
        f = Pcf(np.array([[0, 10], [2, 20]], dtype=np.int32))
        assert f(0) == 10
        assert f(1) == 10
        assert f(2) == 20

    def test_scalar_eval_i64(self):
        f = Pcf(np.array([[0, 10], [2, 20]], dtype=np.int64))
        assert f(0) == 10
        assert f(2) == 20

    def test_array_eval(self):
        f = Pcf(np.array([[0, 10], [2, 20]], dtype=np.int32))
        times = np.array([0, 1, 2, 3], dtype=np.int32)
        result = f(times)
        npt.assert_array_equal(result, [10, 10, 20, 20])

    def test_negative_time_raises(self):
        f = Pcf(np.array([[0, 1]], dtype=np.int32))
        with pytest.raises(Exception):
            f(-1)


class TestIntPcfArithmetic:
    def test_add_i32(self):
        a = Pcf(np.array([[0, 1], [2, 3]], dtype=np.int32))
        b = Pcf(np.array([[0, 10], [2, 20]], dtype=np.int32))
        c = a + b
        result = c.to_numpy()
        assert result[0, 1] == 11
        assert result[1, 1] == 23

    def test_add_i64(self):
        a = Pcf(np.array([[0, 1], [2, 3]], dtype=np.int64))
        b = Pcf(np.array([[0, 10], [2, 20]], dtype=np.int64))
        c = a + b
        result = c.to_numpy()
        assert result[0, 1] == 11
        assert result[1, 1] == 23

    def test_add_mismatched_int_float_raises(self):
        a = Pcf(np.array([[0, 1]], dtype=np.int32))
        b = Pcf(np.array([[0.0, 1.0]], dtype=np.float32))
        with pytest.raises(TypeError, match="Mismatched PCF types"):
            a + b

    def test_add_mismatched_i32_i64_raises(self):
        a = Pcf(np.array([[0, 1]], dtype=np.int32))
        b = Pcf(np.array([[0, 1]], dtype=np.int64))
        with pytest.raises(TypeError, match="Mismatched PCF types"):
            a + b

    def test_div_scalar_i32(self):
        f = Pcf(np.array([[0, 10], [2, 20]], dtype=np.int32))
        f = f / 2
        result = f.to_numpy()
        npt.assert_array_equal(result[:, 1], [5, 10])

    def test_div_truncates(self):
        f = Pcf(np.array([[0, 3], [2, 7]], dtype=np.int32))
        f = f / 2
        result = f.to_numpy()
        npt.assert_array_equal(result[:, 1], [1, 3])


class TestIntPcfStr:
    def test_str_i32(self):
        f = Pcf(np.array([[0, 1]], dtype=np.int32))
        s = str(f)
        assert "int32" in s
        assert "PCF" in s

    def test_str_i64(self):
        f = Pcf(np.array([[0, 1]], dtype=np.int64))
        s = str(f)
        assert "int64" in s


class TestIntPcfAstype:
    def test_i32_to_f32(self):
        f = Pcf(np.array([[0, 1], [2, 3]], dtype=np.int32))
        g = f.astype(sb.pcf32)
        assert g.vtype == sb.float32
        npt.assert_array_almost_equal(g.to_numpy(), [[0.0, 1.0], [2.0, 3.0]])

    def test_f32_to_i32(self):
        f = Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        g = f.astype(sb.pcf32i)
        assert g.vtype == sb.int32
        npt.assert_array_equal(g.to_numpy(), [[0, 1], [2, 3]])

    def test_i32_to_i64(self):
        f = Pcf(np.array([[0, 1], [2, 3]], dtype=np.int32))
        g = f.astype(sb.pcf64i)
        assert g.vtype == sb.int64


class TestIntPcfTensor:
    def test_pcf32i_tensor_create(self):
        t = sb.zeros((3,), dtype=sb.pcf32i)
        assert t.dtype == sb.pcf32i
        assert t.shape == (3,)

    def test_pcf64i_tensor_create(self):
        t = sb.zeros((3,), dtype=sb.pcf64i)
        assert t.dtype == sb.pcf64i
        assert t.shape == (3,)

    def test_tensor_set_get(self):
        t = sb.zeros((2,), dtype=sb.pcf32i)

        f = Pcf(np.array([[0, 10], [2, 20]], dtype=np.int32))
        t[0] = f
        g = t[0]
        assert isinstance(g, Pcf)
        npt.assert_array_equal(g.to_numpy(), f.to_numpy())


class TestIntPcfPickle:
    def test_pickle_roundtrip_i32(self):
        import pickle

        f = Pcf(np.array([[0, 1], [2, 3]], dtype=np.int32))
        data = pickle.dumps(f)
        g = pickle.loads(data)
        assert f == g

    def test_pickle_roundtrip_i64(self):
        import pickle

        f = Pcf(np.array([[0, 1], [2, 3]], dtype=np.int64))
        data = pickle.dumps(f)
        g = pickle.loads(data)
        assert f == g


class TestIntPcfIO:
    def test_save_load_pcf32i_tensor(self):
        from io import BytesIO

        t = sb.zeros((2,), dtype=sb.pcf32i)
        t[0] = Pcf(np.array([[0, 10], [2, 20]], dtype=np.int32))
        t[1] = Pcf(np.array([[0, 30], [4, 40]], dtype=np.int32))

        buf = BytesIO()
        sb.save(t, buf)
        buf.seek(0)
        loaded = sb.load(buf)

        assert isinstance(loaded, sb.IntPcfTensor)
        assert loaded.dtype == sb.pcf32i
        assert loaded.shape == (2,)
        npt.assert_array_equal(loaded[0].to_numpy(), t[0].to_numpy())
        npt.assert_array_equal(loaded[1].to_numpy(), t[1].to_numpy())

    def test_save_load_pcf64i_tensor(self):
        from io import BytesIO

        t = sb.zeros((2,), dtype=sb.pcf64i)
        t[0] = Pcf(np.array([[0, 10], [2, 20]], dtype=np.int64))
        t[1] = Pcf(np.array([[0, 30], [4, 40]], dtype=np.int64))

        buf = BytesIO()
        sb.save(t, buf)
        buf.seek(0)
        loaded = sb.load(buf)

        assert isinstance(loaded, sb.IntPcfTensor)
        assert loaded.dtype == sb.pcf64i
        assert loaded.shape == (2,)
        npt.assert_array_equal(loaded[0].to_numpy(), t[0].to_numpy())
        npt.assert_array_equal(loaded[1].to_numpy(), t[1].to_numpy())
