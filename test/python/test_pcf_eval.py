import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


# PCF: f(t) = 1.0 on [0, 1), 2.0 on [1, 3), 0.5 on [3, inf)
def make_f32():
    return sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.5]], dtype=np.float32))


def make_f64():
    return sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.5]], dtype=np.float64))


class TestScalarEval:
    def test_f32_at_zero(self):
        f = make_f32()
        assert f(0.0) == pytest.approx(1.0)

    def test_f32_mid_interval(self):
        f = make_f32()
        assert f(0.5) == pytest.approx(1.0)
        assert f(1.5) == pytest.approx(2.0)
        assert f(5.0) == pytest.approx(0.5)

    def test_f32_at_breakpoints(self):
        f = make_f32()
        assert f(1.0) == pytest.approx(2.0)
        assert f(3.0) == pytest.approx(0.5)

    def test_f64_at_zero(self):
        f = make_f64()
        assert f(0.0) == pytest.approx(1.0)

    def test_f64_mid_interval(self):
        f = make_f64()
        assert f(0.5) == pytest.approx(1.0)
        assert f(1.5) == pytest.approx(2.0)
        assert f(5.0) == pytest.approx(0.5)

    def test_negative_time_raises(self):
        f = make_f32()
        with pytest.raises(Exception):
            f(-1.0)

    def test_int_argument(self):
        f = make_f32()
        assert f(0) == pytest.approx(1.0)
        assert f(2) == pytest.approx(2.0)


class TestNdarrayEval:
    def test_f32_1d(self):
        f = make_f32()
        t = np.array([0.0, 0.5, 1.0, 1.5, 3.0, 5.0], dtype=np.float32)
        result = f(t)
        npt.assert_array_almost_equal(result, [1.0, 1.0, 2.0, 2.0, 0.5, 0.5])
        assert result.dtype == np.float32

    def test_f64_1d(self):
        f = make_f64()
        t = np.array([0.0, 0.5, 1.0, 1.5, 3.0, 5.0], dtype=np.float64)
        result = f(t)
        npt.assert_array_almost_equal(result, [1.0, 1.0, 2.0, 2.0, 0.5, 0.5])
        assert result.dtype == np.float64

    def test_2d(self):
        f = make_f32()
        t = np.array([[0.0, 1.5], [3.0, 0.5]], dtype=np.float32)
        result = f(t)
        assert result.shape == (2, 2)
        npt.assert_array_almost_equal(result, [[1.0, 2.0], [0.5, 1.0]])

    def test_unsorted_times(self):
        f = make_f32()
        t = np.array([5.0, 0.0, 1.5, 3.0], dtype=np.float32)
        result = f(t)
        npt.assert_array_almost_equal(result, [0.5, 1.0, 2.0, 0.5])

    def test_negative_time_in_array_raises(self):
        f = make_f32()
        t = np.array([0.0, -1.0, 1.0], dtype=np.float32)
        with pytest.raises(Exception):
            f(t)


class TestListEval:
    def test_list_of_floats(self):
        f = make_f32()
        result = f([0.0, 0.5, 1.5, 5.0])
        npt.assert_array_almost_equal(result, [1.0, 1.0, 2.0, 0.5])


class TestTensorEval:
    def test_float32_tensor(self):
        f = make_f32()
        t = sb.FloatTensor(np.array([0.0, 1.5, 5.0], dtype=np.float32))
        result = f(t)
        assert isinstance(result, sb.FloatTensor)
        assert result.dtype == sb.float32
        npt.assert_array_almost_equal(np.asarray(result), [1.0, 2.0, 0.5])

    def test_float64_tensor(self):
        f = make_f64()
        t = sb.FloatTensor(np.array([0.0, 1.5, 5.0], dtype=np.float64))
        result = f(t)
        assert isinstance(result, sb.FloatTensor)
        assert result.dtype == sb.float64
        npt.assert_array_almost_equal(np.asarray(result), [1.0, 2.0, 0.5])
