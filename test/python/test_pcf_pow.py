import numpy as np
import numpy.testing as npt
import pytest

from stablebear.functional import Pcf


@pytest.fixture(params=[np.float32, np.float64])
def dtype(request):
    return request.param


class TestPcfPowSquare:
    def test_pow_2(self, dtype):
        f = Pcf(np.array([[0.0, 2.0], [1.0, 3.0], [3.0, -4.0]], dtype=dtype))
        g = f ** 2
        expected = Pcf(np.array([[0.0, 4.0], [1.0, 9.0], [3.0, 16.0]], dtype=dtype))
        assert g == expected


class TestPcfPowGeneral:
    def test_pow_half(self, dtype):
        f = Pcf(np.array([[0.0, 4.0], [1.0, 9.0], [3.0, 16.0]], dtype=dtype))
        g = f ** 0.5
        expected = Pcf(np.array([[0.0, 2.0], [1.0, 3.0], [3.0, 4.0]], dtype=dtype))
        assert g == expected

    def test_pow_3(self, dtype):
        f = Pcf(np.array([[0.0, 2.0], [1.0, 3.0]], dtype=dtype))
        g = f ** 3.0
        expected = Pcf(np.array([[0.0, 8.0], [1.0, 27.0]], dtype=dtype))
        assert g == expected

    def test_pow_negative_exponent(self, dtype):
        f = Pcf(np.array([[0.0, 2.0], [1.0, 4.0]], dtype=dtype))
        g = f ** -1.0
        expected = Pcf(np.array([[0.0, 0.5], [1.0, 0.25]], dtype=dtype))
        assert g == expected

    def test_pow_negative_base_fractional_exp_warns(self, dtype):
        f = Pcf(np.array([[0.0, -2.0], [1.0, 4.0]], dtype=dtype))
        with pytest.warns(RuntimeWarning):
            g = f ** 0.5
        assert np.isnan(g.to_numpy()[0, 1])
        npt.assert_almost_equal(g.to_numpy()[1, 1], 2.0)

    def test_pow_zero_base_negative_exp_warns(self, dtype):
        f = Pcf(np.array([[0.0, 0.0], [1.0, 2.0]], dtype=dtype))
        with pytest.warns(RuntimeWarning):
            g = f ** -1.0
        assert np.isinf(g.to_numpy()[0, 1])
        npt.assert_almost_equal(g.to_numpy()[1, 1], 0.5)

    def test_pow_does_not_mutate(self, dtype):
        f = Pcf(np.array([[0.0, 2.0], [1.0, 3.0]], dtype=dtype))
        original = Pcf(np.array([[0.0, 2.0], [1.0, 3.0]], dtype=dtype))
        _ = f ** 2
        assert f == original

    def test_ipow(self, dtype):
        f = Pcf(np.array([[0.0, 2.0], [1.0, 3.0]], dtype=dtype))
        f **= 2
        expected = Pcf(np.array([[0.0, 4.0], [1.0, 9.0]], dtype=dtype))
        assert f == expected
