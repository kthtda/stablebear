import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


class TestConstruction:
    def test_array_gives_breakpoints(self):
        arr = np.array([[0.0, 0.0],
                        [1.0, 2.0],
                        [3.0, 1.0]])

        f = sb.Plf(arr)

        npt.assert_array_equal(f.to_numpy(), arr)

    def test_list_gives_breakpoints(self):
        f = sb.Plf([[0, 0],
                    [1, 2],
                    [3, 1]])

        npt.assert_array_equal(f.to_numpy(), [[0, 0],
                                              [1, 2],
                                              [3, 1]])

    def test_to_numpy_round_trip(self):
        f = sb.Plf([[0, 0],
                    [1, 2],
                    [3, 1]])

        assert sb.Plf(f.to_numpy()) == f

    def test_copy(self):
        f = sb.Plf([[0, 0],
                    [1, 2],
                    [3, 1]], dtype=sb.float32)

        g = sb.Plf(f)

        assert g == f
        assert g.dtype is sb.float32


class TestDtype:
    @pytest.mark.parametrize("np_dtype, expected", [
        (np.float32, sb.float32),
        (np.float64, sb.float64),
        (np.int32, sb.float64),
    ])
    def test_inferred_from_array(self, np_dtype, expected):
        f = sb.Plf(np.array([[0, 0],
                             [1, 2]], dtype=np_dtype))

        assert f.dtype is expected

    def test_list_gives_float64(self):
        f = sb.Plf([[0, 0],
                    [1, 2]])

        assert f.dtype is sb.float64

    @pytest.mark.parametrize("np_dtype", [np.float32, np.float64, np.int32])
    @pytest.mark.parametrize("dtype, expected_np_dtype", [
        (sb.float32, np.float32),
        (sb.float64, np.float64),
    ])
    def test_dtype_overrides_array(self, np_dtype, dtype, expected_np_dtype):
        f = sb.Plf(np.array([[0, 0],
                             [1, 2]], dtype=np_dtype), dtype=dtype)

        assert f.dtype is dtype
        assert f.to_numpy().dtype == expected_np_dtype

    def test_dtype_overrides_list(self):
        f = sb.Plf([[0, 0],
                    [1, 2]], dtype=sb.float32)

        assert f.dtype is sb.float32
        assert f.to_numpy().dtype == np.float32

    @pytest.mark.parametrize("source_dtype, dtype, expected_np_dtype", [
        (sb.float64, sb.float32, np.float32),
        (sb.float32, sb.float64, np.float64),
    ])
    def test_dtype_overrides_plf(self, source_dtype, dtype, expected_np_dtype):
        f = sb.Plf([[0, 0],
                    [1, 2]], dtype=source_dtype)

        g = sb.Plf(f, dtype=dtype)

        assert g.dtype is dtype
        assert g.to_numpy().dtype == expected_np_dtype

    def test_invalid_dtype_raises(self):
        with pytest.raises(TypeError):
            sb.Plf([[0, 0],
                    [1, 2]], dtype=sb.int64)

    def test_non_real_array_raises(self):
        with pytest.raises(TypeError):
            sb.Plf(np.array([[0, 0],
                             [1, 2]], dtype=np.complex128))

    @pytest.mark.parametrize("source_dtype, dtype", [
        (sb.float64, sb.float32),
        (sb.float32, sb.float64),
    ])
    def test_astype(self, source_dtype, dtype):
        f = sb.Plf([[0, 0],
                    [1, 2]], dtype=source_dtype)

        g = f.astype(dtype)

        assert g.dtype is dtype
        assert g == f


@pytest.mark.parametrize("breakpoints", [
    pytest.param(np.empty((0, 2)), id="no breakpoints"),
    pytest.param([[1, 10],
                  [2, 20]], id="nonzero first time"),
    pytest.param([[0, 0],
                  [2, 20],
                  [2, 30]], id="repeated time"),
    pytest.param([[0, 0],
                  [2, 20],
                  [1, 10]], id="decreasing time"),
    pytest.param([[0, 0],
                  [np.inf, 20]], id="infinite time"),
    pytest.param([0, 1, 2], id="not (n, 2)"),
])
@pytest.mark.parametrize("dtype", [sb.float32, sb.float64])
def test_invalid_breakpoints_raise(breakpoints, dtype):
    with pytest.raises(ValueError):
        sb.Plf(breakpoints, dtype=dtype)


class TestEquality:
    def test_equal_breakpoints(self):
        f = sb.Plf([[0, 0],
                    [1, 2]])
        g = sb.Plf([[0, 0],
                    [1, 2]])

        assert f == g

    def test_different_values(self):
        f = sb.Plf([[0, 0],
                    [1, 2]])
        g = sb.Plf([[0, 0],
                    [1, 3]])

        assert f != g
