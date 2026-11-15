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


class TestCall:
    # Breakpoints (0, 0), (1, 2), (3, 1): slope 2, then -0.5.
    def make(self, dtype=sb.float64):
        return sb.Plf([[0, 0],
                       [1, 2],
                       [3, 1]], dtype=dtype)

    def test_at_breakpoints(self):
        f = self.make()

        assert [f(0.0), f(1.0), f(3.0)] == [0.0, 2.0, 1.0]

    def test_between_breakpoints(self):
        f = self.make()

        assert [f(0.5), f(2.0)] == [1.0, 1.5]

    def test_after_last_continues_sloped_segment(self):
        f = self.make()

        assert f(5.0) == 0.0

    def test_after_last_flat_segment_is_constant(self):
        f = sb.Plf([[0, 0],
                    [1, 2],
                    [3, 2]])

        assert f(10.0) == 2.0

    def test_int_scalar(self):
        f = self.make()

        assert f(2) == 1.5

    def test_2d_array_keeps_shape(self):
        f = self.make()

        result = f(np.array([[0.0, 0.5],
                             [2.0, 5.0]]))

        npt.assert_array_equal(result, [[0.0, 1.0],
                                        [1.5, 0.0]])

    def test_unsorted_array_keeps_order(self):
        f = self.make()

        result = f(np.array([5.0, 0.5, 2.0, 0.0]))

        npt.assert_array_equal(result, [0.0, 1.0, 1.5, 0.0])

    def test_transposed_array(self):
        f = self.make()
        t = np.array([[0.0, 0.5],
                      [2.0, 5.0]])

        result = f(t.T)

        npt.assert_array_equal(result, [[0.0, 1.5],
                                        [1.0, 0.0]])

    def test_reversed_array(self):
        f = self.make()
        t = np.array([0.0, 0.5, 2.0, 5.0])

        result = f(t[::-1])

        npt.assert_array_equal(result, [0.0, 1.5, 1.0, 0.0])

    def test_list(self):
        f = self.make()

        result = f([0.5, 2.0])

        assert isinstance(result, np.ndarray)
        npt.assert_array_equal(result, [1.0, 1.5])

    @pytest.mark.parametrize("dtype", [sb.float32, sb.float64])
    def test_float_tensor(self, dtype):
        f = self.make(dtype=dtype)
        t = sb.FloatTensor(np.array([0.5, 2.0, 5.0]), dtype=dtype)

        result = f(t)

        assert isinstance(result, sb.FloatTensor)
        assert result.dtype == dtype
        npt.assert_array_equal(np.asarray(result), [1.0, 1.5, 0.0])

    def test_negative_scalar_raises(self):
        f = self.make()

        with pytest.raises(ValueError):
            f(-1.0)

    def test_negative_in_array_raises(self):
        f = self.make()

        with pytest.raises(ValueError):
            f(np.array([1.0, -1.0]))


class TestIntegrate:
    def test_breakpoints_hold_running_integral(self):
        f = sb.Pcf([[0.0, 2.0],
                    [1.0, -1.0],
                    [3.0, -1.0]])

        F = f.integrate()

        npt.assert_array_equal(F.to_numpy(), [[0.0, 0.0],
                                              [1.0, 2.0],
                                              [3.0, 0.0]])

    def test_different_last_value_adds_breakpoint_one_after_small_last_time(self):
        f = sb.Pcf([[0.0, 2.0],
                    [0.5, 4.0]])

        F = f.integrate()

        npt.assert_array_equal(F.to_numpy(), [[0.0, 0.0],
                                              [0.5, 1.0],
                                              [1.5, 5.0]])

    def test_different_last_value_adds_breakpoint_last_time_after_large_last_time(self):
        f = sb.Pcf([[0.0, 2.0],
                    [1.0, -1.0],
                    [3.0, 0.5]])

        F = f.integrate()

        npt.assert_array_equal(F.to_numpy(), [[0.0, 0.0],
                                              [1.0, 2.0],
                                              [3.0, 0.0],
                                              [6.0, 1.5]])

    def test_single_breakpoint_gives_line_through_zero(self):
        f = sb.Pcf([[0.0, 3.0]])

        F = f.integrate()

        npt.assert_array_equal(F.to_numpy(), [[0.0, 0.0],
                                              [1.0, 3.0]])

    @pytest.mark.parametrize("pcf_dtype, expected", [
        (sb.pcf32, sb.float32),
        (sb.pcf64, sb.float64),
    ])
    def test_keeps_precision(self, pcf_dtype, expected):
        f = sb.Pcf([[0.0, 2.0],
                    [1.0, 2.0]], dtype=pcf_dtype)

        F = f.integrate()

        assert F.dtype is expected
        assert F.to_numpy().dtype == f.to_numpy().dtype

    def test_integer_pcf_raises(self):
        f = sb.Pcf([[0, 2],
                    [1, 3]], dtype=sb.pcf64i)

        with pytest.raises(TypeError):
            f.integrate()


class TestPlfTensor:
    @pytest.mark.parametrize("dtype, element_dtype", [
        (sb.plf32, sb.float32),
        (sb.plf64, sb.float64),
    ])
    def test_zeros_holds_zero_functions(self, dtype, element_dtype):
        Z = sb.zeros((2, 3), dtype=dtype)

        assert isinstance(Z, sb.PlfTensor)
        assert Z.dtype is dtype
        assert Z.shape == (2, 3)
        assert Z[1, 2] == sb.Plf([[0.0, 0.0]], dtype=element_dtype)
        assert Z[1, 2].dtype is element_dtype

    def test_set_and_get_element(self):
        f = sb.Plf([[0.0, 0.0],
                    [1.0, 2.0]])
        Z = sb.zeros((2, 3), dtype=sb.plf64)

        Z[1, 2] = f

        assert Z[1, 2] == f
        assert Z[0, 0] == sb.Plf([[0.0, 0.0]])

    @pytest.mark.parametrize("element_dtype, expected", [
        (sb.float32, sb.plf32),
        (sb.float64, sb.plf64),
    ])
    def test_tensor_infers_dtype(self, element_dtype, expected):
        f = sb.Plf([[0.0, 0.0],
                    [1.0, 2.0]], dtype=element_dtype)

        T = sb.tensor([f, f])

        assert isinstance(T, sb.PlfTensor)
        assert T.dtype is expected

    def test_from_nested_list(self):
        f = sb.Plf([[0.0, 0.0],
                    [1.0, 2.0]])
        g = sb.Plf([[0.0, 1.0],
                    [2.0, 1.0]])

        T = sb.PlfTensor([[f, g],
                          [g, f]])

        assert T.shape == (2, 2)
        assert T[0, 0] == f
        assert T[0, 1] == g
        assert T[1, 0] == g
        assert T[1, 1] == f

    @pytest.mark.parametrize("dtype, element_dtype", [
        (sb.plf32, sb.float32),
        (sb.plf64, sb.float64),
    ])
    def test_copy_keeps_dtype_and_elements(self, dtype, element_dtype):
        f = sb.Plf([[0.0, 0.0],
                    [1.0, 2.0]], dtype=element_dtype)
        Z = sb.zeros((2,), dtype=dtype)
        Z[1] = f

        C = Z.copy()

        assert isinstance(C, sb.PlfTensor)
        assert C.dtype is dtype
        assert C[0] == sb.Plf([[0.0, 0.0]], dtype=element_dtype)
        assert C[1] == f

    def test_copy_is_independent(self):
        f = sb.Plf([[0.0, 0.0],
                    [1.0, 2.0]])
        Z = sb.zeros((2,), dtype=sb.plf64)

        C = Z.copy()
        C[1] = f

        assert Z[1] == sb.Plf([[0.0, 0.0]])


class TestPlfTensorCall:
    @staticmethod
    def make(dtype=sb.float64):
        f = sb.Plf([[0.0, 0.0],
                    [1.0, 2.0],
                    [3.0, 1.0]], dtype=dtype)
        g = sb.Plf([[0.0, 1.0],
                    [2.0, 2.0]], dtype=dtype)
        return f, g

    def test_scalar_on_1d_tensor(self):
        f, g = self.make()
        T = sb.PlfTensor([f, g])

        result = T(0.5)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        npt.assert_array_equal(result, [f(0.5), g(0.5)])

    def test_scalar_on_2d_tensor(self):
        f, g = self.make()
        T = sb.PlfTensor([[f, g],
                          [g, f]])

        result = T(2.0)

        assert result.shape == (2, 2)
        npt.assert_array_equal(result, [[f(2.0), g(2.0)],
                                        [g(2.0), f(2.0)]])

    def test_array_on_1d_tensor(self):
        f, g = self.make()
        T = sb.PlfTensor([f, g])
        t = np.array([0.5, 2.0, 5.0])

        result = T(t)

        assert result.shape == (2, 3)
        npt.assert_array_equal(result, [f(t), g(t)])

    def test_array_on_2d_tensor_has_tensor_then_time_shape(self):
        f, g = self.make()
        T = sb.PlfTensor([[f, g],
                          [g, f]])
        t = np.array([[0.5, 2.0, 5.0],
                      [5.0, 2.0, 0.5]])

        result = T(t)

        assert result.shape == (2, 2, 2, 3)
        npt.assert_array_equal(result, [[f(t), g(t)],
                                        [g(t), f(t)]])

    def test_list(self):
        f, g = self.make()
        T = sb.PlfTensor([f, g])

        result = T([0.5, 2.0])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        npt.assert_array_equal(result, [f([0.5, 2.0]), g([0.5, 2.0])])

    @pytest.mark.parametrize("dtype", [sb.float32, sb.float64])
    def test_float_tensor(self, dtype):
        f, g = self.make(dtype=dtype)
        T = sb.PlfTensor([f, g])
        t = sb.FloatTensor(np.array([0.5, 2.0]), dtype=dtype)

        result = T(t)

        assert isinstance(result, sb.FloatTensor)
        assert result.dtype is dtype
        assert result.shape == (2, 2)
        npt.assert_array_equal(np.asarray(result), [np.asarray(f(t)), np.asarray(g(t))])

    def test_parallel_matches_serial(self):
        T = sb.zeros((20,), dtype=sb.plf64)
        for i in range(20):
            T[i] = sb.Plf([[0.0, float(i)],
                           [1.0, float(i + 1)],
                           [2.0, 0.0]])
        t = np.array([0.0, 0.5, 1.0, 1.5, 2.5])
        old = sb.system.get_parallel_eval_threshold()
        try:
            sb.system.set_parallel_eval_threshold(4)
            parallel = T(t)
            sb.system.set_parallel_eval_threshold(1000)
            serial = T(t)
        finally:
            sb.system.set_parallel_eval_threshold(old)

        npt.assert_array_equal(parallel, serial)
