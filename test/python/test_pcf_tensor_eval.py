#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import numpy.testing as npt
import pytest

import masspcf as mpcf


# f(t) = 1 on [0,1), 2 on [1,3), 5 on [3,inf)
def make_f(np_dtype):
    return mpcf.Pcf(np.array([[0, 1], [1, 2], [3, 5]], dtype=np_dtype))


# g(t) = 3 on [0,2), 1 on [2,inf)
def make_g(np_dtype):
    return mpcf.Pcf(np.array([[0, 3], [2, 1]], dtype=np_dtype))


def make_1d_tensor(pcf_dtype, np_dtype):
    """Create a 1D PcfTensor [f, g]."""
    X = mpcf.zeros((2,), dtype=pcf_dtype)
    X[0] = make_f(np_dtype)
    X[1] = make_g(np_dtype)
    return X


def make_2d_tensor(pcf_dtype, np_dtype):
    """Create a 2D PcfTensor [[f, g], [g, f]]."""
    X = mpcf.zeros((2, 2), dtype=pcf_dtype)
    X[0, 0] = make_f(np_dtype)
    X[0, 1] = make_g(np_dtype)
    X[1, 0] = make_g(np_dtype)
    X[1, 1] = make_f(np_dtype)
    return X


pcf_params = [
    pytest.param((mpcf.pcf32, np.float32), id="pcf32"),
    pytest.param((mpcf.pcf64, np.float64), id="pcf64"),
    pytest.param((mpcf.pcf32i, np.int32), id="pcf32i"),
    pytest.param((mpcf.pcf64i, np.int64), id="pcf64i"),
]


@pytest.fixture(params=pcf_params)
def dtypes(request):
    return request.param


class TestPcfTensorEvalScalar:
    """Scalar t -> output shape == tensor shape."""

    def test_1d(self, dtypes):
        pcf_dtype, np_dtype = dtypes
        X = make_1d_tensor(pcf_dtype, np_dtype)
        result = X(0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result.dtype == np_dtype
        # f(0) = 1, g(0) = 3
        npt.assert_array_almost_equal(result, [1, 3])

    def test_2d(self, dtypes):
        pcf_dtype, np_dtype = dtypes
        X = make_2d_tensor(pcf_dtype, np_dtype)
        result = X(1)
        assert result.shape == (2, 2)
        assert result.dtype == np_dtype
        # f(1) = 2, g(1) = 3
        npt.assert_array_almost_equal(result, [[2, 3], [3, 2]])

    def test_int_argument(self, dtypes):
        pcf_dtype, np_dtype = dtypes
        X = make_1d_tensor(pcf_dtype, np_dtype)
        result = X(2)
        assert result.shape == (2,)
        # f(2) = 2, g(2) = 1
        npt.assert_array_almost_equal(result, [2, 1])


class TestPcfTensorEvalArray:
    """Array t -> output shape == tensor shape + t shape."""

    def test_1d_tensor_1d_times(self, dtypes):
        pcf_dtype, np_dtype = dtypes
        X = make_1d_tensor(pcf_dtype, np_dtype)
        t = np.array([0, 1, 5], dtype=np_dtype)
        result = X(t)
        assert result.shape == (2, 3)
        assert result.dtype == np_dtype
        # f: [1, 2, 5], g: [3, 3, 1]
        npt.assert_array_almost_equal(result, [[1, 2, 5], [3, 3, 1]])

    def test_2d_tensor_1d_times(self, dtypes):
        pcf_dtype, np_dtype = dtypes
        X = make_2d_tensor(pcf_dtype, np_dtype)
        t = np.array([0, 5], dtype=np_dtype)
        result = X(t)
        assert result.shape == (2, 2, 2)
        # f: [1, 5], g: [3, 1]
        expected = np.array(
            [[[1, 5], [3, 1]], [[3, 1], [1, 5]]], dtype=np_dtype
        )
        npt.assert_array_almost_equal(result, expected)

    def test_1d_tensor_2d_times(self, dtypes):
        pcf_dtype, np_dtype = dtypes
        X = make_1d_tensor(pcf_dtype, np_dtype)
        t = np.array([[0, 1], [5, 2]], dtype=np_dtype)
        result = X(t)
        assert result.shape == (2, 2, 2)
        # f: [[1, 2], [5, 2]], g: [[3, 3], [1, 1]]
        expected = np.array(
            [[[1, 2], [5, 2]], [[3, 3], [1, 1]]], dtype=np_dtype
        )
        npt.assert_array_almost_equal(result, expected)

    def test_list_of_times(self, dtypes):
        pcf_dtype, np_dtype = dtypes
        X = make_1d_tensor(pcf_dtype, np_dtype)
        result = X([0, 1, 5])
        assert result.shape == (2, 3)
        npt.assert_array_almost_equal(result, [[1, 2, 5], [3, 3, 1]])

    @pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
    def test_float_tensor_input(self, dtypes, float_dtype):
        pcf_dtype, np_dtype = dtypes
        X = make_1d_tensor(pcf_dtype, np_dtype)
        t = mpcf.FloatTensor(np.array([0, 5], dtype=float_dtype))
        result = X(t)
        assert result.shape == (2, 2)
        npt.assert_array_almost_equal(result, [[1, 5], [3, 1]])


class TestPcfTensorEvalParallel:
    """Verify that parallel tensor_eval produces correct results.

    Uses set_parallel_eval_threshold to force the parallel path on
    tensors large enough to exercise it but small enough to run fast.
    """

    @pytest.fixture(autouse=True)
    def low_threshold(self):
        """Temporarily lower the parallel threshold for these tests."""
        old = mpcf.system.get_parallel_eval_threshold()
        mpcf.system.set_parallel_eval_threshold(4)
        yield
        mpcf.system.set_parallel_eval_threshold(old)

    def _make_tensor(self, n, np_dtype, pcf_dtype=mpcf.pcf64):
        X = mpcf.zeros((n,), dtype=pcf_dtype)
        for i in range(n):
            X[i] = mpcf.Pcf(np.array([[0, i + 1], [1, i + 2]], dtype=np_dtype))
        return X

    def test_scalar_eval_parallel(self):
        X = self._make_tensor(10, np.float64)
        result = X(0.5)
        assert result.shape == (10,)
        for i in range(10):
            assert result[i] == pytest.approx(i + 1)

    def test_array_eval_parallel(self):
        X = self._make_tensor(10, np.float64)
        t = np.array([0.5, 1.5], dtype=np.float64)
        result = X(t)
        assert result.shape == (10, 2)
        for i in range(10):
            assert result[i, 0] == pytest.approx(i + 1)
            assert result[i, 1] == pytest.approx(i + 2)

    def test_2d_tensor_scalar_parallel(self):
        X = mpcf.zeros((5, 4), dtype=mpcf.pcf64)
        for i in range(5):
            for j in range(4):
                X[i, j] = mpcf.Pcf(np.array(
                    [[0, i * 4 + j], [1, 0]], dtype=np.float64))
        result = X(0.5)
        assert result.shape == (5, 4)
        for i in range(5):
            for j in range(4):
                assert result[i, j] == pytest.approx(i * 4 + j)

    def test_matches_sequential(self):
        """Parallel and sequential results must be identical."""
        X = self._make_tensor(20, np.float64)
        t = np.array([0.0, 0.5, 1.0, 1.5, 2.5], dtype=np.float64)

        # Parallel (threshold=4, tensor has 20 elements)
        parallel_result = X(t)

        # Sequential (raise threshold above tensor size)
        mpcf.system.set_parallel_eval_threshold(100)
        sequential_result = X(t)

        npt.assert_array_equal(parallel_result, sequential_result)

    def test_float32(self):
        X = mpcf.zeros((10,), dtype=mpcf.pcf32)
        for i in range(10):
            X[i] = mpcf.Pcf(np.array([[0, i + 1], [1, i + 2]], dtype=np.float32))
        result = X(np.float32(0.5))
        assert result.shape == (10,)
        assert result.dtype == np.float32
        for i in range(10):
            assert result[i] == pytest.approx(i + 1)


class TestPcfTensorEvalErrors:
    def test_negative_time_raises(self):
        X = make_1d_tensor(mpcf.pcf32, np.float32)
        with pytest.raises(Exception):
            X(-1.0)

    def test_negative_time_in_array_raises(self):
        X = make_1d_tensor(mpcf.pcf32, np.float32)
        with pytest.raises(Exception):
            X(np.array([0.5, -1.0], dtype=np.float32))
