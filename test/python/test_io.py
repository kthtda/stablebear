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

import io

import numpy as np
import pytest

import stablebear as sb


def test_float32_tensor_roundtrip():
    original = sb.FloatTensor(np.random.randn(2, 3))

    buf = io.BytesIO()
    sb.save(original, buf)

    buf.seek(0)  # rewind before reading!
    restored = sb.load(buf)

    assert original.array_equal(restored)


def _make_symmetric_matrix(n, dtype):
    mat = sb.SymmetricMatrix(n, dtype=dtype)
    for i in range(n):
        for j in range(i + 1):
            mat[i, j] = float(i * n + j)
    return mat


@pytest.mark.parametrize("symmat_dtype, scalar_dtype", [
    (sb.symmat32, sb.float32),
    (sb.symmat64, sb.float64),
])
def test_symmetric_matrix_tensor_roundtrip(symmat_dtype, scalar_dtype):
    T = sb.zeros((2,), dtype=symmat_dtype)
    T[0] = _make_symmetric_matrix(3, scalar_dtype)
    T[1] = _make_symmetric_matrix(4, scalar_dtype)

    buf = io.BytesIO()
    sb.save(T, buf)

    buf.seek(0)
    restored = sb.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    for i in range(T.shape[0]):
        np.testing.assert_array_equal(T[i].to_dense(), restored[i].to_dense())


@pytest.mark.parametrize("symmat_dtype, scalar_dtype", [
    (sb.symmat32, sb.float32),
    (sb.symmat64, sb.float64),
])
def test_symmetric_matrix_tensor_roundtrip_empty(symmat_dtype, scalar_dtype):
    T = sb.zeros((1,), dtype=symmat_dtype)
    T[0] = sb.SymmetricMatrix(0, dtype=scalar_dtype)

    buf = io.BytesIO()
    sb.save(T, buf)

    buf.seek(0)
    restored = sb.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    assert restored[0].size == 0


def _make_distance_matrix(n, dtype):
    mat = sb.DistanceMatrix(n, dtype=dtype)
    for i in range(n):
        for j in range(i):
            mat[i, j] = float(i * n + j + 1)
    return mat


@pytest.mark.parametrize("distmat_dtype, scalar_dtype", [
    (sb.distmat32, sb.float32),
    (sb.distmat64, sb.float64),
])
def test_distance_matrix_tensor_roundtrip(distmat_dtype, scalar_dtype):
    T = sb.zeros((2,), dtype=distmat_dtype)
    T[0] = _make_distance_matrix(3, scalar_dtype)
    T[1] = _make_distance_matrix(4, scalar_dtype)

    buf = io.BytesIO()
    sb.save(T, buf)

    buf.seek(0)
    restored = sb.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    for i in range(T.shape[0]):
        np.testing.assert_array_equal(T[i].to_dense(), restored[i].to_dense())


@pytest.mark.parametrize("distmat_dtype, scalar_dtype", [
    (sb.distmat32, sb.float32),
    (sb.distmat64, sb.float64),
])
def test_distance_matrix_tensor_roundtrip_empty(distmat_dtype, scalar_dtype):
    T = sb.zeros((1,), dtype=distmat_dtype)
    T[0] = sb.DistanceMatrix(0, dtype=scalar_dtype)

    buf = io.BytesIO()
    sb.save(T, buf)

    buf.seek(0)
    restored = sb.load(buf)

    assert type(restored) is type(T)
    assert restored.shape == T.shape
    assert restored[0].size == 0
