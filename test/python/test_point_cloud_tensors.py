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

import stablebear as sb


def test_can_create_point_clouds():
    X = sb.zeros((2,), dtype=sb.pcloud64)

    assert isinstance(X, sb.PointCloudTensor)
    assert X.dtype == sb.pcloud64

    X[0] = np.random.randn(10, 2)
    X[1] = np.random.randn(20, 2)

    assert X[0].shape == (10, 2)
    assert X[1].shape == (20, 2)

    Y = sb.zeros((2, 3), dtype=sb.pcloud32)

    assert isinstance(Y, sb.PointCloudTensor)
    assert Y.dtype == sb.pcloud32

    Y[0, 0] = np.random.randn(30, 2, 20)
    Y[1, 1] = np.random.randn(40, 15, 10)

    assert Y[0, 0].shape == (30, 2, 20)
    assert Y[1, 1].shape == (40, 15, 10)


def test_stored_is_same_as_numpy():
    shape = (10, 20, 30)
    pclouds = sb.zeros(shape, dtype=sb.pcloud64)
    X = np.random.randn(10, 2).astype(np.float64)

    pclouds[0, 1, 2] = X
    assert pclouds[0, 1, 2].array_equal(X)

    pclouds = sb.zeros(shape, dtype=sb.pcloud32)
    X = np.random.randn(10, 2).astype(np.float32)

    pclouds[0, 1, 2] = X
    assert pclouds[0, 1, 2].array_equal(X)
