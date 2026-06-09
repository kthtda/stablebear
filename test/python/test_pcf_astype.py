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

import stablebear as sb


def test_astype_pcf32_to_pcf64():
    f = sb.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    g = f.astype(sb.pcf64)

    assert g.vtype == sb.float64
    npt.assert_array_almost_equal(f.to_numpy(), g.to_numpy(), decimal=5)


def test_astype_pcf64_to_pcf32():
    f = sb.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64))
    g = f.astype(sb.pcf32)

    assert g.vtype == sb.float32
    npt.assert_array_almost_equal(f.to_numpy(), g.to_numpy(), decimal=5)


def test_astype_np_float32():
    f = sb.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64))
    g = f.astype(np.float32)

    assert g.vtype == sb.float32


def test_astype_np_float64():
    f = sb.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    g = f.astype(np.float64)

    assert g.vtype == sb.float64


def test_astype_same_dtype():
    f = sb.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    g = f.astype(sb.pcf32)

    assert g.vtype == sb.float32
    npt.assert_array_equal(f.to_numpy(), g.to_numpy())


def test_astype_returns_copy():
    f = sb.Pcf(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    g = f.astype(sb.pcf32)

    assert f is not g


def test_astype_rejects_string():
    f = sb.Pcf(np.array([[0.0, 1.0]], dtype=np.float32))

    with pytest.raises(TypeError):
        f.astype("float32")


def test_astype_rejects_int():
    f = sb.Pcf(np.array([[0.0, 1.0]], dtype=np.float32))

    with pytest.raises(TypeError):
        f.astype(int)
