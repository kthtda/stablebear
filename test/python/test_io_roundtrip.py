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

import stablebear as sb


def _roundtrip(tensor):
    buf = io.BytesIO()
    sb.save(tensor, buf)
    buf.seek(0)
    return sb.load(buf)


def _assert_roundtrip(original):
    restored = _roundtrip(original)
    assert type(restored) is type(original)
    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    assert original.array_equal(restored)


# --- Float tensors ---


def test_float32_tensor_roundtrip():
    _assert_roundtrip(sb.FloatTensor(np.array([[1.0, 2.5], [3.0, -4.5]], dtype=np.float32)))


def test_float64_tensor_roundtrip():
    _assert_roundtrip(sb.FloatTensor(np.array([[1.0, 2.5], [3.0, -4.5]], dtype=np.float64)))


def test_float32_tensor_1d():
    _assert_roundtrip(sb.FloatTensor(np.array([1.5, 2.5, 3.5], dtype=np.float32)))


def test_float64_tensor_3d():
    _assert_roundtrip(sb.FloatTensor(np.random.randn(2, 3, 4).astype(np.float64)))


def test_float32_tensor_scalar():
    _assert_roundtrip(sb.FloatTensor(np.array([42.0], dtype=np.float32)))


# --- PCF tensors ---


def test_pcf32_tensor_roundtrip():
    _assert_roundtrip(sb.random.noisy_sin((3, 4), dtype=sb.pcf32))


def test_pcf64_tensor_roundtrip():
    _assert_roundtrip(sb.random.noisy_sin((3, 4), dtype=sb.pcf64))


def test_pcf32_tensor_1d():
    _assert_roundtrip(sb.random.noisy_cos((5,), dtype=sb.pcf32))


def test_pcf32_tensor_zeros():
    _assert_roundtrip(sb.zeros((2, 3), dtype=sb.pcf32))


def test_pcf64_tensor_zeros():
    _assert_roundtrip(sb.zeros((2, 3), dtype=sb.pcf64))


# --- PointCloud tensors ---


def test_point_cloud32_tensor_roundtrip():
    original = sb.zeros((3,), dtype=sb.pcloud32)
    original[0] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    original[1] = np.array([[5.0, 6.0]], dtype=np.float32)
    _assert_roundtrip(original)


def test_point_cloud64_tensor_roundtrip():
    original = sb.zeros((3,), dtype=sb.pcloud64)
    original[0] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    original[1] = np.array([[5.0, 6.0]], dtype=np.float64)
    _assert_roundtrip(original)


def test_point_cloud32_tensor_empty():
    _assert_roundtrip(sb.zeros((2,), dtype=sb.pcloud32))


# --- Barcode tensors ---


def test_barcode32_tensor_roundtrip():
    original = sb.zeros((2,), dtype=sb.barcode32)
    original[0] = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float32)
    _assert_roundtrip(original)


def test_barcode64_tensor_roundtrip():
    original = sb.zeros((2,), dtype=sb.barcode64)
    original[0] = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64)
    _assert_roundtrip(original)


def test_barcode32_tensor_empty():
    _assert_roundtrip(sb.zeros((3,), dtype=sb.barcode32))
