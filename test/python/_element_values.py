"""Two distinct elements of every tensor element type, for parametrized tests.

Object elements also differ in size: clouds of 1 and 2 points, 2x2 and 3x3
matrices, barcodes with 1 and 2 bars, and PCFs with 2 and 3 breakpoints.
"""

import numpy as np
import pytest

import stablebear as sb
from stablebear.persistence import Barcode, BarcodeTensor


def _pcfs(np_dtype):
    return sb.tensor([
        sb.Pcf(np.array([[0, 1], [1, 0]], dtype=np_dtype)),
        sb.Pcf(np.array([[0, 2], [1, 1], [2, 0]], dtype=np_dtype)),
    ])


def _point_clouds(dtype, np_dtype):
    # Clouds of 1 and 2 points.
    return sb.PointCloudTensor([
        np.array([[0.0, 0.0]], dtype=np_dtype),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np_dtype),
    ], dtype=dtype)


def _distance_matrices(dtype, np_dtype):
    tensor = sb.zeros((2,), dtype=dtype)
    tensor[0] = sb.DistanceMatrix(np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=np_dtype))
    tensor[1] = sb.DistanceMatrix(np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 3.0],
        [2.0, 3.0, 0.0],
    ], dtype=np_dtype))
    return tensor


def _symmetric_matrices(dtype, np_dtype):
    tensor = sb.zeros((2,), dtype=dtype)
    tensor[0] = sb.SymmetricMatrix(np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ], dtype=np_dtype))
    tensor[1] = sb.SymmetricMatrix(np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0],
    ], dtype=np_dtype))
    return tensor


def _barcodes(np_dtype):
    # Barcodes with 1 and 2 bars.
    return BarcodeTensor([
        Barcode(np.array([[0.0, 1.0]], dtype=np_dtype)),
        Barcode(np.array([[0.0, 2.0], [0.5, 1.5]], dtype=np_dtype)),
    ])


# Each parameter makes a 1-D tensor of two elements; the ids are the dtypes.
TWO_ELEMENTS_OF_EVERY_DTYPE = [
    pytest.param(lambda: sb.tensor(np.array([True, False])), id="boolean"),
    pytest.param(lambda: sb.tensor(np.array([1.0, 2.0], dtype=np.float32)), id="float32"),
    pytest.param(lambda: sb.tensor(np.array([1.0, 2.0], dtype=np.float64)), id="float64"),
    pytest.param(lambda: sb.tensor(np.array([1, 2], dtype=np.int32)), id="int32"),
    pytest.param(lambda: sb.tensor(np.array([1, 2], dtype=np.int64)), id="int64"),
    pytest.param(lambda: sb.tensor(np.array([1, 2], dtype=np.uint32)), id="uint32"),
    pytest.param(lambda: sb.tensor(np.array([1, 2], dtype=np.uint64)), id="uint64"),
    pytest.param(lambda: _pcfs(np.float32), id="pcf32"),
    pytest.param(lambda: _pcfs(np.float64), id="pcf64"),
    pytest.param(lambda: _pcfs(np.int32), id="pcf32i"),
    pytest.param(lambda: _pcfs(np.int64), id="pcf64i"),
    pytest.param(lambda: _point_clouds(sb.pcloud32, np.float32), id="pcloud32"),
    pytest.param(lambda: _point_clouds(sb.pcloud64, np.float64), id="pcloud64"),
    pytest.param(lambda: _distance_matrices(sb.distmat32, np.float32), id="distmat32"),
    pytest.param(lambda: _distance_matrices(sb.distmat64, np.float64), id="distmat64"),
    pytest.param(lambda: _symmetric_matrices(sb.symmat32, np.float32), id="symmat32"),
    pytest.param(lambda: _symmetric_matrices(sb.symmat64, np.float64), id="symmat64"),
    pytest.param(lambda: _barcodes(np.float32), id="barcode32"),
    pytest.param(lambda: _barcodes(np.float64), id="barcode64"),
]
