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

from . import _sb_cpp as cpp
from ._tensor_base import Shape, ShapeLike
from .tensor import (
    BoolTensor,
    FloatTensor,
    IntPcfTensor,
    IntTensor,
    PcfTensor,
    PointCloudTensor,
)
from .typing import (
    Dtype,
    _assert_valid_dtype,
    barcode32,
    barcode64,
    boolean,
    distmat32,
    distmat64,
    float32,
    float64,
    int32,
    int64,
    pcf32,
    pcf32i,
    pcf64,
    pcf64i,
    pcloud32,
    pcloud64,
    symmat32,
    symmat64,
    uint32,
    uint64,
)

cpp_p = cpp.persistence


def zeros(shape: ShapeLike, dtype: Dtype = pcf32):
    """
    Creates a new `Tensor` of the specified `shape` and `dtype` whose entries are "zero." What "zero" means depends on the `dtype`:

    `dtype=pcf32/64`: A PCF that takes the value 0 for all times.
    `dtype=pcf32i/64i`: An integer PCF that takes the value 0 for all times.
    `dtype=float32/float64`: The number 0.
    `dtype=pcloud32/64`: An empty point cloud.
    `dtype=barcode32/64`: An empty barcode.
    `dtype=symmat32/64`: A 0×0 symmetric matrix.
    `dtype=distmat32/64`: A 0×0 distance matrix.

    Parameters
    ----------
    shape : ShapeLike
        Shape of the returned tensor
    dtype
        The data type of the elements

    Returns
    -------
    Tensor
        The newly created tensor
    """

    from .persistence.ph_tensor import BarcodeTensor

    if not isinstance(shape, Shape):
        shape = Shape(shape)  # If passed as, e.g., tuple of ints

    _assert_valid_dtype(
        dtype,
        [
            pcf32,
            pcf64,
            pcf32i,
            pcf64i,
            float32,
            float64,
            int32,
            int64,
            uint32,
            uint64,
            boolean,
            pcloud32,
            pcloud64,
            barcode32,
            barcode64,
            distmat32,
            distmat64,
            symmat32,
            symmat64,
        ],
    )

    if dtype == boolean:
        return BoolTensor(cpp.BoolTensor(shape, False))
    elif dtype == pcf32:
        return PcfTensor(cpp.Pcf32Tensor(shape))
    elif dtype == pcf64:
        return PcfTensor(cpp.Pcf64Tensor(shape))
    elif dtype == pcf32i:
        return IntPcfTensor(cpp.Pcf32iTensor(shape))
    elif dtype == pcf64i:
        return IntPcfTensor(cpp.Pcf64iTensor(shape))
    elif dtype == float32:
        return FloatTensor(cpp.Float32Tensor(shape, 0.0))
    elif dtype == float64:
        return FloatTensor(cpp.Float64Tensor(shape, 0.0))
    elif dtype == int32:
        return IntTensor(cpp.Int32Tensor(shape, 0))
    elif dtype == int64:
        return IntTensor(cpp.Int64Tensor(shape, 0))
    elif dtype == uint32:
        return IntTensor(cpp.Uint32Tensor(shape, 0))
    elif dtype == uint64:
        return IntTensor(cpp.Uint64Tensor(shape, 0))
    elif dtype == pcloud32:
        return PointCloudTensor(cpp.PointCloud32Tensor(shape))
    elif dtype == pcloud64:
        return PointCloudTensor(cpp.PointCloud64Tensor(shape))
    elif dtype == barcode32:
        return BarcodeTensor(cpp_p.Barcode32Tensor(shape))
    elif dtype == barcode64:
        return BarcodeTensor(cpp_p.Barcode64Tensor(shape))
    elif dtype == distmat32:
        from .distance_matrix import DistanceMatrixTensor
        return DistanceMatrixTensor(cpp.DistanceMatrix32Tensor(shape))
    elif dtype == distmat64:
        from .distance_matrix import DistanceMatrixTensor
        return DistanceMatrixTensor(cpp.DistanceMatrix64Tensor(shape))
    elif dtype == symmat32:
        from .symmetric_matrix import SymmetricMatrixTensor
        return SymmetricMatrixTensor(cpp.SymmetricMatrix32Tensor(shape))
    elif dtype == symmat64:
        from .symmetric_matrix import SymmetricMatrixTensor
        return SymmetricMatrixTensor(cpp.SymmetricMatrix64Tensor(shape))
    else:
        raise NotImplementedError("This dtype has not been implemented.")


def concatenate(tensors, axis=0):
    """Concatenate tensors along an existing axis (outer indexing)."""
    if not tensors:
        raise ValueError("need at least one tensor to concatenate")
    cpp_tensors = [t._data for t in tensors]
    result = type(cpp_tensors[0]).concatenate(cpp_tensors, axis)
    return tensors[0]._to_py_tensor(result)


def stack(tensors, axis=0):
    """Stack tensors along a new axis. All tensors must have the same shape."""
    if not tensors:
        raise ValueError("need at least one tensor to stack")
    cpp_tensors = [t._data for t in tensors]
    result = type(cpp_tensors[0]).stack(cpp_tensors, axis)
    return tensors[0]._to_py_tensor(result)


def split(tensor, indices_or_sections, axis=0):
    """Split a tensor into sub-tensors along an axis.

    Parameters
    ----------
    tensor : Tensor
        The tensor to split.
    indices_or_sections : int or list of int
        If an int, the tensor is split into that many equal parts.
        If a list, it gives the indices where splits occur.
    axis : int
        The axis along which to split (default 0).

    Returns
    -------
    list of Tensor
        A list of tensor views sharing data with the original.

    See Also
    --------
    array_split : Split allowing uneven divisions.
    """
    cpp_data = tensor._data
    if isinstance(indices_or_sections, int):
        parts = type(cpp_data).split_sections(cpp_data, indices_or_sections, axis)
    else:
        parts = type(cpp_data).split_indices(cpp_data, list(indices_or_sections), axis)
    return [tensor._to_py_tensor(p) for p in parts]


def array_split(tensor, indices_or_sections, axis=0):
    """Split a tensor into sub-tensors, allowing uneven splits.

    Like ``split``, but when *indices_or_sections* is an integer and the axis
    size is not evenly divisible, the first sections are one element larger.

    Parameters
    ----------
    tensor : Tensor
        The tensor to split.
    indices_or_sections : int or list of int
        If an int, the tensor is split into that many parts (uneven allowed).
        If a list, it gives the indices where splits occur (same as ``split``).
    axis : int
        The axis along which to split (default 0).

    Returns
    -------
    list of Tensor
        A list of tensor views sharing data with the original.

    See Also
    --------
    split : Split requiring equal divisions.
    """
    cpp_data = tensor._data
    if isinstance(indices_or_sections, int):
        parts = type(cpp_data).array_split(cpp_data, indices_or_sections, axis)
    else:
        parts = type(cpp_data).split_indices(cpp_data, list(indices_or_sections), axis)
    return [tensor._to_py_tensor(p) for p in parts]
