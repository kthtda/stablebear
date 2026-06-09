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

import stablebear._sb_cpp as cpp

from .persistence.barcode import Barcode
from .persistence.ph_tensor import BarcodeTensor
from .distance_matrix import DistanceMatrix, DistanceMatrixTensor
from .symmetric_matrix import SymmetricMatrix, SymmetricMatrixTensor
from .functional.pcf import Pcf
from .tensor import (
    BoolTensor,
    FloatTensor,
    IntPcfTensor,
    IntTensor,
    PcfTensor,
    PointCloudTensor,
    Tensor,
)
from .typing import (
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


def _save(item: Tensor, file):
    _SAVE_DISPATCH = {
        float32: cpp.IoOps.save_float32_tensor,
        float64: cpp.IoOps.save_float64_tensor,
        int32: cpp.IoOps.save_int32_tensor,
        int64: cpp.IoOps.save_int64_tensor,
        uint32: cpp.IoOps.save_uint32_tensor,
        uint64: cpp.IoOps.save_uint64_tensor,
        boolean: cpp.IoOps.save_bool_tensor,
        pcf32: cpp.IoOps.save_pcf32_tensor,
        pcf64: cpp.IoOps.save_pcf64_tensor,
        pcf32i: cpp.IoOps.save_pcf32i_tensor,
        pcf64i: cpp.IoOps.save_pcf64i_tensor,
        pcloud32: cpp.IoOps.save_point_cloud32_tensor,
        pcloud64: cpp.IoOps.save_point_cloud64_tensor,
        barcode32: cpp.IoOps.save_barcode32_tensor,
        barcode64: cpp.IoOps.save_barcode64_tensor,
        symmat32: cpp.IoOps.save_symmetric_matrix32_tensor,
        symmat64: cpp.IoOps.save_symmetric_matrix64_tensor,
        distmat32: cpp.IoOps.save_distance_matrix32_tensor,
        distmat64: cpp.IoOps.save_distance_matrix64_tensor,
    }

    fn = _SAVE_DISPATCH.get(item.dtype)
    if fn is None:
        raise TypeError(f"Unsupported tensor dtype {item.dtype}")
    fn(item._data, file)


def _load(file):
    cpp_p = cpp.persistence

    _LOAD_DISPATCH = {
        cpp.Float32Tensor: FloatTensor,
        cpp.Float64Tensor: FloatTensor,
        cpp.Int32Tensor: IntTensor,
        cpp.Int64Tensor: IntTensor,
        cpp.Uint32Tensor: IntTensor,
        cpp.Uint64Tensor: IntTensor,
        cpp.BoolTensor: BoolTensor,
        cpp.Pcf32Tensor: PcfTensor,
        cpp.Pcf64Tensor: PcfTensor,
        cpp.Pcf32iTensor: IntPcfTensor,
        cpp.Pcf64iTensor: IntPcfTensor,
        cpp.PointCloud32Tensor: PointCloudTensor,
        cpp.PointCloud64Tensor: PointCloudTensor,
        cpp_p.Barcode32Tensor: BarcodeTensor,
        cpp_p.Barcode64Tensor: BarcodeTensor,
        cpp.SymmetricMatrix32Tensor: SymmetricMatrixTensor,
        cpp.SymmetricMatrix64Tensor: SymmetricMatrixTensor,
        cpp.DistanceMatrix32Tensor: DistanceMatrixTensor,
        cpp.DistanceMatrix64Tensor: DistanceMatrixTensor,
    }

    cpp_tensor = cpp.IoOps.load_tensor_from_file(file)
    ctor = _LOAD_DISPATCH.get(type(cpp_tensor))
    if ctor is None:
        raise TypeError(f"File contains unsupported tensor of type {type(cpp_tensor)}")
    return ctor(cpp_tensor)


_OBJECT_SAVE_DISPATCH = None


def _init_object_save_dispatch():
    global _OBJECT_SAVE_DISPATCH
    if _OBJECT_SAVE_DISPATCH is not None:
        return
    cpp_p = cpp.persistence
    _OBJECT_SAVE_DISPATCH = {
        cpp.Pcf_f32_f32: cpp.IoOps.save_pcf32_object,
        cpp.Pcf_f64_f64: cpp.IoOps.save_pcf64_object,
        cpp.Pcf_i32_i32: cpp.IoOps.save_pcf32i_object,
        cpp.Pcf_i64_i64: cpp.IoOps.save_pcf64i_object,
        cpp_p.Barcode32: cpp.IoOps.save_barcode32_object,
        cpp_p.Barcode64: cpp.IoOps.save_barcode64_object,
        cpp.SymmetricMatrix_f32: cpp.IoOps.save_symmetric_matrix32_object,
        cpp.SymmetricMatrix_f64: cpp.IoOps.save_symmetric_matrix64_object,
        cpp.DistanceMatrix_f32: cpp.IoOps.save_distance_matrix32_object,
        cpp.DistanceMatrix_f64: cpp.IoOps.save_distance_matrix64_object,
    }


_OBJECT_LOAD_DISPATCH = None


def _init_object_load_dispatch():
    global _OBJECT_LOAD_DISPATCH
    if _OBJECT_LOAD_DISPATCH is not None:
        return
    cpp_p = cpp.persistence
    _OBJECT_LOAD_DISPATCH = {
        cpp.Pcf_f32_f32: Pcf,
        cpp.Pcf_f64_f64: Pcf,
        cpp.Pcf_i32_i32: Pcf,
        cpp.Pcf_i64_i64: Pcf,
        cpp_p.Barcode32: Barcode,
        cpp_p.Barcode64: Barcode,
        cpp.SymmetricMatrix_f32: SymmetricMatrix,
        cpp.SymmetricMatrix_f64: SymmetricMatrix,
        cpp.DistanceMatrix_f32: DistanceMatrix,
        cpp.DistanceMatrix_f64: DistanceMatrix,
    }


def _save_object(item, file):
    _init_object_save_dispatch()
    fn = _OBJECT_SAVE_DISPATCH.get(type(item._data))
    if fn is None:
        raise TypeError(f"Unsupported object type {type(item._data)}")
    fn(item._data, file)


def _load_object(file):
    _init_object_load_dispatch()
    cpp_obj = cpp.IoOps.load_object_from_file(file)
    ctor = _OBJECT_LOAD_DISPATCH.get(type(cpp_obj))
    if ctor is None:
        raise TypeError(f"File contains unsupported object of type {type(cpp_obj)}")
    return ctor(cpp_obj)


def save(item, file):
    """Save a tensor or object to a file in stablebear's binary format.

    All tensor types and standalone objects (Pcf, Barcode, DistanceMatrix,
    SymmetricMatrix) are supported.

    Parameters
    ----------
    item : Tensor or Pcf or Barcode or DistanceMatrix or SymmetricMatrix
        The item to save.
    file : str or file-like
        A file path or an open file object in binary write mode.
    """
    is_object = isinstance(item, (Pcf, Barcode, DistanceMatrix, SymmetricMatrix))
    save_fn = _save_object if is_object else _save
    if isinstance(file, str):
        with open(file, "wb") as f:
            save_fn(item, f)
    else:
        save_fn(item, file)


def load(file):
    """Load a tensor or object from a file in stablebear's binary format.

    The returned item will have the same type and dtype as what was saved.

    Parameters
    ----------
    file : str or file-like
        A file path or an open file object in binary read mode.

    Returns
    -------
    Tensor or Pcf or Barcode or DistanceMatrix or SymmetricMatrix
        The loaded item.
    """
    if isinstance(file, str):
        with open(file, "rb") as f:
            return _load_any(f)
    else:
        return _load_any(file)


def _unpickle_object(data: bytes):
    import io as _io
    return _load_object(_io.BytesIO(data))


def _load_any(file):
    """Try loading as tensor first, then as object."""
    import io as _io

    # We need to peek at the format type to decide tensor vs object.
    # Read into a buffer so we can try both paths.
    data = file.read()
    buf = _io.BytesIO(data)
    try:
        return _load(buf)
    except RuntimeError:
        buf.seek(0)
        return _load_object(buf)
