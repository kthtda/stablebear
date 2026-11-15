import functools
from typing import Any

import stablebear._sb_cpp as cpp

from .persistence.barcode import Barcode
from .persistence.ph_tensor import BarcodeTensor
from .distance_matrix import DistanceMatrix, DistanceMatrixTensor
from .symmetric_matrix import SymmetricMatrix, SymmetricMatrixTensor
from .functional.pcf import Pcf
from .functional.plf import Plf
from .base_tensor import (
    BoolTensor,
    FloatTensor,
    IntPcfTensor,
    IntTensor,
    PcfTensor,
    Tensor,
)
from .nested_tensor import NestedTensor
from .point_cloud import PointCloud, PointCloudTensor
from .typing import float32, float64
from .timeseries.series import _SERIES_NATIVE, _SERIES_TENSOR_TO_DTYPE, TimeSeries, TimeSeriesTensor


_BINARY_MAGIC = b"\x01MPCF"


@functools.cache
def _tensor_io():
    """Each native tensor type, with its save function and its wrapper."""
    cpp_p = cpp.persistence
    ops = cpp.IoOps
    table = {
        cpp.Float32Tensor: (ops.save_float32_tensor, FloatTensor),
        cpp.Float64Tensor: (ops.save_float64_tensor, FloatTensor),
        cpp.Int32Tensor: (ops.save_int32_tensor, IntTensor),
        cpp.Int64Tensor: (ops.save_int64_tensor, IntTensor),
        cpp.Uint32Tensor: (ops.save_uint32_tensor, IntTensor),
        cpp.Uint64Tensor: (ops.save_uint64_tensor, IntTensor),
        cpp.BoolTensor: (ops.save_bool_tensor, BoolTensor),
        cpp.Pcf32Tensor: (ops.save_pcf32_tensor, PcfTensor),
        cpp.Pcf64Tensor: (ops.save_pcf64_tensor, PcfTensor),
        cpp.Pcf32iTensor: (ops.save_pcf32i_tensor, IntPcfTensor),
        cpp.Pcf64iTensor: (ops.save_pcf64i_tensor, IntPcfTensor),
        cpp.PointCloud32Tensor: (ops.save_point_cloud32_tensor, PointCloudTensor),
        cpp.PointCloud64Tensor: (ops.save_point_cloud64_tensor, PointCloudTensor),
        cpp._IndexedPointCloud32Tensor: (ops.save_indexed_point_cloud32_tensor, PointCloudTensor),
        cpp._IndexedPointCloud64Tensor: (ops.save_indexed_point_cloud64_tensor, PointCloudTensor),
        cpp_p.Barcode32Tensor: (ops.save_barcode32_tensor, BarcodeTensor),
        cpp_p.Barcode64Tensor: (ops.save_barcode64_tensor, BarcodeTensor),
        cpp.SymmetricMatrix32Tensor: (ops.save_symmetric_matrix32_tensor, SymmetricMatrixTensor),
        cpp.SymmetricMatrix64Tensor: (ops.save_symmetric_matrix64_tensor, SymmetricMatrixTensor),
        cpp.DistanceMatrix32Tensor: (ops.save_distance_matrix32_tensor, DistanceMatrixTensor),
        cpp.DistanceMatrix64Tensor: (ops.save_distance_matrix64_tensor, DistanceMatrixTensor),
        cpp._IndexedDistanceMatrix32Tensor: (ops.save_indexed_distance_matrix32_tensor, DistanceMatrixTensor),
        cpp._IndexedDistanceMatrix64Tensor: (ops.save_indexed_distance_matrix64_tensor, DistanceMatrixTensor),
    }
    for native in _SERIES_TENSOR_TO_DTYPE:
        table[native] = (native._save, TimeSeriesTensor)
    return table


@functools.cache
def _object_io():
    """Each native object type, with its save function and its wrapper."""
    cpp_p = cpp.persistence
    ops = cpp.IoOps
    table = {
        cpp.Pcf_f32_f32: (ops.save_pcf32_object, Pcf),
        cpp.Pcf_f64_f64: (ops.save_pcf64_object, Pcf),
        cpp.Pcf_i32_i32: (ops.save_pcf32i_object, Pcf),
        cpp.Pcf_i64_i64: (ops.save_pcf64i_object, Pcf),
        cpp.Plf_f32_f32: (ops.save_plf32_object, lambda data: Plf._from_cpp(data, float32)),
        cpp.Plf_f64_f64: (ops.save_plf64_object, lambda data: Plf._from_cpp(data, float64)),
        cpp.PointCloud32: (ops.save_point_cloud32_object, PointCloud),
        cpp.PointCloud64: (ops.save_point_cloud64_object, PointCloud),
        cpp_p.Barcode32: (ops.save_barcode32_object, Barcode),
        cpp_p.Barcode64: (ops.save_barcode64_object, Barcode),
        cpp.SymmetricMatrix_f32: (ops.save_symmetric_matrix32_object, SymmetricMatrix),
        cpp.SymmetricMatrix_f64: (ops.save_symmetric_matrix64_object, SymmetricMatrix),
        cpp.DistanceMatrix_f32: (ops.save_distance_matrix32_object, DistanceMatrix),
        cpp.DistanceMatrix_f64: (ops.save_distance_matrix64_object, DistanceMatrix),
    }
    for native in _SERIES_NATIVE.values():
        table[native] = (native._save, TimeSeries._from_cpp)
    return table


def _save(item: Tensor, file):
    if isinstance(item, NestedTensor):
        cpp.IoOps.save_nested_tensor(item._root, file)
        return
    entry = _tensor_io().get(type(item._data))
    if entry is None:
        raise TypeError(f"Unsupported tensor type {type(item._data)}")
    entry[0](item._data, file)


def _load(file):
    cpp_tensor = cpp.IoOps.load_tensor_from_file(file)
    if NestedTensor._is_cpp_nested_tensor(cpp_tensor):
        return NestedTensor._from_cpp(cpp_tensor)
    entry = _tensor_io().get(type(cpp_tensor))
    if entry is None:
        raise TypeError(f"File contains unsupported tensor of type {type(cpp_tensor)}")
    return entry[1](cpp_tensor)


def _save_object(item, file):
    data = item._binary_io_data()
    entry = _object_io().get(type(data))
    if entry is None:
        raise TypeError(f"Unsupported object type {type(data)}")
    entry[0](data, file)


def _load_object(file):
    cpp_obj = cpp.IoOps.load_object_from_file(file)
    entry = _object_io().get(type(cpp_obj))
    if entry is None:
        raise TypeError(f"File contains unsupported object of type {type(cpp_obj)}")
    return entry[1](cpp_obj)


def save(item, file):
    """Save a tensor or object to a file in stablebear's binary format.

    All tensor types, standalone objects (Pcf, Plf, PointCloud, Barcode,
    DistanceMatrix, SymmetricMatrix), and time series are supported.

    Parameters
    ----------
    item : Tensor or Pcf or Plf or PointCloud or Barcode or DistanceMatrix or SymmetricMatrix or TimeSeries
        The item to save.
    file : str or typing.BinaryIO
        A file path or an open file object in binary write mode.
    """
    is_object = isinstance(
        item, (Pcf, Plf, PointCloud, Barcode, DistanceMatrix, SymmetricMatrix, TimeSeries)
    )
    save_fn = _save_object if is_object else _save
    if isinstance(file, str):
        with open(file, "wb") as f:
            save_fn(item, f)
    else:
        save_fn(item, file)


def load(file) -> Any:  # Any: the type is whatever the file holds.
    """Load a tensor or object from a file in stablebear's binary format.

    The returned item will have the same type and dtype as what was saved.

    Parameters
    ----------
    file : str or typing.BinaryIO
        A file path or an open file object in binary read mode.

    Returns
    -------
    Tensor or Pcf or Plf or PointCloud or Barcode or DistanceMatrix or SymmetricMatrix or TimeSeries
        The loaded item.
    """
    if isinstance(file, str):
        with open(file, "rb") as f:
            return _load_any(f)
    else:
        return _load_any(file)


def _pickle_reduce(item):
    """Return the shared binary-IO reduction used by public data objects."""
    import io as _io

    buf = _io.BytesIO()
    save(item, buf)
    return _unpickle, (buf.getvalue(),)


def _unpickle(data: bytes, legacy_loader=None):
    """Restore a binary pickle, or delegate a recognized legacy payload.

    The magic check is intentionally performed before loading. A corrupt
    Stablebear binary payload must report its binary-format error rather than
    being reinterpreted by a legacy decoder.
    """
    import io as _io
    import pickle as _pickle

    if data.startswith(_BINARY_MAGIC):
        return load(_io.BytesIO(data))
    if legacy_loader is not None:
        return legacy_loader(data)
    raise _pickle.UnpicklingError("Unrecognized Stablebear pickle payload")


# Retain the callable paths embedded in pickles written since binary pickling
# was introduced. New reducers use _unpickle directly.
def _unpickle_object(data: bytes):
    return _unpickle(data)


def _unpickle_tensor(data: bytes):
    return _unpickle(data)


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
