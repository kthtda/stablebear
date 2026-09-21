import stablebear._sb_cpp as cpp

from .persistence.barcode import Barcode
from .persistence.ph_tensor import BarcodeTensor
from .distance_matrix import DistanceMatrix, DistanceMatrixTensor
from .symmetric_matrix import SymmetricMatrix, SymmetricMatrixTensor
from .functional.pcf import Pcf
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


_BINARY_MAGIC = b"\x01MPCF"


def _save(item: Tensor, file):
    cpp_p = cpp.persistence
    data = item._root if isinstance(item, NestedTensor) else item._data
    if isinstance(
        data,
        (cpp._IndexedPointCloud32Tensor, cpp._IndexedPointCloud64Tensor),
    ):
        data = data.materialize()
    _SAVE_DISPATCH = {
        cpp.Float32Tensor: cpp.IoOps.save_float32_tensor,
        cpp.Float64Tensor: cpp.IoOps.save_float64_tensor,
        cpp.Int32Tensor: cpp.IoOps.save_int32_tensor,
        cpp.Int64Tensor: cpp.IoOps.save_int64_tensor,
        cpp.Uint32Tensor: cpp.IoOps.save_uint32_tensor,
        cpp.Uint64Tensor: cpp.IoOps.save_uint64_tensor,
        cpp.BoolTensor: cpp.IoOps.save_bool_tensor,
        cpp.NestedFloat32: cpp.IoOps.save_nested_float32_tensor,
        cpp.NestedFloat64: cpp.IoOps.save_nested_float64_tensor,
        cpp.NestedInt32: cpp.IoOps.save_nested_int32_tensor,
        cpp.NestedInt64: cpp.IoOps.save_nested_int64_tensor,
        cpp.NestedUint32: cpp.IoOps.save_nested_uint32_tensor,
        cpp.NestedUint64: cpp.IoOps.save_nested_uint64_tensor,
        cpp.Pcf32Tensor: cpp.IoOps.save_pcf32_tensor,
        cpp.Pcf64Tensor: cpp.IoOps.save_pcf64_tensor,
        cpp.Pcf32iTensor: cpp.IoOps.save_pcf32i_tensor,
        cpp.Pcf64iTensor: cpp.IoOps.save_pcf64i_tensor,
        cpp.PointCloud32Tensor: cpp.IoOps.save_point_cloud32_tensor,
        cpp.PointCloud64Tensor: cpp.IoOps.save_point_cloud64_tensor,
        cpp_p.Barcode32Tensor: cpp.IoOps.save_barcode32_tensor,
        cpp_p.Barcode64Tensor: cpp.IoOps.save_barcode64_tensor,
        cpp.SymmetricMatrix32Tensor: cpp.IoOps.save_symmetric_matrix32_tensor,
        cpp.SymmetricMatrix64Tensor: cpp.IoOps.save_symmetric_matrix64_tensor,
        cpp.DistanceMatrix32Tensor: cpp.IoOps.save_distance_matrix32_tensor,
        cpp.DistanceMatrix64Tensor: cpp.IoOps.save_distance_matrix64_tensor,
    }

    fn = _SAVE_DISPATCH.get(type(data))
    if fn is None:
        raise TypeError(f"Unsupported tensor type {type(data)}")
    fn(data, file)


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
    if NestedTensor._is_cpp_nested_tensor(cpp_tensor):
        return NestedTensor._from_cpp(cpp_tensor)

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
        cpp.PointCloud32: cpp.IoOps.save_point_cloud32_object,
        cpp.PointCloud64: cpp.IoOps.save_point_cloud64_object,
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
        cpp.PointCloud32: PointCloud,
        cpp.PointCloud64: PointCloud,
        cpp_p.Barcode32: Barcode,
        cpp_p.Barcode64: Barcode,
        cpp.SymmetricMatrix_f32: SymmetricMatrix,
        cpp.SymmetricMatrix_f64: SymmetricMatrix,
        cpp.DistanceMatrix_f32: DistanceMatrix,
        cpp.DistanceMatrix_f64: DistanceMatrix,
    }


def _save_object(item, file):
    _init_object_save_dispatch()
    data = item._binary_io_data()
    fn = _OBJECT_SAVE_DISPATCH.get(type(data))
    if fn is None:
        raise TypeError(f"Unsupported object type {type(data)}")
    fn(data, file)


def _load_object(file):
    _init_object_load_dispatch()
    cpp_obj = cpp.IoOps.load_object_from_file(file)
    ctor = _OBJECT_LOAD_DISPATCH.get(type(cpp_obj))
    if ctor is None:
        raise TypeError(f"File contains unsupported object of type {type(cpp_obj)}")
    return ctor(cpp_obj)


def save(item, file):
    """Save a tensor or object to a file in stablebear's binary format.

    All tensor types and standalone objects (Pcf, PointCloud, Barcode,
    DistanceMatrix, SymmetricMatrix) are supported.

    Parameters
    ----------
    item : Tensor or Pcf or PointCloud or Barcode or DistanceMatrix or SymmetricMatrix
        The item to save.
    file : str or file-like
        A file path or an open file object in binary write mode.
    """
    is_object = isinstance(
        item, (Pcf, PointCloud, Barcode, DistanceMatrix, SymmetricMatrix)
    )
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
    Tensor or Pcf or PointCloud or Barcode or DistanceMatrix or SymmetricMatrix
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
