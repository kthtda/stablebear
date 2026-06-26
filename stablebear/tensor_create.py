from . import _sb_cpp as cpp
from ._tensor_base import Shape, ShapeLike, _resolve_axis
from .base_tensor import (
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


# Single source of truth for dtype dispatch, shared by zeros() and tensor().
# Maps each dtype sentinel to (wrapper class, C++ tensor class, zero-fill args).
# Built lazily and cached because the matrix / barcode wrappers import back
# into this module.
_DTYPE_DISPATCH = {}

# Families whose constructor selects precision via a ``dtype=`` keyword; the
# rest (bool, PCF, barcode) infer it from the data or have a single precision.
_DATA_DTYPE_KW = frozenset({
    float32, float64, int32, int64, uint32, uint64,
    pcloud32, pcloud64, distmat32, distmat64, symmat32, symmat64,
})


def _dtype_dispatch():
    if not _DTYPE_DISPATCH:
        from .persistence.ph_tensor import BarcodeTensor
        from .distance_matrix import DistanceMatrixTensor
        from .symmetric_matrix import SymmetricMatrixTensor
        # Insertion order is the canonical dtype list used in error messages.
        _DTYPE_DISPATCH.update({
            pcf32: (PcfTensor, cpp.Pcf32Tensor, ()),
            pcf64: (PcfTensor, cpp.Pcf64Tensor, ()),
            pcf32i: (IntPcfTensor, cpp.Pcf32iTensor, ()),
            pcf64i: (IntPcfTensor, cpp.Pcf64iTensor, ()),
            float32: (FloatTensor, cpp.Float32Tensor, (0.0,)),
            float64: (FloatTensor, cpp.Float64Tensor, (0.0,)),
            int32: (IntTensor, cpp.Int32Tensor, (0,)),
            int64: (IntTensor, cpp.Int64Tensor, (0,)),
            uint32: (IntTensor, cpp.Uint32Tensor, (0,)),
            uint64: (IntTensor, cpp.Uint64Tensor, (0,)),
            boolean: (BoolTensor, cpp.BoolTensor, (False,)),
            pcloud32: (PointCloudTensor, cpp.PointCloud32Tensor, ()),
            pcloud64: (PointCloudTensor, cpp.PointCloud64Tensor, ()),
            barcode32: (BarcodeTensor, cpp_p.Barcode32Tensor, ()),
            barcode64: (BarcodeTensor, cpp_p.Barcode64Tensor, ()),
            distmat32: (DistanceMatrixTensor, cpp.DistanceMatrix32Tensor, ()),
            distmat64: (DistanceMatrixTensor, cpp.DistanceMatrix64Tensor, ()),
            symmat32: (SymmetricMatrixTensor, cpp.SymmetricMatrix32Tensor, ()),
            symmat64: (SymmetricMatrixTensor, cpp.SymmetricMatrix64Tensor, ()),
        })
    return _DTYPE_DISPATCH


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

    if not isinstance(shape, Shape):
        shape = Shape(shape)  # If passed as, e.g., tuple of ints

    dispatch = _dtype_dispatch()
    _assert_valid_dtype(dtype, list(dispatch))
    wrapper, cpp_cls, zero_fill = dispatch[dtype]
    return wrapper(cpp_cls(shape, *zero_fill))


def tensor(data, dtype: Dtype = None):
    """Create a tensor from existing data, dispatching on *dtype*.

    A NumPy-like factory that wraps the per-family constructors so any
    supported tensor can be built from an ndarray (or nested list) in one
    call, instead of allocating with :func:`zeros` and assigning elements in
    a Python loop.

    Parameters
    ----------
    data : ndarray, list, or tuple
        The source data. For point-cloud / matrix / barcode tensors this is
        interpreted by the corresponding constructor (see those classes).
    dtype : Dtype, optional
        Target element dtype. When ``None``, a numeric dtype is inferred from
        the array (bool/int/float); non-numeric tensors require an explicit
        dtype.

    Returns
    -------
    Tensor
    """
    if dtype is None:
        import numpy as np
        arr = np.asarray(data)
        if arr.dtype == np.bool_:
            return BoolTensor(arr)
        if np.issubdtype(arr.dtype, np.integer):
            return IntTensor(arr)
        if np.issubdtype(arr.dtype, np.floating):
            return FloatTensor(arr)
        raise TypeError(
            f"Cannot infer a tensor dtype from data of dtype {arr.dtype}; "
            "pass an explicit dtype= argument")

    dispatch = _dtype_dispatch()
    # Barcode tensors are not constructible from raw array/list data.
    if dtype not in dispatch or dtype in (barcode32, barcode64):
        raise TypeError(f"Unsupported dtype {dtype} for tensor()")
    wrapper = dispatch[dtype][0]
    if dtype in _DATA_DTYPE_KW:
        return wrapper(data, dtype=dtype)
    return wrapper(data)


def _require_same_dtype(tensors, op):
    """Raise a clear error if *tensors* do not all share one dtype.

    Joining tensors of mixed dtype/precision otherwise leaked the raw pybind
    overload error; there is no implicit upcasting, so report the mismatch and
    point at ``astype`` (issue #74).
    """
    dtypes = [t.dtype for t in tensors]
    if len(set(dtypes)) > 1:
        names = ", ".join(d.name for d in dtypes)
        raise TypeError(
            f"all tensors passed to {op}() must have the same dtype (got {names}); "
            "cast with .astype() first")


def _normalize_split_indices(indices, axis_size):
    """Resolve negative split indices NumPy-style.

    A negative index counts from the end (``idx + axis_size``) and is clamped to
    0 if still negative; positive overflow is left for the C++ layer to clamp to
    ``axis_size`` (issue #21).
    """
    return [max(int(idx) + axis_size if int(idx) < 0 else int(idx), 0)
            for idx in indices]


def concatenate(tensors, axis=0):
    """Concatenate tensors along an existing axis (outer indexing).

    *axis* may be negative (counting from the last axis), as in NumPy.
    """
    if not tensors:
        raise ValueError("need at least one tensor to concatenate")
    _require_same_dtype(tensors, "concatenate")
    axis = _resolve_axis(axis, tensors[0].ndim)
    cpp_tensors = [t._data for t in tensors]
    result = type(cpp_tensors[0]).concatenate(cpp_tensors, axis)
    return tensors[0]._to_py_tensor(result)


def stack(tensors, axis=0):
    """Stack tensors along a new axis. All tensors must have the same shape."""
    if not tensors:
        raise ValueError("need at least one tensor to stack")
    _require_same_dtype(tensors, "stack")
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
        If a list, it gives the indices where splits occur. Negative indices
        count from the end of the axis, as in NumPy.
    axis : int
        The axis along which to split (default 0). May be negative.

    Returns
    -------
    list of Tensor
        A list of tensor views sharing data with the original.

    See Also
    --------
    array_split : Split allowing uneven divisions.
    """
    cpp_data = tensor._data
    axis = _resolve_axis(axis, tensor.ndim)
    if isinstance(indices_or_sections, int):
        parts = type(cpp_data).split_sections(cpp_data, indices_or_sections, axis)
    else:
        indices = _normalize_split_indices(indices_or_sections, tensor.shape[axis])
        parts = type(cpp_data).split_indices(cpp_data, indices, axis)
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
        Negative indices count from the end of the axis, as in NumPy.
    axis : int
        The axis along which to split (default 0). May be negative.

    Returns
    -------
    list of Tensor
        A list of tensor views sharing data with the original.

    See Also
    --------
    split : Split requiring equal divisions.
    """
    cpp_data = tensor._data
    axis = _resolve_axis(axis, tensor.ndim)
    if isinstance(indices_or_sections, int):
        parts = type(cpp_data).array_split(cpp_data, indices_or_sections, axis)
    else:
        indices = _normalize_split_indices(indices_or_sections, tensor.shape[axis])
        parts = type(cpp_data).split_indices(cpp_data, indices, axis)
    return [tensor._to_py_tensor(p) for p in parts]
