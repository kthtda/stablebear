from __future__ import annotations

import numpy as np

from . import _sb_cpp as cpp
from ._tensor_base import ArithmeticTensorMixin, FunctionTensorMixin, Shape, Tensor, _tensor_from_nested
from .functional.pcf import Pcf
from .typing import (
    _NP_TO_SB,
    _validate_dtype,
    boolean,
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
    uint32,
    uint64,
)

_FLOAT_CPP_TO_DTYPE = {
    cpp.Float32Tensor: float32,
    cpp.Float64Tensor: float64,
}

_PCF_CPP_TO_DTYPE = {
    cpp.Pcf32Tensor: pcf32,
    cpp.Pcf64Tensor: pcf64,
}

_INTPCF_CPP_TO_DTYPE = {
    cpp.Pcf32iTensor: pcf32i,
    cpp.Pcf64iTensor: pcf64i,
}

_PCLOUD_CPP_TO_DTYPE = {
    cpp.PointCloud32Tensor: pcloud32,
    cpp.PointCloud64Tensor: pcloud64,
}

_PCLOUD_TO_FLOAT_DTYPE = {pcloud32: float32, pcloud64: float64}


class NumericTensor(Tensor, ArithmeticTensorMixin):
    def __init__(self):
        super().__init__()

    def __float__(self):
        if self.size != 1:
            raise TypeError(
                "Only single-element tensors can be converted to a Python scalar. "
                f"This tensor has {self.size} elements."
            )
        idx = [0] * self.ndim
        return float(self._data._get_element(idx))

    def __int__(self):
        return int(self.__float__())

    def _get_valid_setitem_dtypes(self):
        return [NumericTensor, float, int, np.ndarray]

    def _decay_value(self, val):
        return val

    def _represent_element(self, element):
        return element

    def __array__(self, dtype=None, copy=None):
        data = self._data if self._data.is_contiguous() else self.copy()._data

        arr = np.array(data)

        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __eq__(self, other):
        if isinstance(other, np.ndarray):
            other = type(self)(other)
        return super().__eq__(other)

    def array_equal(self, other) -> bool:
        if isinstance(other, np.ndarray):
            return np.array_equal(np.asarray(self), other)
        return super().array_equal(other)

    def __floordiv__(self, rhs):
        rhs_arr = np.asarray(rhs) if isinstance(rhs, NumericTensor) else rhs
        return self._to_py_tensor(np.asarray(self) // rhs_arr)

    def __rfloordiv__(self, lhs):
        return self._to_py_tensor(lhs // np.asarray(self))

    def __ifloordiv__(self, rhs):
        rhs_arr = np.asarray(rhs) if isinstance(rhs, NumericTensor) else rhs
        new = self._to_py_tensor(np.asarray(self) // rhs_arr)
        self._data = new._data
        return self


class FloatTensor(NumericTensor):
    def __init__(self, data: cpp.Float32Tensor | cpp.Float64Tensor | FloatTensor | np.ndarray | list | tuple, dtype=None):
        super().__init__()
        if isinstance(data, (list, tuple)):
            data = np.asarray(data)

        if isinstance(data, FloatTensor):
            data = data._data
        elif isinstance(data, np.ndarray):
            if dtype is None:
                if data.dtype == np.float32:
                    dtype = float32
                else:
                    dtype = float64
            if dtype == float32:
                data = cpp.ndarray_to_tensor_32(np.asarray(data, dtype=np.float32))
            else:
                data = cpp.ndarray_to_tensor_64(np.asarray(data, dtype=np.float64))
        elif not isinstance(data, (cpp.Float32Tensor, cpp.Float64Tensor)):
            raise TypeError(f"Cannot create FloatTensor from {type(data)}")

        self._data = data
        self.dtype = _FLOAT_CPP_TO_DTYPE[type(self._data)]

    def _to_py_tensor(self, data):
        return FloatTensor(data)

    def __repr__(self):
        return np.asarray(self).__repr__()

    def __str__(self):
        return np.asarray(self).__str__()


_INT_CPP_TO_DTYPE = {
    cpp.Int32Tensor: int32,
    cpp.Int64Tensor: int64,
    cpp.Uint32Tensor: uint32,
    cpp.Uint64Tensor: uint64,
}


class IntTensor(NumericTensor):
    def __init__(self, data: cpp.Int32Tensor | cpp.Int64Tensor | cpp.Uint32Tensor | cpp.Uint64Tensor | IntTensor | np.ndarray | list | tuple, dtype=None):
        super().__init__()
        if isinstance(data, (list, tuple)):
            data = np.asarray(data)

        if isinstance(data, IntTensor):
            data = data._data
        elif isinstance(data, np.ndarray):
            if dtype is None:
                dtype = _NP_TO_SB.get(data.dtype.type, int64)
            convert = {
                int32: lambda d: cpp.ndarray_to_tensor_i32(np.asarray(d, dtype=np.int32)),
                int64: lambda d: cpp.ndarray_to_tensor_i64(np.asarray(d, dtype=np.int64)),
                uint32: lambda d: cpp.ndarray_to_tensor_u32(np.asarray(d, dtype=np.uint32)),
                uint64: lambda d: cpp.ndarray_to_tensor_u64(np.asarray(d, dtype=np.uint64)),
            }
            data = convert[dtype](data)
        elif not isinstance(data, tuple(_INT_CPP_TO_DTYPE.keys())):
            raise TypeError(f"Cannot create IntTensor from {type(data)}")

        self._data = data
        self.dtype = _INT_CPP_TO_DTYPE[type(self._data)]

    def _to_py_tensor(self, data):
        return IntTensor(data)

    def _as_float64(self):
        """Convert to float64 FloatTensor, matching NumPy int division promotion."""
        return FloatTensor(np.asarray(self).astype(np.float64))

    def __truediv__(self, rhs):
        if isinstance(rhs, IntTensor):
            return self._as_float64() / rhs._as_float64()
        return self._as_float64() / rhs

    def __rtruediv__(self, lhs):
        if isinstance(lhs, IntTensor):
            return lhs._as_float64() / self._as_float64()
        return lhs / self._as_float64()

    def __itruediv__(self, rhs):
        raise TypeError(
            "In-place true division is not supported for IntTensor "
            "(result is float). Use `x = x / y` instead."
        )

    def __neg__(self):
        if self.dtype in (uint32, uint64):
            raise TypeError(f"Negation is not supported for unsigned dtype {self.dtype.__name__}")
        return super().__neg__()

    def __repr__(self):
        return np.asarray(self).__repr__()

    def __str__(self):
        return np.asarray(self).__str__()


class _PcfTensorBase(Tensor, ArithmeticTensorMixin, FunctionTensorMixin):
    def __init__(self):
        super().__init__()

    def _get_valid_setitem_dtypes(self):
        return [_PcfTensorBase, Pcf]

    def _decay_value(self, val):
        return val._data

    def _represent_element(self, element):
        return Pcf(element)


class PcfTensor(_PcfTensorBase):
    def __init__(self, data):
        super().__init__()
        if isinstance(data, PcfTensor):
            data = data._data
        elif isinstance(data, (list, tuple)):
            data = _tensor_from_nested(data, {
                cpp.Pcf_f32_f32: cpp.Pcf32Tensor,
                cpp.Pcf_f64_f64: cpp.Pcf64Tensor,
            })
        elif not isinstance(data, (cpp.Pcf32Tensor, cpp.Pcf64Tensor)):
            raise TypeError(f"Cannot create PcfTensor from {type(data)}")
        self._data = data
        self.dtype = _PCF_CPP_TO_DTYPE[type(self._data)]

    def _to_py_tensor(self, data):
        return PcfTensor(data)


class IntPcfTensor(_PcfTensorBase):
    def __init__(self, data):
        super().__init__()
        if isinstance(data, IntPcfTensor):
            data = data._data
        elif isinstance(data, (list, tuple)):
            data = _tensor_from_nested(data, {
                cpp.Pcf_i32_i32: cpp.Pcf32iTensor,
                cpp.Pcf_i64_i64: cpp.Pcf64iTensor,
            })
        elif not isinstance(data, (cpp.Pcf32iTensor, cpp.Pcf64iTensor)):
            raise TypeError(f"Cannot create IntPcfTensor from {type(data)}")
        self._data = data
        self.dtype = _INTPCF_CPP_TO_DTYPE[type(self._data)]

    def _to_py_tensor(self, data):
        return IntPcfTensor(data)


class PointCloud:
    """A single (rank-2) point cloud, of shape ``(n_points, dim)``.

    May be an *indexed view* that shares another cloud's coordinates and selects
    rows through an index array (the memory-frugal output of
    :func:`stablebear.sampling.subsample`). All operations work directly on the view;
    the coordinates are only copied when the cloud is converted to NumPy.
    """

    def __init__(self, data):
        self._data = data
        self.dtype = float64 if isinstance(data, cpp.Float64Tensor) else float32

    @property
    def shape(self):
        return Shape([self._data.n_points, self._data.n_dims])

    @property
    def is_indexed(self):
        """Whether this cloud is an indexed view rather than owning its coordinates."""
        return self._data.is_indexed

    @property
    def indices(self):
        """The selected source-row indices, or ``None`` when not an indexed view."""
        return IntTensor(self._data.indices) if self._data.is_indexed else None

    def materialize(self) -> FloatTensor:
        """Return a contiguous ``FloatTensor`` of the (selected) coordinates."""
        return FloatTensor(self._data.materialize())

    def __getitem__(self, index):
        """Index into the cloud's ``(n_points, dim)`` coordinates as a ``FloatTensor``.

        The (selected) coordinates are materialized, so the natural NumPy idiom
        ``pc[:, 0]`` / ``pc[:, 1]`` (e.g. for plotting) works directly on a cloud.
        """
        return self.materialize()[index]

    def to_numpy(self):
        return np.asarray(self._data.materialize())

    def __array__(self, dtype=None):
        arr = self.to_numpy()
        return arr if dtype is None else arr.astype(dtype)

    def array_equal(self, other) -> bool:
        return np.array_equal(self.to_numpy(), np.asarray(other))

    def __repr__(self):
        return self.to_numpy().__repr__()

    def __str__(self):
        return self.to_numpy().__str__()


def _pcloud_dtype_for(arr_dtype, dtype):
    if dtype is not None:
        if dtype not in (pcloud32, pcloud64):
            raise TypeError(
                f"dtype must be pcloud32 or pcloud64, got {getattr(dtype, '__name__', dtype)}")
        return dtype
    return pcloud32 if np.dtype(arr_dtype) == np.float32 else pcloud64


def _pointcloud_cpp_from_array(arr, cloud_ndim, dtype):
    """Build a C++ point-cloud tensor from a dense ndarray.

    The trailing ``cloud_ndim`` axes of *arr* form each cloud; the leading
    axes form the tensor shape. E.g. a ``(3, 5, 4, 2)`` array with
    ``cloud_ndim=2`` yields a ``(3, 5)`` tensor of ``(4, 2)`` clouds.
    """
    from .tensor_create import zeros
    arr = np.asarray(arr)
    if cloud_ndim < 1:
        raise ValueError("cloud_ndim must be >= 1")
    if arr.ndim < cloud_ndim:
        raise ValueError(
            f"array with {arr.ndim} dimension(s) is too small for cloud_ndim={cloud_ndim}")
    dt = _pcloud_dtype_for(arr.dtype, dtype)
    np_float = np.float32 if dt == pcloud32 else np.float64
    arr = np.ascontiguousarray(arr, dtype=np_float)
    tensor_shape = arr.shape[: arr.ndim - cloud_ndim]
    t = zeros(tensor_shape, dtype=dt)
    for idx in np.ndindex(*tensor_shape):
        t[idx] = arr[idx]
    return t._data


def _pointcloud_cpp_from_list(seq, dtype):
    """Build a 1-D C++ point-cloud tensor from a list of cloud ndarrays.

    Clouds may have differing numbers of points (ragged), which a single
    dense ndarray cannot represent.
    """
    from .tensor_create import zeros
    clouds = [np.asarray(c) for c in seq]
    if not clouds:
        raise ValueError(
            "Cannot build a PointCloudTensor from an empty list; "
            "use zeros((0,), dtype=pcloud64) for an empty tensor")
    dt = _pcloud_dtype_for(clouds[0].dtype, dtype)
    np_float = np.float32 if dt == pcloud32 else np.float64
    t = zeros((len(clouds),), dtype=dt)
    for i, c in enumerate(clouds):
        # When inferring (dtype is None), each cloud is a separate array; rather
        # than silently downcast clouds whose precision differs from the first,
        # require them to agree (or pass an explicit dtype=).
        if dtype is None and _pcloud_dtype_for(c.dtype, None) != dt:
            raise TypeError(
                f"point clouds have differing dtypes "
                f"({np.dtype(clouds[0].dtype).name} and {np.dtype(c.dtype).name}); "
                "pass an explicit dtype= (pcloud32 or pcloud64)")
        t[i] = np.ascontiguousarray(c, dtype=np_float)
    return t._data


class PointCloudTensor(Tensor):
    """Tensor whose elements are point clouds (each a ``(n_points, dim)`` array).

    Parameters
    ----------
    data : ndarray, list of ndarray, PointCloudTensor, or C++ tensor
        An ndarray whose trailing ``cloud_ndim`` axes form each cloud and
        whose leading axes form the tensor shape; or a list of cloud arrays
        (possibly ragged) forming a 1-D tensor; or an existing tensor.
    cloud_ndim : int, optional
        Number of trailing axes of an ndarray *data* that make up each cloud,
        by default 2 (``(n_points, dim)``).
    dtype : pcloud32 | pcloud64 | None, optional
        Element precision. Inferred from the array dtype when ``None``.
    """

    def __init__(self, data, cloud_ndim=2, dtype=None):
        super().__init__()
        if isinstance(data, PointCloudTensor):
            data = data._data
        elif isinstance(data, np.ndarray):
            data = _pointcloud_cpp_from_array(data, cloud_ndim, dtype)
        elif isinstance(data, (list, tuple)):
            data = _pointcloud_cpp_from_list(data, dtype)
        elif not isinstance(data, (cpp.PointCloud32Tensor, cpp.PointCloud64Tensor)):
            raise TypeError(f"Cannot create PointCloudTensor from {type(data)}")
        self._data = data
        self.dtype = _PCLOUD_CPP_TO_DTYPE[type(self._data)]

    def _to_py_tensor(self, data):
        return PointCloudTensor(data)

    def _represent_element(self, element):
        return PointCloud(element)

    def _single_cloud(self):
        """Return the lone cloud of a 0-d tensor as a ``(n_points, dim)`` ``PointCloud``."""
        return self._represent_element(self._data._get_element([]))

    def __getitem__(self, index):
        """Index the tensor of clouds, or — when this is a single cloud — the cloud itself.

        A 0-d ``PointCloudTensor`` wraps exactly one point cloud. Indexing it
        delegates to that cloud's ``(n_points, dim)`` array, so the natural
        NumPy idiom for plotting a cloud works directly::

            plt.scatter(pc[:, 0], pc[:, 1])

        For tensors of rank >= 1, indexing selects clouds as usual (a full
        integer index returns one cloud as a ``FloatTensor``).
        """
        if self.ndim == 0:
            return self._single_cloud()[index]
        return super().__getitem__(index)

    def _decay_value(self, val):
        float_dtype = _PCLOUD_TO_FLOAT_DTYPE[self.dtype]
        t = FloatTensor(val, dtype=float_dtype)
        if t.ndim != 2:
            raise ValueError(
                f"A point cloud must be 2-D (n_points, dim); got shape {tuple(t.shape)}."
            )
        return t._data

    def _get_valid_setitem_dtypes(self):
        return [FloatTensor, np.ndarray, float, int]


class BoolTensor(Tensor):
    """Tensor of boolean values, typically produced by elementwise comparisons."""

    def __init__(self, data: cpp.BoolTensor | BoolTensor | np.ndarray | list | tuple):
        super().__init__()
        if isinstance(data, (list, tuple)):
            data = np.asarray(data)

        if isinstance(data, BoolTensor):
            data = data._data
        elif isinstance(data, np.ndarray):
            data = cpp.ndarray_to_bool_tensor(np.asarray(data, dtype=np.bool_))
        elif not isinstance(data, cpp.BoolTensor):
            raise TypeError(f"Cannot create BoolTensor from {type(data)}")
        self._data = data
        self.dtype = boolean

    def _to_py_tensor(self, data):
        return BoolTensor(data)

    def _decay_value(self, val):
        return val

    def _represent_element(self, element):
        return element

    def _get_valid_setitem_dtypes(self):
        return [BoolTensor, bool]

    def __bool__(self):
        total = 1
        for d in self.shape:
            total *= d
        if total == 1:
            idx = [0] * len(self.shape)
            return bool(self._data._get_element(idx))
        raise ValueError(
            "The truth value of a tensor with more than one element is ambiguous. "
            "Use array_equal() for whole-tensor comparison."
        )

    def __array__(self, dtype=None, copy=None):
        data = self._data if self._data.is_contiguous() else self.copy()._data
        arr = np.array(data)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr


PcfContainerLike = Tensor | list[Pcf] | Pcf


def _to_tensor_pcf(fs: PcfContainerLike):
    if isinstance(fs, _PcfTensorBase):
        return fs

    if isinstance(fs, Pcf):
        return PcfTensor([fs])

    if isinstance(fs, (list, tuple)):
        if not fs:
            return PcfTensor(fs)
        first = fs[0] if not isinstance(fs[0], (list, tuple)) else fs[0][0]
        if isinstance(first, Pcf):
            if first.vtype in (int32, int64):
                return IntPcfTensor(fs)
            return PcfTensor(fs)

    raise TypeError("Input should be convertible to a PcfTensor.")


def _get_backend(fs, backendMapping: dict):
    if isinstance(fs, Tensor):
        _validate_dtype(fs.dtype, backendMapping.keys())

        backend = backendMapping.get(fs.dtype)
        if backend is None:
            raise ValueError(
                f"Operation not supported for tensors of this type ({fs._type} with dtype {fs.dtype})"
            )
        return backend, fs
    elif isinstance(fs, np.ndarray):
        return _get_backend(FloatTensor(fs), backendMapping)
    elif isinstance(fs, (Pcf, list, tuple)):
        return _get_backend(_to_tensor_pcf(fs), backendMapping)
    elif hasattr(fs, "dtype"):
        _validate_dtype(fs.dtype, backendMapping.keys())
        backend = backendMapping.get(fs.dtype)
        if backend is None:
            raise ValueError(
                f"Operation not supported for objects of this type ({type(fs)} with dtype {fs.dtype})"
            )
        return backend, fs

    raise ValueError(f"Operation not supported for data of this type ({type(fs)})")


def _resolve_pcf_inputs(backend_map: dict, *inputs: PcfContainerLike):
    """Convert one or more PcfContainerLike inputs to tensors and look up the backend.

    Returns ``(backend, tensor1, tensor2, ...)`` with all tensors guaranteed
    to have the same dtype.

    Raises
    ------
    TypeError
        If inputs have mismatched dtypes.
    """
    results = [_get_backend(inp, backend_map) for inp in inputs]
    backend = results[0][0]
    tensors = [r[1] for r in results]

    dtypes = [t.dtype for t in tensors]
    if len(set(id(d) for d in dtypes)) > 1:
        names = ", ".join(d.__name__ for d in dtypes)
        raise TypeError(f"All inputs must have the same dtype (got {names}).")

    return (backend, *tensors)
