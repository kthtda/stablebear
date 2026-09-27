from __future__ import annotations

import numpy as np

from . import _sb_cpp as cpp
from ._binary_io import _BinaryIoMixin
from ._tensor_base import IndexedElementTensor
from .typing import float32, float64, pcloud32, pcloud64


_PCLOUD_CPP_TO_DTYPE = {
    cpp.PointCloud32Tensor: pcloud32,
    cpp.PointCloud64Tensor: pcloud64,
    cpp._IndexedPointCloud32Tensor: pcloud32,
    cpp._IndexedPointCloud64Tensor: pcloud64,
}

_INDEXED_PCLOUD_CPP_TYPES = (
    cpp._IndexedPointCloud32Tensor,
    cpp._IndexedPointCloud64Tensor,
)

_POINT_CLOUD_CPP_TYPES = (cpp.PointCloud32, cpp.PointCloud64)

_PCLOUD_TO_FLOAT_DTYPE = {pcloud32: float32, pcloud64: float64}


class PointCloud(_BinaryIoMixin):
    """A rank-2 point-cloud view.

    Point clouds returned from a :class:`PointCloudTensor` retain their owner.
    Basic coordinate slices are views; advanced indexing copies coordinates.
    A write asks the owner to materialize if necessary.
    """

    def __init__(self, data, dtype=None, *, _owner=None, _outer_index=None):
        from .base_tensor import FloatTensor

        self._owner = _owner
        self._outer_index = _outer_index

        if _owner is None:
            if isinstance(data, _POINT_CLOUD_CPP_TYPES):
                self._value = data
            else:
                if dtype in _PCLOUD_TO_FLOAT_DTYPE:
                    dtype = _PCLOUD_TO_FLOAT_DTYPE[dtype]
                tensor = FloatTensor(data, dtype=dtype)
                cpp_type = (
                    cpp.PointCloud32 if tensor.dtype == float32 else cpp.PointCloud64
                )
                self._value = cpp_type(tensor._data)
        else:
            self._value = None

    @property
    def shape(self):
        point_cloud = self._current_point_cloud()
        return (point_cloud.n_points, point_cloud.n_dims)

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    @property
    def dtype(self):
        if self._owner is not None:
            return _PCLOUD_TO_FLOAT_DTYPE[self._owner.dtype]
        return float32 if isinstance(self._value, cpp.PointCloud32) else float64

    def _current_point_cloud(self):
        if self._owner is None:
            return self._value
        return self._owner._data._get_element(self._outer_index)

    def _cpp_point_cloud(self):
        return self._current_point_cloud()

    def _binary_io_data(self):
        return self._current_point_cloud()

    def _coordinates(self):
        from ._point_cloud_coordinates import CoordinateView
        return CoordinateView(self)

    def __getitem__(self, index):
        if (isinstance(index, tuple) and len(index) == 2
                and all(isinstance(i, (int, np.integer))
                        and not isinstance(i, (bool, np.bool_)) for i in index)):
            resolved = []
            for i, size in zip(index, self.shape):
                i = int(i)
                i = i + size if i < 0 else i
                if not 0 <= i < size:
                    raise IndexError("point-cloud coordinate index out of bounds")
                resolved.append(i)
            return self._current_point_cloud()._coordinate(*resolved)
        return self._coordinates()[index]

    def __setitem__(self, index, value):
        self._coordinates()[index] = value

    def _writeable_point_cloud(self):
        if self._owner is not None:
            self._owner._ensure_writeable()
            return self._owner._data._get_writeable_element(
                self._outer_index
            )
        if self._value.is_indexed:
            self._value = self._value.copy()
        return self._value

    def __array__(self, dtype=None, copy=None):
        return self._coordinates().__array__(dtype=dtype, copy=copy)

    def array_equal(self, other):
        return np.array_equal(np.asarray(self), np.asarray(other))

    def copy(self):
        return PointCloud(self._current_point_cloud().copy())

    def __repr__(self):
        return f"PointCloud(shape={self.shape}, dtype={self.dtype})"


def _pcloud_dtype_for(arr_dtype, dtype):
    if dtype is not None:
        if dtype not in (pcloud32, pcloud64):
            raise TypeError(
                f"dtype must be pcloud32 or pcloud64, got {getattr(dtype, '__name__', dtype)}")
        return dtype
    return pcloud32 if np.dtype(arr_dtype) == np.float32 else pcloud64


def _pointcloud_cpp_from_array(arr, cloud_ndim, dtype):
    """Build a C++ point-cloud tensor from a dense ndarray."""
    from .tensor_create import zeros

    arr = np.asarray(arr)
    if cloud_ndim != 2:
        raise ValueError(
            "PointCloud elements must have 2 dimensions; cloud_ndim must be 2"
        )
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
    """Build a 1-D C++ point-cloud tensor from a list of cloud arrays."""
    from .tensor_create import zeros

    clouds = [np.asarray(c) for c in seq]
    if not clouds:
        raise ValueError(
            "Cannot build a PointCloudTensor from an empty list; "
            "use zeros((0,), dtype=pcloud64) for an empty tensor")
    dt = _pcloud_dtype_for(clouds[0].dtype, dtype)
    np_float = np.float32 if dt == pcloud32 else np.float64
    t = zeros((len(clouds),), dtype=dt)
    for i, cloud in enumerate(clouds):
        if cloud.ndim != 2:
            raise ValueError(
                f"PointCloud must have 2 dimensions, got {cloud.ndim}"
            )
        if dtype is None and _pcloud_dtype_for(cloud.dtype, None) != dt:
            raise TypeError(
                f"point clouds have differing dtypes "
                f"({np.dtype(clouds[0].dtype).name} and {np.dtype(cloud.dtype).name}); "
                "pass an explicit dtype= (pcloud32 or pcloud64)")
        t[i] = np.ascontiguousarray(cloud, dtype=np_float)
    return t._data


class PointCloudTensor(IndexedElementTensor):
    """Tensor whose elements are point clouds (each a ``(n_points, dim)`` array).

    Parameters
    ----------
    data : ndarray, list of ndarray, PointCloudTensor, or C++ tensor
        An ndarray whose trailing ``cloud_ndim`` axes form each cloud and
        whose leading axes form the tensor shape; or a list of cloud arrays
        (possibly ragged) forming a 1-D tensor; or an existing tensor.
    cloud_ndim : int, optional
        Must be 2. Retained as a keyword for compatibility.
    dtype : pcloud32 | pcloud64 | None, optional
        Element precision. Inferred from the array dtype when ``None``.
    """

    _indexed_cpp_types = _INDEXED_PCLOUD_CPP_TYPES
    _indexed_element_name = "Point-cloud"

    def __init__(self, data, cloud_ndim=2, dtype=None):
        super().__init__()
        if isinstance(data, PointCloudTensor):
            data = data._data
        elif isinstance(data, np.ndarray):
            data = _pointcloud_cpp_from_array(data, cloud_ndim, dtype)
        elif isinstance(data, (list, tuple)):
            data = _pointcloud_cpp_from_list(data, dtype)
        elif not isinstance(data, tuple(_PCLOUD_CPP_TO_DTYPE)):
            raise TypeError(f"Cannot create PointCloudTensor from {type(data)}")
        self._data = data
        self.dtype = _PCLOUD_CPP_TO_DTYPE[type(self._data)]

    def _to_py_tensor(self, data):
        return PointCloudTensor(data)

    def _represent_element(self, element):
        return PointCloud(element)

    def _element_view(self, element, index):
        return PointCloud(element, _owner=self, _outer_index=index)

    def astype(self, dtype):
        if self._has_indexed_storage():
            # Casting is an out-of-place operation. Materialize a temporary
            # value rather than detaching only this wrapper from the indexed
            # backing shared by its parent and sibling views.
            return PointCloudTensor(self._data.materialize()).astype(dtype)
        return super().astype(dtype)

    def _decay_value(self, val):
        from .base_tensor import FloatTensor

        float_dtype = _PCLOUD_TO_FLOAT_DTYPE[self.dtype]
        if isinstance(val, PointCloud):
            val = np.asarray(val)
        tensor = FloatTensor(val, dtype=float_dtype)
        if tensor.ndim != 2:
            raise ValueError(
                f"PointCloud must have 2 dimensions, got {tensor.ndim}"
            )
        return tensor._data

    def _get_valid_setitem_dtypes(self):
        from .base_tensor import FloatTensor

        return [PointCloud, FloatTensor, np.ndarray]
