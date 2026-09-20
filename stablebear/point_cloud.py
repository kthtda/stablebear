from __future__ import annotations

import numpy as np

from . import _sb_cpp as cpp
from ._tensor_base import Tensor
from .nested_tensor import NestedTensor
from .typing import float32, float64, pcloud32, pcloud64, uint64


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


class PointCloud:
    """A rank-2 point-cloud view.

    Point clouds returned from a :class:`PointCloudTensor` retain their owner.
    Reads resolve only this cloud. A write asks the owner to materialize if
    necessary, then delegates the actual indexing operation to ``FloatTensor``.
    """

    def __init__(self, data, dtype=None, *, _owner=None, _outer_index=None):
        from .base_tensor import FloatTensor

        self._owner = _owner
        self._outer_index = _outer_index

        if _owner is None:
            if isinstance(data, _POINT_CLOUD_CPP_TYPES):
                self._value = data
                tensor = data.coords
            else:
                if dtype in _PCLOUD_TO_FLOAT_DTYPE:
                    dtype = _PCLOUD_TO_FLOAT_DTYPE[dtype]
                tensor = FloatTensor(data, dtype=dtype)
                cpp_type = (
                    cpp.PointCloud32 if tensor.dtype == float32 else cpp.PointCloud64
                )
                self._value = cpp_type(tensor._data)
            if len(tensor.shape) != 2:
                raise ValueError(
                    "PointCloud must have 2 dimensions, "
                    f"got {len(tensor.shape)}"
                )
        else:
            self._value = None
            if tuple(data.coords.shape) != () and len(data.coords.shape) != 2:
                raise ValueError(
                    "PointCloud must have 2 dimensions, "
                    f"got {len(data.coords.shape)}"
                )

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

    def _readable_coords(self):
        from .base_tensor import FloatTensor

        return FloatTensor(self._current_point_cloud().materialize())

    def _writable_coords(self):
        from .base_tensor import FloatTensor

        if self._owner is not None:
            self._owner._ensure_writeable()
            point_cloud = self._owner._data._get_writeable_element(
                self._outer_index
            )
        else:
            point_cloud = self._value
        return FloatTensor(point_cloud._mutable_coords())

    def __getitem__(self, index):
        return self._readable_coords()[index]

    def __setitem__(self, index, value):
        self._writable_coords()[index] = value

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self._readable_coords(), dtype=dtype)
        if copy:
            return array.copy()
        return array

    def array_equal(self, other):
        return np.array_equal(np.asarray(self), np.asarray(other))

    def materialize(self):
        """Return this point cloud as a standalone ``FloatTensor``."""
        return self._readable_coords().copy()

    def copy(self):
        return PointCloud(self.materialize())

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


class PointCloudTensor(Tensor):
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

    def _point_cloud(self, index):
        element = self._data._get_element(index)
        return PointCloud(element, _owner=self, _outer_index=index)

    def _single_cloud(self):
        return self._point_cloud([])

    def __getitem__(self, index):
        """Select clouds, or points when this tensor contains one cloud.

        A ``NestedTensor`` selects rows independently from each source cloud
        and returns lazy indexed point-cloud views.
        """
        if isinstance(index, NestedTensor):
            if index.dtype is not uint64:
                raise TypeError("Point-cloud indices must have uint64 leaves")
            if index.depth != 2:
                raise ValueError("Point-cloud indexing requires Tensor<Tensor<uint64>>")
            return PointCloudTensor(self._data._index_elements(index._root))

        if self.ndim == 0:
            cloud = self._point_cloud([])
            if index == () or index is Ellipsis:
                return cloud
            return cloud[index]

        entries, inserts = self._normalize_index(index)
        if (
            not inserts
            and len(entries) == self.ndim
            and all(isinstance(entry, int) for entry in entries)
        ):
            resolved = [
                self._resolve_axis_int(entry, axis)
                for axis, entry in enumerate(entries)
            ]
            return self._point_cloud(resolved)

        return super().__getitem__(index)

    def _ensure_writeable(self):
        if isinstance(self._data, _INDEXED_PCLOUD_CPP_TYPES):
            self._data._ensure_materialized()
        super()._ensure_writeable()

    def to_dense(self):
        """Return an independent tensor with ordinary point-cloud storage.

        Indexed tensors resolve their logical row selections without changing
        the source tensor or any views that share its indexed backing. Calling
        this on an already-dense tensor still returns an independent copy.
        """
        return PointCloudTensor(self._data.copy())

    def astype(self, dtype):
        if isinstance(self._data, _INDEXED_PCLOUD_CPP_TYPES):
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
