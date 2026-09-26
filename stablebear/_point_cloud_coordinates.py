"""Numeric coordinate views that retain a point cloud's write owner."""

import numpy as np

from .base_tensor import FloatTensor, NumericTensor
from .typing import float32


class CoordinateView(FloatTensor):
    def __init__(self, cloud, rows=None, columns=None, view=lambda array: array,
                 readonly=False):
        self._cloud = cloud
        self.dtype = cloud.dtype
        self._view = view
        self._readonly = readonly
        if rows is None:
            n, d = cloud.shape
            rows = np.broadcast_to(np.arange(n)[:, None], (n, d))
            columns = np.broadcast_to(np.arange(d)[None, :], (n, d))
        self._rows, self._columns = rows, columns

    @property
    def shape(self):
        return self._rows.shape

    @property
    def ndim(self):
        return self._rows.ndim

    @property
    def size(self):
        return self._rows.size

    @property
    def _data(self):
        return FloatTensor(np.asarray(self))._data

    @_data.setter
    def _data(self, data):
        self[...] = FloatTensor(data)

    def _ensure_writeable(self):
        if self._readonly:
            raise ValueError("assignment destination is read-only")

    def _decay_operand(self, value):
        if isinstance(value, (np.ndarray, list, tuple)):
            return FloatTensor(np.asarray(value), dtype=self.dtype)._data
        return super()._decay_operand(value)

    def _read(self, rows, columns):
        cloud = self._cloud._current_point_cloud()
        if cloud.is_indexed:
            rows = np.asarray(cloud.indices)[rows]
        return cloud._storage_array()[rows, columns]

    @staticmethod
    def _index(index):
        if isinstance(index, tuple):
            return tuple(CoordinateView._index(i) for i in index)
        return np.asarray(index) if isinstance(index, NumericTensor) else index

    def __getitem__(self, index):
        index = self._index(index)
        rows, columns = self._rows[index], self._columns[index]
        items = index if isinstance(index, tuple) else (index,)
        basic = all(i is None or i is Ellipsis or isinstance(i, slice)
                    or (isinstance(i, (int, np.integer))
                        and not isinstance(i, (bool, np.bool_))) for i in items)
        if np.isscalar(rows):
            return self._read(rows, columns)
        if basic:
            return CoordinateView(
                self._cloud, rows, columns, lambda array: self._view(array)[index],
                self._readonly)
        return FloatTensor(np.asarray(self._read(rows, columns)))

    def __setitem__(self, index, value):
        self._ensure_writeable()
        index = self._index(index)
        rows, columns = self._rows[index], self._columns[index]
        # Let NumPy check broadcasting and conversion before changing storage.
        values = np.empty(np.shape(rows), dtype=(
            np.float32 if self.dtype is float32 else np.float64))
        values[...] = np.asarray(value) if isinstance(value, NumericTensor) else value
        if values.size == 0:
            return
        cloud = self._cloud._writeable_point_cloud()
        cloud._storage_array()[rows, columns] = values

    def __array__(self, dtype=None, copy=None):
        cloud = self._cloud._current_point_cloud()
        if cloud.is_indexed:
            if copy is False:
                raise ValueError("indexed coordinates require a copy for a NumPy array")
            array = self._read(self._rows, self._columns)
        else:
            array = self._view(cloud._storage_array())
            owner = self._cloud._owner
            if self._readonly or (owner is not None and any(
                    size > 1 and stride == 0
                    for size, stride in zip(owner.shape, owner.strides))):
                array = array.view()
                array.flags.writeable = False
        if dtype is not None and np.dtype(dtype) != array.dtype:
            if copy is False:
                raise ValueError("dtype conversion requires a copy")
            array = array.astype(dtype)
        return array.copy() if copy else array

    def __ipow__(self, exponent):
        self[...] = np.asarray(self) ** exponent
        return self

    def _transform(self, operation, readonly=False):
        return CoordinateView(
            self._cloud, operation(self._rows), operation(self._columns),
            lambda array: operation(self._view(array)),
            self._readonly or readonly)

    def reshape(self, shape):
        array = np.asarray(self)
        reshaped = array.reshape(shape)
        if array.size and not np.shares_memory(array, reshaped):
            return FloatTensor(reshaped)
        return self._transform(lambda array: array.reshape(shape))

    def transpose(self, axes=None):
        return self._transform(lambda array: array.transpose(axes))

    def swapaxes(self, axis1, axis2):
        return self._transform(lambda array: array.swapaxes(axis1, axis2))

    def squeeze(self, axis=None):
        return self._transform(lambda array: array.squeeze(axis))

    def expand_dims(self, axis):
        return self._transform(lambda array: np.expand_dims(array, axis))

    def broadcast_to(self, shape):
        return self._transform(lambda array: np.broadcast_to(array, shape), readonly=True)
