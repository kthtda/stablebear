from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _sb_cpp as cpp
from ._indexable import Indexable
from ._tensor_base import Shape
from .typing import float32, float64, uint64

if TYPE_CHECKING:
    from .base_tensor import FloatTensor


class PointCloud(Indexable):
    """A single (rank-2) point cloud, of shape ``(n_points, dim)``.

    May be an *indexed view* that shares another cloud's coordinates and selects
    rows through an index array (the memory-frugal output of
    :func:`stablebear.sampling.subsample_relative`). All operations work directly on the view;
    the coordinates are only copied when the cloud is converted to NumPy.

    A cloud that *owns* its coordinates is mutable: ``pc[i, j] = value`` writes a
    coordinate in place. Writing to an *indexed view* is copy-on-write: the view
    first detaches into a private owning copy, so the write lands there and the
    shared source cloud is never touched.
    """

    def __init__(self, data):
        self._data = data
        self.dtype = float64 if isinstance(data, cpp.PointCloud64) else float32

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
        if not self._data.is_indexed:
            return None
        from .base_tensor import IntTensor
        return IntTensor(self._data.indices)

    def materialize(self) -> FloatTensor:
        """Return a contiguous ``FloatTensor`` of the (selected) coordinates."""
        from .base_tensor import FloatTensor
        return FloatTensor(self._data.materialize())

    def copy(self, keep_source: bool = True) -> PointCloud:
        """Return an independent copy.

        An owning cloud deep-copies its coordinates. For an indexed view,
        ``keep_source=True`` (the default) copies only the index array and keeps
        sharing the source coordinates; ``keep_source=False`` also deep-copies the
        source, yielding a view that shares nothing with the original. The copy of
        an indexed view is itself an indexed view either way — use
        :meth:`materialize` for the selected coordinates as a contiguous tensor.

        Parameters
        ----------
        keep_source : bool, optional
            Whether to keep sharing the source coordinates of an indexed view, by
            default True. Moot for an owning cloud, which never shares.
        """
        return PointCloud(self._data.copy(keep_source))

    def __getitem__(self, index):
        """Index into the cloud's ``(n_points, dim)`` coordinates.

        Selecting whole points -- a slice, an integer array, or a boolean mask
        over the leading axis -- returns a :class:`PointCloud` that *shares*
        these coordinates and selects through an index array, so slicing a cloud
        costs no copy of the coordinates however large they are::

            near = pc[:100]         # a PointCloud view, no coordinates copied
            picked = pc[[3, 17, 42]]

        Every other index materializes the selected coordinates into a
        ``FloatTensor``: an integer (which drops the point axis, as in NumPy) and
        anything reaching the coordinate axis, so the natural plotting idiom
        ``pc[:, 0]`` / ``pc[:, 1]`` keeps working and still yields numbers::

            first = pc[0]           # FloatTensor of shape (dim,)
            xs = pc[:, 0]           # FloatTensor of the x coordinates
        """
        rows = self._row_selection(index, self.shape[0])
        if rows is None:
            return self.materialize()[index]

        # Indices address the source coordinates, so selecting rows of a view
        # must compose through the view's own indices rather than reuse them.
        if self.is_indexed:
            rows = np.asarray(self._data.indices, dtype=np.uint64)[rows]

        from .base_tensor import IntTensor
        cpp_cloud = type(self._data)(
            self._data.coords, IntTensor(np.ascontiguousarray(rows), dtype=uint64)._data)
        return PointCloud(cpp_cloud)

    def __setitem__(self, index, value):
        """Write a coordinate: ``pc[i, j] = value``.

        An owning cloud writes in place. Writing to an *indexed view* is
        copy-on-write: the view first detaches into a private owning copy, so
        the write lands there and the shared source cloud is never touched.
        After the write the cloud is no longer an indexed view.
        """
        if self.is_indexed:
            self._data = type(self._data)(self._data.materialize())
        # An owning cloud's materialize() shares its coordinate buffer (it is not
        # an indexed view), so writing through it mutates the cloud in place.
        self.materialize()[index] = value

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
