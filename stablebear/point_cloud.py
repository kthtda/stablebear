from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _sb_cpp as cpp
from ._tensor_base import Shape
from .typing import float32, float64

if TYPE_CHECKING:
    from .base_tensor import FloatTensor


class PointCloud:
    """A single (rank-2) point cloud, of shape ``(n_points, dim)``.

    May be an *indexed view* that shares another cloud's coordinates and selects
    rows through an index array (the memory-frugal output of
    :func:`stablebear.sampling.subsample_relative`). All operations work directly on the view;
    the coordinates are only copied when the cloud is converted to NumPy.

    A cloud that *owns* its coordinates is mutable: ``pc[i, j] = value`` writes a
    coordinate in place. An *indexed view* is read-only (writing to it would
    corrupt the shared source cloud); call :meth:`materialize` to obtain a
    writable owning copy.
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

    def __getitem__(self, index):
        """Index into the cloud's ``(n_points, dim)`` coordinates as a ``FloatTensor``.

        The (selected) coordinates are materialized, so the natural NumPy idiom
        ``pc[:, 0]`` / ``pc[:, 1]`` (e.g. for plotting) works directly on a cloud.
        """
        return self.materialize()[index]

    def __setitem__(self, index, value):
        """Write a coordinate in place: ``pc[i, j] = value``.

        Only an owning cloud is mutable. An indexed view shares its source
        cloud's coordinates, so writing would corrupt every cloud that shares the
        source; it raises instead — call :meth:`materialize` for a writable copy.
        """
        if self.is_indexed:
            raise TypeError(
                "indexed point-cloud views are read-only; call .materialize() "
                "for a writable copy."
            )
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
