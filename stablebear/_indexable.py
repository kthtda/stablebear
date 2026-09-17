"""The shared indexing contract for stablebear's indexable containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Indexable(ABC):
    """One indexing contract for tensors and point clouds.

    Implementors accept the NumPy index objects -- integers, slices,
    ``Ellipsis``, integer arrays and boolean masks, combined per axis in a
    tuple -- so ``X[1:4]`` and ``X[1:4, 0:2]`` mean the same thing whichever
    container ``X`` is.

    The class also carries the row-selection logic that lets a container answer
    a leading-axis index cheaply. ``_row_selection`` recognises the subset of
    indices that select whole rows and nothing else; an implementor that can
    represent a selection of rows without copying (as
    :class:`~stablebear.PointCloud` can, via an indexed view) uses it to decide
    when that is possible, and falls back to materializing otherwise.
    """

    @abstractmethod
    def __getitem__(self, key):
        """Return the element or sub-container selected by ``key``."""

    @abstractmethod
    def __setitem__(self, key, value):
        """Assign ``value`` into the region selected by ``key``."""

    @staticmethod
    def _row_selection(key, n_rows):
        """The row indices ``key`` selects, or ``None`` if it is not a pure row selection.

        A "pure row selection" indexes the leading axis only and keeps it: a
        slice, a 1-D integer array, or a 1-D boolean mask. Anything else --
        a plain integer (which drops the axis, as in NumPy), a tuple spanning
        more than one axis, ``Ellipsis``, or ``()`` -- returns ``None``, leaving
        the caller to fall back on its general path.

        Parameters
        ----------
        key : object
            The index passed to ``__getitem__``.
        n_rows : int
            Length of the leading axis, used to resolve slices and to bounds-check.

        Returns
        -------
        numpy.ndarray or None
            A ``uint64`` array of selected row indices, or ``None``.
        """
        from .base_tensor import BoolTensor, IntTensor

        # A length-1 tuple indexes the leading axis only, like the bare index.
        if isinstance(key, tuple):
            if len(key) != 1:
                return None
            key = key[0]

        if isinstance(key, slice):
            return np.arange(n_rows, dtype=np.uint64)[key]

        if isinstance(key, (BoolTensor, IntTensor, list)):
            key = np.asarray(key)
        if not isinstance(key, np.ndarray) or key.ndim != 1:
            return None

        if key.dtype == bool:
            if key.shape[0] != n_rows:
                raise IndexError(
                    f"boolean row mask has {key.shape[0]} entries but there are {n_rows} rows.")
            return np.arange(n_rows, dtype=np.uint64)[key]

        if not np.issubdtype(key.dtype, np.integer):
            return None

        # astype copies, so the negative-index shift cannot touch the caller's array.
        rows = key.astype(np.int64)
        if np.issubdtype(key.dtype, np.signedinteger):
            rows[rows < 0] += n_rows
        if rows.size and np.any((rows < 0) | (rows >= n_rows)):
            raise IndexError(
                f"row index out of range (-{n_rows} <= index < {n_rows}).")
        return np.ascontiguousarray(rows, dtype=np.uint64)
