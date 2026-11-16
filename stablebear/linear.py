"""Inner products and linear subspaces of point clouds."""

from __future__ import annotations

from typing import Any, overload

import numpy as np

from .base_tensor import FloatTensor
from .point_cloud import PointCloud
from .typing import Dtype, pcloud32, pcloud64


def _float_array(values: Any, name: str) -> np.ndarray:
    """A float32 array for float32 input and a float64 array for any other
    real input."""
    array = np.asarray(values)
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
        raise TypeError(f"{name} must hold real numbers, not {array.dtype}.")
    return array.astype(np.float32 if array.dtype == np.float32 else np.float64)


def _as_point_cloud(points: Any) -> PointCloud:
    """A single point cloud from an (n, d) array, FloatTensor, or
    PointCloud."""
    if isinstance(points, PointCloud):
        return points
    if isinstance(points, FloatTensor):
        points = np.asarray(points)
    if isinstance(points, (np.ndarray, list, tuple)):
        array = _float_array(points, "The points")
        if array.ndim != 2:
            raise ValueError(f"A point cloud must have shape (n, d), one point per row (got shape {array.shape}).")
        return PointCloud(array)
    raise TypeError(f"Expected a point cloud, got {type(points).__name__}.")


class InnerProduct:
    r"""An inner product on vectors of dimension :math:`d`, given by its
    Gram matrix.

    The inner product of :math:`x` and :math:`y` is :math:`x^T G y` for the
    Gram matrix :math:`G`. The identity matrix gives the Euclidean inner
    product, which point clouds have without an ``InnerProduct``.

    Calling the inner product on a point cloud gives an
    :class:`InnerProductSpace`, which operations that use an inner product
    take in place of the point cloud.

    Parameters
    ----------
    gram : array_like
        The Gram matrix, of shape ``(d, d)``, symmetric and positive
        definite. A ``float32`` array gives a ``float32`` inner product,
        and any other real array a ``float64`` one.

    Raises
    ------
    ValueError
        If ``gram`` is not a square matrix, not symmetric, or not positive
        definite.
    TypeError
        If ``gram`` does not hold real numbers.

    Examples
    --------
    >>> import stablebear as sb
    >>> ip = sb.InnerProduct([[2.0, 0.0], [0.0, 1.0]])
    >>> ip.dim
    2
    >>> space = ip([[0.0, 0.0], [3.0, 4.0]])
    >>> space.inner_product == ip
    True
    """

    def __init__(self, gram: Any):
        gram = _float_array(gram, "The Gram matrix")
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
            raise ValueError(f"The Gram matrix must be square (got shape {gram.shape}).")
        if not np.allclose(gram, gram.T):
            raise ValueError("The Gram matrix must be symmetric.")
        try:
            np.linalg.cholesky(gram)
        except np.linalg.LinAlgError:
            raise ValueError("The Gram matrix must be positive definite.") from None
        self._gram = gram

    @property
    def gram(self) -> FloatTensor:
        """The Gram matrix, of shape ``(d, d)``."""
        return FloatTensor(self._gram.copy())

    @property
    def dim(self) -> int:
        """The dimension :math:`d` of the vectors."""
        return int(self._gram.shape[0])

    def __call__(self, vectors: Any) -> InnerProductSpace:
        """The vectors with this inner product.

        Parameters
        ----------
        vectors : PointCloud, FloatTensor, or numpy.ndarray
            A point cloud of shape ``(n, d)``, one point per row. Its
            points are the vectors.

        Returns
        -------
        InnerProductSpace
            The point cloud with this inner product.

        Raises
        ------
        ValueError
            If the points do not have the dimension of the inner product.
        """
        return InnerProductSpace(vectors, self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InnerProduct):
            return NotImplemented
        return np.array_equal(self._gram, other._gram)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"InnerProduct(dim={self.dim})"


class InnerProductSpace:
    """Vectors together with the inner product of their space.

    An operation that uses an inner product, such as
    :meth:`LinearSubspace.project`, takes an ``InnerProductSpace`` in place
    of the plain vectors and uses its inner product. It is usually made by
    calling an :class:`InnerProduct` on the vectors.

    Parameters
    ----------
    vectors : PointCloud, FloatTensor, or numpy.ndarray
        A point cloud of shape ``(n, d)``, one point per row. Its points
        are the vectors.
    inner_product : InnerProduct
        The inner product, on vectors of dimension ``d``.

    Raises
    ------
    ValueError
        If the points do not have the dimension of the inner product.
    """

    def __init__(self, vectors: Any, inner_product: InnerProduct):
        if not isinstance(inner_product, InnerProduct):
            raise TypeError(f"Expected an InnerProduct, got {type(inner_product).__name__}.")
        cloud = _as_point_cloud(vectors)
        if cloud.shape[1] != inner_product.dim:
            raise ValueError(
                f"The points have dimension {cloud.shape[1]}, but the inner product is on vectors of "
                f"dimension {inner_product.dim}.")
        self._vectors = cloud
        self._inner_product = inner_product

    @property
    def vectors(self) -> Any:
        """The vectors: for a point cloud, the ``PointCloud`` itself."""
        return self._vectors

    @property
    def inner_product(self) -> InnerProduct:
        """The inner product of the space."""
        return self._inner_product

    @property
    def dtype(self) -> Dtype:
        """The dtype of what holds the vectors, such as ``pcloud64``."""
        return pcloud32 if np.asarray(self._vectors).dtype == np.float32 else pcloud64

    def __repr__(self) -> str:
        return f"InnerProductSpace({self._vectors!r}, {self._inner_product!r})"


class LinearSubspace:
    r"""The linear subspace spanned by a set of vectors.

    Parameters
    ----------
    vectors : array_like
        The vectors, as the rows of an array of shape ``(k, d)``: ``k``
        vectors of dimension ``d``. They must be linearly independent, but
        need not be orthogonal or of unit length. A ``float32`` array gives
        a ``float32`` subspace, and any other real array a ``float64`` one.

    Raises
    ------
    ValueError
        If ``vectors`` is not a matrix, has no rows, or its rows are
        linearly dependent.
    TypeError
        If ``vectors`` does not hold real numbers.

    Examples
    --------
    >>> import numpy as np
    >>> import stablebear as sb
    >>> plane = sb.LinearSubspace([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> plane.dim, plane.ambient_dim
    (2, 3)
    >>> np.asarray(plane.project([[1.0, 2.0, 3.0]]))
    array([[1., 2., 0.]])
    """

    def __init__(self, vectors: Any):
        vectors = _float_array(vectors, "The vectors")
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError(
                f"The vectors must be the rows of an array of shape (k, d) with k >= 1 (got shape {vectors.shape}).")
        if np.linalg.matrix_rank(vectors) < vectors.shape[0]:
            raise ValueError("The vectors must be linearly independent.")
        self._vectors = vectors

    @property
    def vectors(self) -> FloatTensor:
        """The vectors that span the subspace, as the rows of a ``(k, d)``
        tensor."""
        return FloatTensor(self._vectors.copy())

    @property
    def dim(self) -> int:
        """The dimension :math:`k` of the subspace."""
        return int(self._vectors.shape[0])

    @property
    def ambient_dim(self) -> int:
        """The dimension :math:`d` of the vectors."""
        return int(self._vectors.shape[1])

    @overload
    def project(self, points: InnerProductSpace) -> InnerProductSpace: ...
    @overload
    def project(self, points: PointCloud | FloatTensor | np.ndarray | list) -> PointCloud: ...
    def project(self, points: Any) -> Any:
        r"""Project the points of a point cloud orthogonally onto the
        subspace.

        Each point is replaced by the point of the subspace closest to it,
        given in the coordinates of the points: the result has the shape of
        the input. Distances between the projected points are at most
        those between the points.

        What "orthogonal" and "closest" mean depends on the inner product.
        A plain point cloud has the Euclidean one, and an
        :class:`InnerProductSpace` its own. With the vectors of the subspace
        as the rows of :math:`V`, the points as the rows of :math:`X`, and
        the Gram matrix :math:`G` of the inner product, the projected points
        are the rows of

        .. math::
            X G V^T (V G V^T)^{-1} V.

        Parameters
        ----------
        points : PointCloud, FloatTensor, numpy.ndarray, or InnerProductSpace
            A point cloud of shape ``(n, d)``, one point per row, or such a
            point cloud with its inner product.

        Returns
        -------
        PointCloud or InnerProductSpace
            The projected points, of shape ``(n, d)``: a ``PointCloud`` for
            a plain point cloud, and an ``InnerProductSpace`` with the same
            inner product for an ``InnerProductSpace``.

        Raises
        ------
        ValueError
            If the points do not have the dimension of the vectors of the
            subspace.
        """
        space = points if isinstance(points, InnerProductSpace) else None
        cloud = space.vectors if space is not None else _as_point_cloud(points)
        X = np.asarray(cloud)
        if X.shape[1] != self.ambient_dim:
            raise ValueError(
                f"The points have dimension {X.shape[1]}, but the subspace is of vectors of dimension "
                f"{self.ambient_dim}.")
        V = self._vectors
        dtype = np.result_type(X.dtype, V.dtype)
        if space is None:
            projection = V.T @ np.linalg.solve(V @ V.T, V)
        else:
            G = space.inner_product._gram
            dtype = np.result_type(dtype, G.dtype)
            projection = G @ V.T @ np.linalg.solve(V @ G @ V.T, V)
        projected = PointCloud((X @ projection).astype(dtype))
        return projected if space is None else space.inner_product(projected)

    def __repr__(self) -> str:
        return f"LinearSubspace(dim={self.dim}, ambient_dim={self.ambient_dim})"
