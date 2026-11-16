"""Inner products and linear subspaces of point clouds."""

from __future__ import annotations

from typing import Any

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
