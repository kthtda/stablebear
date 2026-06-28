======================
Distances and Norms
======================

stablebear provides GPU-accelerated computation of pairwise distance matrices and norms for collections of piecewise constant functions. These are key building blocks for downstream machine learning tasks such as clustering, classification, and dimensionality reduction.


Mathematical background
========================

Given two PCFs :math:`f` and :math:`g`, the :math:`L_p` distance between them is

.. math::

   d_p(f, g) = \left( \int_0^\infty |f(t) - g(t)|^p \, dt \right)^{1/p}.

The :math:`L_p` norm of a single PCF :math:`f` is

.. math::

   \| f \|_p = \left( \int_0^\infty |f(t)|^p \, dt \right)^{1/p}.

For piecewise constant functions, these integrals reduce to finite sums over the breakpoint intervals, which stablebear evaluates exactly.


Distance between two PCFs
=========================

:py:func:`~stablebear.lp_distance` computes the :math:`L_p` distance between two individual PCFs, returning a scalar.

::

   import stablebear as sb

   f = sb.Pcf([[0.0, 3.0], [1.0, 0.0]])
   g = sb.Pcf([[0.0, 1.0], [2.0, 0.0]])

   d = sb.lp_distance(f, g)        # L1 distance (default)
   d2 = sb.lp_distance(f, g, p=2)  # L2 distance

This is the simplest distance API and is useful for quick comparisons or unit testing. For computing distances between all pairs in a collection, use :py:func:`~stablebear.pdist` or :py:func:`~stablebear.cdist` instead.


Pairwise distances
==================

:py:func:`~stablebear.pdist` computes the full pairwise :math:`L_p` distance matrix for a 1-D tensor of PCFs. The result is a :py:class:`~stablebear.DistanceMatrix` — a compressed symmetric matrix where entry :math:`(i, j)` is the :math:`L_p` distance between the :math:`i`-th and :math:`j`-th PCF.

Basic usage
-----------

::

   import stablebear as sb
   from stablebear.random import noisy_sin

   X = noisy_sin((50,), n_points=100)

   # L1 distance matrix (default)
   D = sb.pdist(X)
   # D.size is 50, D[i, j] gives the distance between X[i] and X[j]

The ``p`` parameter controls which :math:`L_p` distance is computed::

   D1 = sb.pdist(X, p=1)   # L1 distance
   D2 = sb.pdist(X, p=2)   # L2 distance

Progress output
---------------

By default, progress output is suppressed. To show a progress bar during computation, pass ``verbose=True``::

   D = sb.pdist(X, verbose=True)

Input requirements
------------------

``pdist`` requires a **1-D** PCF tensor. If your data is in a higher-dimensional tensor, slice out the dimension you want first::

   A = sb.zeros((2, 50))
   # ... fill A ...

   # Distances between the 50 PCFs in the first row
   D = sb.pdist(A[0, :])

GPU acceleration
-----------------

For large collections of PCFs, ``pdist`` automatically offloads the computation to the GPU when one is available. The number of pairwise integrals grows as :math:`n(n-1)/2`, so GPU acceleration can provide dramatic speedups for large :math:`n`.


Cross-distances
================

:py:func:`~stablebear.cdist` computes the pairwise :math:`L_p` distances between two tensors of PCFs. Unlike ``pdist``, the two input tensors can differ in shape and size. The result is a :py:class:`~stablebear.FloatTensor` whose shape is the concatenation of the two input shapes.

::

   import stablebear as sb
   from stablebear.random import noisy_sin, noisy_cos

   X = noisy_sin((30,), n_points=100)
   Y = noisy_cos((20,), n_points=100)

   D = sb.cdist(X, Y)          # shape (30, 20)
   D2 = sb.cdist(X, Y, p=2)   # L2 cross-distances

Higher-dimensional tensors are supported -- the output shape is ``(*X.shape, *Y.shape)``::

   A = noisy_sin((5, 10), n_points=50)
   B = noisy_cos((8,), n_points=50)

   D = sb.cdist(A, B)   # shape (5, 10, 8)


L2 kernel matrices
===================

:py:func:`~stablebear.l2_kernel` computes the pairwise :math:`L_2` inner-product (kernel) matrix for a 1-D tensor of PCFs. The result is a :py:class:`~stablebear.SymmetricMatrix` where entry :math:`(i, j)` is

.. math::

   K_{ij} = \langle f_i, f_j \rangle_{L_2}
          = \int_0^\infty f_i(t) \, f_j(t) \, dt.

::

   K = sb.l2_kernel(X)
   # K[i, j] gives the L2 inner product between X[i] and X[j]

The kernel matrix is symmetric and includes diagonal entries (self inner products). To use it with scikit-learn, convert to a dense NumPy array::

   K_dense = sb.l2_kernel(X, verbose=False).to_dense()


Distance matrices
==================

:py:class:`~stablebear.DistanceMatrix` provides a compressed storage format for distance matrices. Since a distance matrix is symmetric with zeros on the diagonal, it stores only the strict lower triangle — :math:`n(n-1)/2` elements instead of :math:`n^2`. Entries are enforced to be nonnegative, and writes to the diagonal are rejected unless the value is zero.

::

   from stablebear import DistanceMatrix
   from stablebear.typing import float32

   m = DistanceMatrix(100, dtype=float32)
   m[3, 7] = 2.5
   assert m[7, 3] == 2.5   # symmetric access
   assert m[3, 3] == 0.0   # diagonal is always zero

To convert to a full NumPy array::

   dense = m.to_dense()   # shape (100, 100), dtype float32

Besides scalar ``m[i, j]`` access, the matrix exposes ``m.shape`` (``(n, n)``),
``len(m)`` (``n``), and ``m.diagonal()`` (all zeros for a distance matrix). Any
non-``(i, j)`` index is applied to the dense matrix, so ``m[i]`` / ``m[i, :]``
give a row, ``m[:, j]`` a column, and ``m[a:b, c:d]`` a sub-array (NumPy arrays).

Tensors of distance matrices
------------------------------

Distance matrices can be stored in tensors just like PCFs or point clouds.
Use the ``distmat32`` or ``distmat64`` dtypes::

   import stablebear as sb

   # A 1-D tensor holding 10 distance matrices
   T = sb.zeros((10,), dtype=sb.distmat64)

   # Assign a matrix into the tensor
   m = sb.DistanceMatrix(5, dtype=sb.float64)
   m[0, 1] = 3.14
   T[0] = m

   # Read it back — symmetric access works as expected
   T[0][0, 1]   # 3.14
   T[0][1, 0]   # 3.14

Slicing, copying, flattening, and equality comparison all work as for other
tensor types::

   sub = T[2:5]           # slice — shape (3,)
   T2 = T.copy()          # independent deep copy
   flat = T.flatten()     # shape (10,)
   T == T2                # True


Symmetric matrices
===================

:py:class:`~stablebear.SymmetricMatrix` provides a more general compressed storage format for symmetric matrices without the distance matrix constraints. It stores the lower triangle including the diagonal — :math:`n(n+1)/2` elements instead of :math:`n^2`.

::

   from stablebear import SymmetricMatrix
   from stablebear.typing import float32

   m = SymmetricMatrix(100, dtype=float32)
   m[3, 7] = 2.5
   assert m[7, 3] == 2.5   # symmetric access

To convert to a full NumPy array::

   dense = m.to_dense()   # shape (100, 100), dtype float32

As for :py:class:`~stablebear.DistanceMatrix`, the matrix exposes ``m.shape``
(``(n, n)``), ``len(m)`` (``n``), and ``m.diagonal()`` (the stored diagonal).
Any non-``(i, j)`` index is applied to the dense matrix, so ``m[i]`` /
``m[i, :]`` give a row, ``m[:, j]`` a column, and ``m[a:b, c:d]`` a sub-array.

Tensors of symmetric matrices use the ``symmat32`` or ``symmat64`` dtypes::

   T = sb.zeros((10,), dtype=sb.symmat64)


Norms
=====

:py:func:`~stablebear.lp_norm` computes the :math:`L_p` norm of every PCF in a tensor, returning a NumPy array of the same shape.

::

   import stablebear as sb
   from stablebear.random import noisy_sin

   X = noisy_sin((50,), n_points=100)

   # L1 norms of all 50 PCFs
   norms = sb.lp_norm(X, p=1)
   # norms.shape is (50,)

For higher-dimensional tensors, the output shape matches the input::

   A = sb.zeros((3, 10))
   # ... fill A ...

   norms = sb.lp_norm(A, p=1)
   # norms.shape is (3, 10)


Using distances in machine learning
=====================================

``pdist`` returns a :py:class:`~stablebear.DistanceMatrix`. Call :py:meth:`~stablebear.DistanceMatrix.to_dense` to obtain a standard NumPy array for use with scikit-learn and other libraries.

Clustering
----------

::

   from sklearn.cluster import AgglomerativeClustering

   D = sb.pdist(X, verbose=False).to_dense()

   clustering = AgglomerativeClustering(
       n_clusters=3,
       metric='precomputed',
       linkage='average'
   )
   labels = clustering.fit_predict(D)

Multidimensional scaling
-------------------------

::

   from sklearn.manifold import MDS

   D = sb.pdist(X, verbose=False).to_dense()

   mds = MDS(n_components=2, dissimilarity='precomputed')
   coords = mds.fit_transform(D)

Classification with distance-based kernels
--------------------------------------------

A common approach is to convert a distance matrix into a kernel matrix and use it with a kernel SVM. For example, using a Gaussian (RBF) kernel:

.. math::

   K_{ij} = \exp\!\left(-\frac{D_{ij}^2}{2\sigma^2}\right)

::

   import numpy as np
   from sklearn.svm import SVC

   D = sb.pdist(X, verbose=False).to_dense()
   sigma = np.median(D[D > 0])
   K = np.exp(-D**2 / (2 * sigma**2))

   clf = SVC(kernel='precomputed')
   clf.fit(K, labels)
