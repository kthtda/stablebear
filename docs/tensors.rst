======================
Working with Tensors
======================

This guide covers the practical details of creating, manipulating, and persisting tensors in stablebear.

Creating tensors
================

Using zeros
-----------

The most common way to create a tensor is :py:func:`~stablebear.zeros`, which allocates a tensor of a given shape filled with "zero" elements::

   import stablebear as sb

   # 1-D tensor of 100 PCFs (32-bit, the default)
   X = sb.zeros((100,))

   # 3-D tensor of 64-bit PCFs
   Y = sb.zeros((4, 10, 25), dtype=sb.pcf64)

   # Scalar float tensor
   Z = sb.zeros((5, 5), dtype=sb.float64)

For PCF dtypes, "zero" is a function that is identically zero. For numeric dtypes, it is the number 0. For point cloud dtypes, it is an empty point cloud.

Generating random data
-----------------------

For quick experimentation, :py:mod:`stablebear.random` provides functions that generate tensors of noisy trigonometric PCFs::

   from stablebear.random import noisy_sin, noisy_cos

   # 200 noisy sin(2*pi*t) functions, each with 100 breakpoints
   sines = noisy_sin((200,), n_points=100)

   # 2-D: 10 x 50 noisy cosine functions with 30 breakpoints each
   cosines = noisy_cos((10, 50), n_points=30)

These functions return ``PcfTensor`` by default. Pass ``dtype=sb.pcf64`` for 64-bit.

From lists
----------

All tensor types can be constructed directly from Python lists or tuples::

   import stablebear as sb

   # Numeric tensors — same as wrapping in np.array()
   X = sb.FloatTensor([1.0, 2.0, 3.0])
   Y = sb.IntTensor([[1, 2], [3, 4]])
   Z = sb.BoolTensor([True, False, True])

For non-numeric tensors, nested lists define the shape::

   f = sb.Pcf([[0, 1.0], [1, 2.0]])
   g = sb.Pcf([[0, 3.0], [2, 4.0]])

   # 1-D tensor
   t = sb.PcfTensor([f, g])             # shape (2,)

   # 2-D tensor from nested lists
   t2 = sb.PcfTensor([[f, g], [g, f]])  # shape (2, 2)

This works the same way for ``IntPcfTensor`` and ``BarcodeTensor``.
The precision (32- or 64-bit) is inferred from the elements.
An empty list produces a shape ``(0,)`` tensor.


From NumPy arrays
-----------------

Point-cloud, distance-matrix, and symmetric-matrix tensors can be built
directly from a single NumPy array, avoiding an explicit element-assignment
loop.

For a :py:class:`~stablebear.PointCloudTensor`, the trailing ``cloud_ndim``
axes (2 by default) form each ``(n_points, dim)`` cloud and the leading axes
form the tensor shape::

   import numpy as np
   import stablebear as sb

   arr = np.random.rand(3, 5, 4, 2)        # 3 x 5 grid of (4, 2) clouds
   pc = sb.PointCloudTensor(arr)           # shape (3, 5)

   batch = np.random.rand(10, 8, 2)        # 10 clouds of 8 points in 2-D
   clouds = sb.PointCloudTensor(batch)     # shape (10,)

A list of cloud arrays (which may have differing numbers of points) builds a
1-D tensor::

   ragged = sb.PointCloudTensor([np.random.rand(3, 2), np.random.rand(5, 2)])

For matrix tensors, the trailing two axes of the array form each ``n x n``
matrix and the leading axes form the tensor shape::

   distances = np.zeros((6, 4, 4))         # 6 distance matrices, each 4 x 4
   dmats = sb.DistanceMatrixTensor.from_numpy(distances)   # shape (6,)

   values = np.zeros((2, 3, 5, 5))         # 2 x 3 grid of 5 x 5 matrices
   smats = sb.SymmetricMatrixTensor.from_numpy(values)     # shape (2, 3)

The precision is inferred from the array dtype (``float32`` → the 32-bit
variant, otherwise 64-bit) and can be overridden with ``dtype=``. These
batch constructors are the natural entry point for computing persistent
homology across many clouds or distance matrices in one parallel call.

The :py:func:`~stablebear.tensor` factory is a NumPy-like front end that
dispatches to the right constructor based on ``dtype``::

   X = sb.tensor([1.0, 2.0, 3.0])                  # FloatTensor (inferred)
   pc = sb.tensor(arr, dtype=sb.pcloud64)          # PointCloudTensor
   dmats = sb.tensor(distances, dtype=sb.distmat64)  # DistanceMatrixTensor


From serialized NumPy data
---------------------------

:py:func:`~stablebear.from_serial_content` constructs a tensor from PCF data already stored in NumPy arrays — a flat content array and an enumeration array that describes how to split it::

   import numpy as np
   import stablebear as sb

   # Three PCFs packed into a single content array
   content = np.array([
       [0.0, 2.5], [1.5, 1.2], [3.14, 0.0],   # PCF 0 (3 points)
       [0.0, 7.0], [3.8, 5.5], [4.5, 1.5], [7.0, 0.0],  # PCF 1 (4 points)
       [0.0, 3.0], [2.0, 0.0],                   # PCF 2 (2 points)
   ])

   # Each row gives (start, end) indices into content
   enumeration = np.array([[0, 3], [3, 7], [7, 9]])

   F = sb.from_serial_content(content, enumeration)
   # F is a PcfTensor of shape (3,)

The enumeration array can be multidimensional. If it has shape ``(n1, n2, ..., nk, 2)``, the resulting tensor has shape ``(n1, n2, ..., nk)``.


Shape and copying
=================

Every tensor has a :py:attr:`shape` property, along with ``ndim``, ``size``,
and ``len()`` — matching the NumPy interface::

   X = sb.zeros((10, 5, 4))
   X.shape        # (10, 5, 4)
   X.ndim         # 3
   X.size         # 200
   len(X)         # 10  (first axis)

To create an independent copy (not a view)::

   Y = X.copy()

To collapse all dimensions into one::

   flat = X.flatten()  # shape (200,)

To change the shape without changing the data, use ``reshape``. One dimension
may be ``-1`` to infer its size::

   X = sb.FloatTensor(np.arange(12, dtype=np.float32))
   X.reshape((3, 4))     # shape (3, 4)
   X.reshape((2, -1))    # shape (2, 6) — inferred

For contiguous tensors, ``reshape`` returns a view (shared data). For
non-contiguous tensors (e.g. from slicing with a step), it copies first.

To reverse the order of axes, use the ``.T`` property. For finer control,
``transpose`` accepts an explicit axis permutation::

   A = sb.FloatTensor(np.arange(12, dtype=np.float32).reshape(3, 4))
   A.T              # shape (4, 3)
   A.transpose((1, 0))  # same as .T for 2-D

   B = sb.FloatTensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
   B.transpose((2, 0, 1))  # shape (4, 2, 3)

Transpose always returns a view.

To swap exactly two axes, use ``swapaxes``::

   C = sb.FloatTensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
   C.swapaxes(0, 2)     # shape (4, 3, 2)
   C.swapaxes(-1, -3)   # same — negative indices count from the last axis

To remove size-1 dimensions, use ``squeeze``. With no argument it removes all
of them; with an axis argument it removes only that one::

   X = sb.FloatTensor(np.arange(6, dtype=np.float32).reshape(1, 6, 1))
   X.squeeze()      # shape (6,)
   X.squeeze(0)     # shape (6, 1)

Squeeze always returns a view. Squeezing an axis whose size is not 1 raises
``ValueError``.

The inverse operation, ``expand_dims``, inserts a size-1 dimension at the given
position (negative indexing supported)::

   Y = sb.FloatTensor(np.arange(6, dtype=np.float32))
   Y.expand_dims(0)    # shape (1, 6)
   Y.expand_dims(-1)   # shape (6, 1)

Expand dims also returns a view.


Type casting
============

``astype`` converts a tensor to a different dtype. Same-family precision changes
(e.g. float32 to float64) and numeric cross-family casts (e.g. int to float)
are supported::

   X = sb.FloatTensor(np.array([1.5, 2.5, 3.5], dtype=np.float32))
   X.astype(sb.float64)    # FloatTensor, float64
   X.astype(sb.int32)      # IntTensor, int32 (truncates)

PCF and point cloud tensors support precision changes within their family::

   F = sb.zeros((5,), dtype=sb.pcf32)
   F.astype(sb.pcf64)      # PcfTensor, pcf64

``astype`` always returns a new tensor (copy).


Joining tensors
===============

``concatenate`` joins tensors along an existing axis::

   A = sb.FloatTensor(np.array([[1, 2], [3, 4]], dtype=np.float32))  # (2, 2)
   B = sb.FloatTensor(np.array([[5, 6]], dtype=np.float32))          # (1, 2)
   sb.concatenate((A, B), axis=0)   # (3, 2)

All tensors must have the same shape except along the join axis, and must
share a dtype — joining mixed dtypes raises ``TypeError`` (cast with
``astype`` first). ``axis`` may be negative, counting from the last axis.

``stack`` joins tensors along a new axis (all shapes must match)::

   X = sb.FloatTensor(np.array([1, 2, 3], dtype=np.float32))  # (3,)
   Y = sb.FloatTensor(np.array([4, 5, 6], dtype=np.float32))  # (3,)
   sb.stack((X, Y), axis=0)    # (2, 3)
   sb.stack((X, Y), axis=1)    # (3, 2)


Splitting tensors
=================

``split`` divides a tensor into parts along an axis. Pass an integer for equal
splits, or a list of indices for custom split points::

   X = sb.FloatTensor(np.arange(12, dtype=np.float32).reshape(4, 3))

   # Equal split: 4 rows into 2 parts of 2 rows each
   a, b = sb.split(X, 2, axis=0)       # each shape (2, 3)

   # Index split: split at rows 1 and 3
   p, q, r = sb.split(X, [1, 3], axis=0)  # shapes (1,3), (2,3), (1,3)

The returned parts are views sharing data with the original tensor. An equal
split raises ``ValueError`` if the axis size is not divisible by the number of
sections. ``axis`` may be negative, and negative split indices count from the
end of the axis (clamped at 0), matching NumPy.

``array_split`` works the same way but allows uneven divisions — the first
sections get one extra element when the size is not evenly divisible::

   Y = sb.FloatTensor(np.arange(9, dtype=np.float32))
   parts = sb.array_split(Y, 4)   # sizes: 3, 2, 2, 2


Iterating
=========

Iterating over a tensor yields sub-tensors along the first axis, just like
NumPy::

   X = sb.FloatTensor(np.arange(12, dtype=np.float32).reshape(3, 4))
   for row in X:
       print(row.shape)   # (4,)

For a 1-D tensor, iteration yields scalar elements.

This also enables ``list()``, ``tuple()``, and unpacking::

   a, b, c = X   # three rows

Nested iteration works as expected::

   Y = sb.FloatTensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
   for matrix in Y:          # shape (3, 4)
       for row in matrix:    # shape (4,)
           print(row)
