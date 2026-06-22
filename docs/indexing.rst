====================
Indexing and Masking
====================

Tensors support NumPy-style indexing. The behavior depends on whether the index uses integers or slices.

Single-element access
---------------------

Indexing with all integers returns the element at that position::

   X = sb.zeros((10, 5))
   f = X[3, 2]   # returns a Pcf object

For a ``PcfTensor``, the returned element is a :py:class:`~stablebear.Pcf`. For a ``FloatTensor``, it is a Python float. For a ``PointCloudTensor``, it is a ``FloatTensor`` (representing the point cloud as a numeric array).

Negative integers count from the end, as in NumPy, and an out-of-range
integer raises ``IndexError``::

   X[-1, -1]   # last element
   X[-1]       # last row (shape (5,))

A NumPy integer scalar (e.g. ``np.int64(3)``) is accepted anywhere a Python
``int`` is.

Indexing a single point cloud
-----------------------------

A 0-d ``PointCloudTensor`` wraps exactly one point cloud (for example
``sb.PointCloudTensor(arr)`` built from an ``(n_points, dim)`` array). Indexing
it delegates to that cloud's ``(n_points, dim)`` array, so the natural NumPy
idiom for plotting works directly::

   pc = sb.PointCloudTensor(arr)   # arr has shape (n_points, 2)

   plt.scatter(pc[:, 0], pc[:, 1])  # x and y coordinate columns
   first_point = pc[0]              # shape (2,)

Tensors of clouds (rank ≥ 1) index over the clouds instead: ``X[i]`` returns
the ``i``-th cloud as a ``FloatTensor`` (see above), which then supports the
same column indexing.

Slicing
-------

Using slices returns a tensor (view)::

   X = sb.zeros((10, 5, 4))

   row = X[3, :, :]          # shape (5, 4)
   sub = X[2:8, 1:, 2]       # shape (6, 4)
   every_other = X[::2, :, :]  # shape (5, 5, 4)

Negative steps are supported for reversing or striding backwards::

   Y = sb.FloatTensor(np.array([1, 2, 3, 4, 5], dtype=np.float32))
   Y[::-1]       # [5, 4, 3, 2, 1]
   Y[::-2]       # [5, 3, 1]
   Y[3:0:-1]     # [4, 3, 2]

Negative slice bounds are resolved against the axis size following Python's
``slice.indices`` rules (the same as NumPy)::

   Y[-3:-1]      # [3, 4]
   Y[-2:]        # [4, 5]
   Y[:-1]        # [1, 2, 3, 4]

A zero step raises ``ValueError`` (``slice step cannot be zero``).

Views share the underlying data with the original tensor, so no data is copied.

Ellipsis and newaxis
--------------------

``...`` (``Ellipsis``) expands to as many full slices as needed so that the
remaining axes are indexed in full, and ``None`` / ``np.newaxis`` inserts a new
length-1 axis::

   X = sb.zeros((4, 6))

   X[...]          # the whole tensor
   X[..., 0]       # shape (4,)   — last axis indexed, leading axes full
   X[0, ...]       # shape (6,)
   X[None]         # shape (1, 4, 6)
   X[:, None]      # shape (4, 1, 6)

At most one ``Ellipsis`` may appear in an index. A partial integer/slice index
(fewer entries than the rank) leaves the trailing axes in full, and supplying
more indices than the rank raises ``IndexError``.

Assignment
----------

Tensors support assignment with the same indexing syntax::

   from stablebear.random import noisy_sin, noisy_cos

   A = sb.zeros((2, 10))

   # Assign noisy sin functions into the first row
   A[0, :] = noisy_sin((10,), n_points=100)

   # Assign noisy cos functions into the second row
   A[1, :] = noisy_cos((10,), n_points=15)

Individual elements can also be assigned::

   f = sb.Pcf([[0, 1.0], [1, 2.0], [3, 0.0]])
   A[0, 0] = f


Boolean masking
===============

A ``BoolTensor`` can be used as an index to select elements where the mask is
``True``. Comparison operators return ``BoolTensor`` objects, so the result of
a comparison can be used directly as a mask.

Full-shape masking
------------------

When a ``BoolTensor`` has the same shape as the tensor it indexes, the result
is a flat 1-D tensor of the elements where the mask is ``True``::

   import numpy as np

   X = sb.FloatTensor(np.array([[1, 2, 3],
                                [4, 5, 6]], dtype=np.float32))
   mask = sb.BoolTensor(np.array([[True,  False, True],
                                  [False, True,  False]]))

   X[mask]   # FloatTensor: [1, 3, 5]

This behaves the same as NumPy::

   arr = np.array([[1, 2, 3], [4, 5, 6]])
   arr[np.array([[True, False, True], [False, True, False]])]
   # array([1, 3, 5])

Assignment with a full-shape mask is also supported::

   X[mask] = 0.0          # scalar fill: set masked positions to 0
   X[mask] = some_tensor   # tensor assign: must have the right number of elements

Leading-axes masking
--------------------

A ``BoolTensor`` whose shape matches the *leading* axes of the tensor selects
along those axes, collapsing them into one and keeping the trailing axes::

   X = sb.FloatTensor(np.arange(24, dtype=np.float32).reshape(4, 6))

   row_mask = sb.BoolTensor(np.array([True, False, True, False]))
   X[row_mask]          # shape (2, 6) — rows where the mask is True

   Y = sb.FloatTensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
   mask = sb.BoolTensor(np.array([[True, False, True],
                                  [False, True, False]]))  # shape (2, 3)
   Y[mask]              # shape (3, 4)

This matches NumPy, where a ``k``-dimensional boolean mask applied to the first
``k`` axes yields a result of shape ``(n_true, *shape[k:])``.

Axis masking
------------

A 1-D ``BoolTensor`` can be used at a specific axis position alongside slices
and integer indices. This selects along that axis where the mask is ``True``,
preserving other dimensions::

   X = sb.FloatTensor(np.arange(12, dtype=np.float32).reshape(3, 4))

   col_mask = sb.BoolTensor(np.array([True, False, True, False]))
   X[:, col_mask]       # shape (3, 2) — selects columns 0 and 2

   row_mask = sb.BoolTensor(np.array([False, True, True]))
   X[row_mask, :]       # shape (2, 4) — selects rows 1 and 2

This works with slices too::

   Y = sb.FloatTensor(np.arange(60, dtype=np.float32).reshape(3, 4, 5))

   mask = sb.BoolTensor(np.array([True, False, True, False]))
   Y[:, mask, 1:4]      # shape (3, 2, 3)

Multiple masks can be used in the same expression. Each mask selects
independently along its own axis (outer indexing)::

   X = sb.FloatTensor(np.arange(12, dtype=np.float32).reshape(3, 4))

   row_mask = np.array([True, False, True])
   col_mask = np.array([False, True, True, False])
   X[row_mask, col_mask]   # shape (2, 2) — rows 0, 2 × columns 1, 2

Assignment with multiple masks is also supported::

   X[row_mask, col_mask] = -1.0   # fill selected submatrix with -1

Creating BoolTensors
--------------------

``BoolTensor`` can be created from NumPy arrays or from comparison operators::

   # From a NumPy array
   mask = sb.BoolTensor(np.array([True, False, True]))

   # From a comparison
   X = sb.FloatTensor(np.array([1, 2, 3, 4, 5], dtype=np.float32))
   threshold = sb.FloatTensor(np.full(5, 3.0, dtype=np.float32))
   mask = X > threshold   # BoolTensor: [False, False, False, True, True]

.. _masking-numpy-differences:

Differences from NumPy
----------------------

Axis masking follows **outer indexing** semantics: each mask independently
selects along its own axis. This matches what most users expect and is the
behavior described in `NEP 21 <https://numpy.org/neps/nep-0021-advanced-indexing.html>`_.

When multiple boolean masks appear in the same expression, stablebear treats them
as an outer product (each mask filters its axis independently). To get the same
result in NumPy, use ``np.ix_``::

   # stablebear
   X[row_mask, col_mask]

   # NumPy equivalent
   arr[np.ix_(row_mask, col_mask)]

In NumPy, ``arr[row_mask, col_mask]`` instead pairs elements (like ``zip``),
which requires both masks to have the same number of ``True`` values.

Similarly, when an integer index and a boolean mask appear together, stablebear
applies them left-to-right without reordering dimensions, while NumPy may
reorder axes.

For expressions with a single mask and slices (the common case), stablebear and
NumPy produce identical results.


Advanced indexing
=================

An integer array (NumPy ``ndarray`` or ``IntTensor``) can be used as an index to
gather elements along an axis, just like
`NumPy advanced indexing <https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing>`_.

Gathering
---------

Pass an integer array to select elements in a given order. Duplicates and
negative indices are supported::

   import numpy as np
   import stablebear as sb

   X = sb.FloatTensor(np.array([10, 20, 30, 40, 50], dtype=np.float32))
   X[np.array([2, 0, 4])]    # [30, 10, 50]
   X[np.array([1, 1, 2, 0])] # [20, 20, 30, 10]  — duplicates allowed
   X[np.array([-1, -2])]     # [50, 40]           — negative indices

A plain Python list is treated as an integer (or boolean) array, exactly like
the equivalent ``np.array``::

   X[[2, 0, 4]]              # [30, 10, 50]

A multi-dimensional integer index array is also supported: its shape is adopted
into the result, e.g. indexing a ``(4, 6)`` tensor with a ``(2, 2)`` index array
gives shape ``(2, 2, 6)``.

For multi-dimensional tensors, one axis can use an integer array while the
others use slices::

   A = sb.FloatTensor(np.array([[1, 2, 3, 4],
                                [5, 6, 7, 8]], dtype=np.float32))
   A[:, np.array([1, 3])]    # columns 1 and 3 → shape (2, 2)

An ``IntTensor`` can be used in place of a NumPy integer array::

   idx = sb.IntTensor(np.array([4, 1, 0]))
   X[idx]    # [50, 20, 10]

Assignment with integer indices
-------------------------------

Both scalar fill and tensor assignment work with integer array indices::

   X[np.array([1, 3])] = 0.0                       # scalar fill
   X[np.array([0, 2])] = sb.FloatTensor(...)      # tensor assign

Multiple index arrays
---------------------

Multiple integer arrays and boolean masks can be combined freely in the same
expression. Each index selects independently along its own axis (outer
indexing), consistent with the boolean masking behavior::

   arr = np.arange(12, dtype=np.float32).reshape(3, 4)
   X = sb.FloatTensor(arr)
   X[np.array([0, 2]), np.array([1, 3])]   # shape (2, 2) — rows 0, 2 × cols 1, 3

Boolean and integer indices can be mixed::

   X[np.array([True, False, True]), np.array([0, 3])]  # shape (2, 2)

Assignment works with any combination of indices::

   X[np.array([0, 2]), np.array([1, 3])] = -1.0        # scalar fill
   X[np.array([True, False, True]), np.array([0, 3])] = some_tensor  # tensor assign
