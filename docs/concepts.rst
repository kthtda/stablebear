==============
Core Concepts
==============

This page introduces the foundational ideas behind stablebear: what piecewise constant functions are, how they are stored in tensors, and how the type system works.

Piecewise constant functions
============================

A **piecewise constant function** (PCF) is a function that takes a constant value on each of a finite number of intervals. For example, the function

.. math::

   f(t) = \begin{cases}
     1 & \text{if } 0 \le t < 2 \\
     3 & \text{if } 2 \le t < 5 \\
     0 & \text{if } 5 \le t < 7
   \end{cases}

is a PCF with three pieces.

In stablebear, a PCF is represented as an :math:`n \times 2` array of ``(time, value)`` pairs, where each row gives a breakpoint time and the value the function takes starting at that time. The function above would be represented as::

   [[0, 1],
    [2, 3],
    [5, 0]]

The value at each breakpoint is the value on the interval starting at that time and continuing until the next breakpoint (or until the end of the function's domain).

Why PCFs?
---------

Many invariants in **Topological Data Analysis** (TDA) are naturally piecewise constant. Examples include:

- **Stable rank functions**
- **Betti curves** -- the Betti number as a function of the filtration parameter
- **Euler characteristic curves** -- the Euler characteristic as a function of the filtration parameter

By representing these invariants as PCFs, stablebear enables efficient statistical analysis: computing means, distances, and norms over large collections of such functions, potentially leveraging GPU acceleration.

The Pcf class
--------------

An individual PCF is represented by :py:class:`~stablebear.Pcf`. You create one from a NumPy array or a list::

   import numpy as np
   import stablebear as sb

   # From a NumPy array
   data = np.array([[0.0, 1.0],
                     [2.0, 3.0],
                     [5.0, 0.0]], dtype=np.float32)
   f = sb.Pcf(data)

   # From a list (defaults to float32)
   g = sb.Pcf([[0, 1], [2, 3], [5, 0]])

You can convert a ``Pcf`` back to a NumPy array with :py:meth:`~stablebear.Pcf.to_numpy`::

   arr = f.to_numpy()  # shape (3, 2), dtype float32

Individual PCFs support arithmetic (``+``, ``-``, ``*``, ``/``, ``**``) with
other PCFs and with scalars::

   f = sb.Pcf([[0, 4.0], [1, 9.0]])
   g = f ** 0.5    # square root: values become [2.0, 3.0]
   h = f * 2.0     # scale: values become [8.0, 18.0]

See :doc:`arithmetic` for the full arithmetic reference, including broadcasting.

Iterating over rectangles
--------------------------

Given two PCFs :math:`f` and :math:`g`, :py:func:`~stablebear.iterate_rectangles` produces the list of intervals on which both functions are constant. Each interval is returned as a :py:class:`~stablebear.functional.pcf.Rectangle` with four properties: ``left`` and ``right`` (the time boundaries) and ``f_value`` and ``g_value`` (the values of each function on that interval).

This corresponds to the rectangle decomposition described in :footcite:`Wehlin2024` and is useful for inspecting how two PCFs interact or for implementing custom integration-like operations::

   import stablebear as sb

   f = sb.Pcf([[0, 1.0], [2, 3.0], [5, 0.0]])
   g = sb.Pcf([[0, 2.0], [3, 1.0], [5, 0.0]])

   rects = sb.iterate_rectangles(f, g)
   for r in rects:
       print(f"[{r.left}, {r.right}): f={r.f_value}, g={r.g_value}")
   # [0.0, 2.0): f=1.0, g=2.0
   # [2.0, 3.0): f=3.0, g=2.0
   # [3.0, 5.0): f=3.0, g=1.0
   # [5.0, inf): f=0.0, g=0.0

Optional ``a`` and ``b`` parameters restrict the iteration to a sub-interval::

   rects = sb.iterate_rectangles(f, g, a=1.0, b=4.0)

.. note::

   ``iterate_rectangles`` is intended for exploration and prototyping. For
   performance-critical workloads such as computing distances or norms over
   large collections, use the dedicated functions (:py:func:`~stablebear.pdist`,
   :py:func:`~stablebear.lp_norm`, etc.) which are implemented in optimized
   C++/CUDA.

Tensors
=======

While you *can* work with individual ``Pcf`` objects, stablebear is designed for working with **collections** of PCFs. These collections are stored in **tensors** -- multidimensional arrays, similar to NumPy's ``ndarray``.

A tensor can have any number of dimensions. For example:

- A 1-D tensor of shape ``(100,)`` holds 100 PCFs.
- A 2-D tensor of shape ``(10, 50)`` holds 500 PCFs arranged in a 10-by-50 grid.

Creating tensors
----------------

The primary way to create a tensor is with :py:func:`~stablebear.zeros`::

   import stablebear as sb

   # A 1-D tensor of 100 "zero" PCFs (32-bit, the default)
   X = sb.zeros((100,))

   # A 2-D tensor of 64-bit PCFs
   Y = sb.zeros((10, 50), dtype=sb.pcf64)

   # A tensor of scalar floats
   Z = sb.zeros((5, 5), dtype=sb.float32)

What "zero" means depends on the dtype: for PCF types, it is a function that is identically zero; for numeric types, it is the number 0; for point cloud types, it is an empty point cloud.

You can also generate random PCF tensors for experimentation::

   from stablebear.random import noisy_sin, noisy_cos

   # 200 noisy sin(2*pi*t) functions, each sampled at 100 time points
   sines = noisy_sin((200,), n_points=100)

   # A 2-D array: 10 x 50 noisy cosine functions
   cosines = noisy_cos((10, 50), n_points=30)

Indexing and slicing
---------------------

Tensors support NumPy-style indexing and slicing::

   X = sb.zeros((10, 5, 4))

   # Single element -- returns a Pcf
   f = X[3, 2, 1]

   # Slicing -- returns a tensor (view)
   row = X[3, :, :]        # shape (5, 4)
   sub = X[2:8, 1:, 2]     # shape (6, 4)

You can also assign into tensors::

   from stablebear.random import noisy_sin

   A = sb.zeros((2, 10))
   A[0, :] = noisy_sin((10,), n_points=100)
   A[1, :] = noisy_sin((10,), n_points=50)

Boolean masks can select elements by condition::

   import numpy as np

   X = sb.FloatTensor(np.arange(12, dtype=np.float32).reshape(3, 4))
   mask = sb.BoolTensor(np.array([True, False, True, False]))

   X[:, mask]   # shape (3, 2) — select columns where mask is True
   X[X > threshold]  # flat 1D — all elements matching the condition

See :doc:`indexing` for full details on :ref:`boolean masking <masking-numpy-differences>`.

Tensor types
------------

There are several concrete tensor types, each corresponding to a dtype:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Tensor class
     - dtype
     - Contents
   * - ``PcfTensor``
     - ``pcf32`` / ``pcf64``
     - Piecewise constant functions
   * - ``IntPcfTensor``
     - ``pcf32i`` / ``pcf64i``
     - Integer-valued piecewise constant functions
   * - ``FloatTensor``
     - ``float32`` / ``float64``
     - Floating-point scalars
   * - ``IntTensor``
     - ``int32`` / ``int64`` / ``uint32`` / ``uint64``
     - Integer scalars
   * - ``PointCloudTensor``
     - ``pcloud32`` / ``pcloud64``
     - Point clouds
   * - ``BarcodeTensor``
     - ``barcode32`` / ``barcode64``
     - Persistence barcodes
   * - ``SymmetricMatrixTensor``
     - ``symmat32`` / ``symmat64``
     - Symmetric matrices
   * - ``DistanceMatrixTensor``
     - ``distmat32`` / ``distmat64``
     - Distance matrices
   * - ``BoolTensor``
     - ``boolean``
     - Boolean values (returned by comparison operators)

You can construct tensor types directly from Python lists::

   # Numeric tensors
   X = sb.FloatTensor([1.0, 2.0, 3.0])
   Y = sb.IntTensor([[1, 2], [3, 4]])

   # Non-numeric tensors from lists of elements
   f = sb.Pcf([[0, 1.0], [1, 2.0]])
   g = sb.Pcf([[0, 3.0], [2, 4.0]])
   T = sb.PcfTensor([f, g])

You can also use :py:func:`~stablebear.zeros` or functions like :py:func:`~stablebear.random.noisy_sin` that return the appropriate tensor type automatically. See :doc:`tensors` for all construction methods.


Evaluation
==========

Since PCFs represent functions, they can be evaluated by calling them as such. Pass a single time to get a single value, or an array of times to evaluate at many points at once::

   f = sb.Pcf([[0, 1], [2, 3], [5, 0]])

   f(1.0)    # 1.0  (on the interval [0, 2))
   f(3.5)    # 3.0  (on the interval [2, 5))

PCF tensors are also callable -- evaluating every element at the given time(s)::

   X = sb.zeros((3, 4), dtype=sb.pcf32)
   # ... fill X with PCFs ...

   X(1.5)        # shape (3, 4) -- one value per PCF
   X([0, 1, 5])  # shape (3, 4, 3) -- each PCF evaluated at 3 times

See :doc:`tensors` for full details on scalar, array, and tensor evaluation, including accepted input types and output shapes.


The dtype system
================

The ``dtype`` parameter controls the element type of a tensor, analogous to NumPy's ``dtype``. stablebear defines the following dtypes in :py:mod:`stablebear.typing` (also re-exported from the top-level ``stablebear`` module):

.. list-table:: Dtype overview
   :header-rows: 1
   :widths: 25 15 60

   * - dtype
     - Precision
     - Description
   * - ``pcf32`` / ``pcf64``
     - float
     - Piecewise constant functions (``pcf32`` is the default)
   * - ``pcf32i`` / ``pcf64i``
     - int
     - Integer-valued piecewise constant functions
   * - ``float32`` / ``float64``
     - float
     - Scalar floating-point values
   * - ``int32`` / ``int64`` / ``uint32`` / ``uint64``
     - int
     - Scalar integer values (signed and unsigned)
   * - ``pcloud32`` / ``pcloud64``
     - float
     - Point clouds
   * - ``barcode32`` / ``barcode64``
     - float
     - Persistence barcodes
   * - ``symmat32`` / ``symmat64``
     - float
     - Symmetric matrices — n(n+1)/2 storage
   * - ``distmat32`` / ``distmat64``
     - float
     - Distance matrices — n(n-1)/2 storage, zero diagonal, nonnegative

PCF types
---------

- :py:class:`~stablebear.pcf32` -- 32-bit floating-point piecewise constant functions (the default dtype)
- :py:class:`~stablebear.pcf64` -- 64-bit floating-point piecewise constant functions
- :py:class:`~stablebear.pcf32i` -- 32-bit integer piecewise constant functions
- :py:class:`~stablebear.pcf64i` -- 64-bit integer piecewise constant functions

Use ``pcf32`` for most work. Use ``pcf64`` when you need higher numerical precision.
``pcf32i`` and ``pcf64i`` provide integer-valued PCFs. They support construction,
evaluation, arithmetic, and serialization, but not norms or distances.

Numeric types
-------------

- :py:class:`~stablebear.float32` -- 32-bit floating-point scalars
- :py:class:`~stablebear.float64` -- 64-bit floating-point scalars

These are used for tensors that hold scalar values, such as the results of norm or distance computations.

- :py:class:`~stablebear.int32` -- 32-bit signed integer scalars
- :py:class:`~stablebear.int64` -- 64-bit signed integer scalars
- :py:class:`~stablebear.uint32` -- 32-bit unsigned integer scalars
- :py:class:`~stablebear.uint64` -- 64-bit unsigned integer scalars

These are used for tensors that hold integer values.

Point cloud types
-----------------

- :py:class:`~stablebear.pcloud32` -- 32-bit point clouds
- :py:class:`~stablebear.pcloud64` -- 64-bit point clouds

Used when working with point cloud data, e.g., as input to persistent homology computations.

Barcode types
-------------

- :py:class:`~stablebear.barcode32` -- 32-bit persistence barcodes
- :py:class:`~stablebear.barcode64` -- 64-bit persistence barcodes

Used to store persistence barcodes produced by homology computations.

Symmetric matrix types
-----------------------

- :py:class:`~stablebear.symmat32` -- 32-bit symmetric matrices
- :py:class:`~stablebear.symmat64` -- 64-bit symmetric matrices

Compressed symmetric matrices using lower-triangular storage (n*(n+1)/2 elements for an n×n matrix).

Distance matrix types
-----------------------

- :py:class:`~stablebear.distmat32` -- 32-bit distance matrices
- :py:class:`~stablebear.distmat64` -- 64-bit distance matrices

Compressed distance matrices with implicit zero diagonal and nonnegative entries (n*(n-1)/2 elements for an n×n matrix).

Precision: 32-bit vs. 64-bit
------------------------------

Each dtype family comes in 32-bit and 64-bit variants. The 32-bit variants use less memory and are faster, especially on GPUs where single-precision throughput is typically much higher. Use 64-bit variants when numerical precision is important for your application.


CPU and GPU execution
=====================

stablebear automatically detects available NVIDIA GPUs and uses them for computations when beneficial. The library decides at runtime whether to execute a given operation on the CPU or GPU based on problem size. See :doc:`gpu` for details on GPU detection, controlling execution, and performance considerations.


References
==========


.. footbibliography::
