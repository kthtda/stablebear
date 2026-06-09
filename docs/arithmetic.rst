========================
Arithmetic & Comparisons
========================

Arithmetic
==========

All tensor types with arithmetic support ``+``, ``-``, ``*``, ``/``, ``**``,
unary ``-``, and their in-place counterparts ``+=``, ``-=``, ``*=``, ``/=``,
``**=``. Numeric tensors (``FloatTensor``, ``IntTensor``) additionally support
floor division ``//`` and ``//=``.

Scalar arithmetic
-----------------

Every operator accepts a scalar on either side::

   X = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))

   Y = X * 2.0      # [2.0, 4.0, 6.0]
   Z = 10.0 + X     # [11.0, 12.0, 13.0]
   W = 10.0 / X     # [10.0, 5.0, 3.33...]
   X /= 5.0         # in-place: [0.2, 0.4, 0.6]

PCF tensors support all four operators with both ``Pcf`` operands (pointwise)
and numeric scalars. Scalar ``+`` and ``-`` shift the values, while ``*`` and
``/`` scale them::

   X = sb.zeros((5,))
   # ... fill X with PCFs ...
   X * 3.0           # scale values
   X + 10.0          # shift values up
   1.0 / X           # elementwise reciprocal
   -X                # negate
   X + some_pcf      # pointwise PCF addition

Power
-----

The ``**`` operator raises every element to a given exponent. It works for both
numeric and PCF tensors::

   X = sb.FloatTensor(np.array([4.0, 9.0, 16.0]))
   Y = X ** 0.5       # [2.0, 3.0, 4.0]
   X **= 2            # in-place: [16.0, 81.0, 256.0]

   F = sb.zeros((5,))
   # ... fill F with PCFs ...
   G = F ** 2          # square every PCF's values
   F **= 3             # cube in place

A ``RuntimeWarning`` is emitted if the result contains NaN or infinity (e.g.
raising a negative value to a fractional power).

Division
--------

For ``FloatTensor``, ``/`` performs true division as expected::

   X = sb.FloatTensor(np.array([10.0, 21.0, 35.0]))
   X / 4.0   # [2.5, 5.25, 8.75]

For ``IntTensor``, ``/`` returns a ``FloatTensor`` (float64), matching NumPy::

   A = sb.IntTensor(np.array([10, 21, 35]))
   A / 4      # FloatTensor: [2.5, 5.25, 8.75]

Floor division (``//``) rounds down to the nearest integer (e.g. ``7 // 2 = 3``
and ``-7 // 2 = -4``). It is available on ``FloatTensor`` and ``IntTensor``,
matching NumPy::

   X = sb.FloatTensor(np.array([10.5, -7.3, 21.0]))
   X // 3.0   # [3.0, -3.0, 7.0]

   A = sb.IntTensor(np.array([10, -7, 21]))
   A // 3     # IntTensor: [3, -3, 7]

Tensor-tensor arithmetic (broadcasting)
----------------------------------------

When both operands are tensors of the same type, elementwise arithmetic is
performed with `NumPy-style broadcasting
<https://numpy.org/doc/stable/user/basics.broadcasting.html>`_:

- Shapes are compared dimension-by-dimension from the right.
- Dimensions match if they are equal, or one of them is 1.
- A missing leading dimension is treated as size 1.

::

   import numpy as np
   import stablebear as sb

   A = sb.FloatTensor(np.array([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0]]))    # shape (2, 3)
   B = sb.FloatTensor(np.array([10.0, 20.0, 30.0]))    # shape (3,)

   C = A + B   # shape (2, 3) — B is broadcast along dim 0
   # C == [[11, 22, 33],
   #       [14, 25, 36]]

Both operands can be expanded at the same time::

   col = sb.FloatTensor(np.array([[1.0], [2.0]]))       # shape (2, 1)
   row = sb.FloatTensor(np.array([[10.0, 20.0, 30.0]])) # shape (1, 3)

   result = col + row   # shape (2, 3)
   # result == [[11, 21, 31],
   #            [12, 22, 32]]

In-place operators (``+=``, ``-=``, ``*=``, ``/=``) broadcast the right-hand
side but never expand the left-hand side — the output shape must equal the
shape of the left operand, just like NumPy::

   A += B          # OK:  (2,3) + (3,) -> (2,3) matches A
   # B += A        # ValueError: (3,) + (2,3) -> (2,3) != (3,)

Incompatible shapes raise ``ValueError``::

   X = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))
   Y = sb.FloatTensor(np.array([1.0, 2.0]))
   # X + Y  -> ValueError: shapes (3,) and (2,) are not broadcast-compatible

Broadcasting also works with PCF tensors::

   F = sb.zeros((4, 10))
   # ... fill F with PCFs ...

   bias = sb.zeros((10,))
   # ... fill bias ...

   adjusted = F + bias  # shape (4, 10) — bias broadcast along dim 0

broadcast_to
------------

For advanced use, :py:meth:`~stablebear._tensor_base.Tensor.broadcast_to` returns
a view of a tensor as if it had the given shape. Size-1 dimensions are
virtually repeated without copying data::

   X = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))   # shape (3,)
   view = X.broadcast_to((4, 3))                         # shape (4, 3)
   # Every row of view is [1, 2, 3]; view shares data with X


Comparisons
===========

Tensors support the comparison operators ``==``, ``!=``, ``<``, ``<=``, ``>``,
and ``>=``. Each returns a :py:class:`~stablebear.BoolTensor` containing the
element-wise result, just like NumPy::

   import numpy as np
   import stablebear as sb

   A = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))
   B = sb.FloatTensor(np.array([1.0, 9.0, 3.0]))

   result = A == B   # BoolTensor: [True, False, True]
   result = A < B    # BoolTensor: [False, True, False]

Broadcasting
------------

Comparisons follow the same broadcasting rules as arithmetic. Shapes are
compared dimension-by-dimension from the right, and size-1 or missing
dimensions are expanded::

   A = sb.FloatTensor(np.array([[1.0, 2.0],
                                [3.0, 4.0]]))   # shape (2, 2)
   B = sb.FloatTensor(np.array([1.0, 4.0]))        # shape (2,)

   result = A == B
   # BoolTensor of shape (2, 2):
   # [[True,  False],
   #  [False, True]]

Column and scalar broadcasting also work::

   col = sb.FloatTensor(np.array([[2.0], [3.0]]))  # shape (2, 1)
   result = A < col
   # BoolTensor of shape (2, 2):
   # [[True, False],
   #  [False, False]]

   scalar = sb.FloatTensor(np.array([2.0]))        # shape (1,)
   result = A >= scalar
   # BoolTensor of shape (2, 2):
   # [[False, True],
   #  [True,  True]]

Converting to Python bool
-------------------------

Calling ``bool()`` on a single-element ``BoolTensor`` returns a Python ``bool``.
For multi-element tensors, ``bool()`` raises ``ValueError``, matching NumPy's
behavior::

   A = sb.FloatTensor(np.array([1.0]))
   B = sb.FloatTensor(np.array([1.0]))
   bool(A == B)   # True

   C = sb.FloatTensor(np.array([1.0, 2.0]))
   D = sb.FloatTensor(np.array([1.0, 2.0]))
   bool(C == D)   # ValueError: more than one element

array_equal
-----------

To check whether two tensors are entirely equal (as a single ``bool``), use
:py:meth:`~stablebear._tensor_base.Tensor.array_equal`::

   A = sb.FloatTensor(np.array([1.0, 2.0, 3.0]))
   B = A.copy()

   A.array_equal(B)   # True
   A.array_equal(np.array([1.0, 2.0, 3.0]))  # also accepts NumPy arrays

   C = sb.FloatTensor(np.array([1.0, 9.0, 3.0]))
   A.array_equal(C)   # False

Tensors with different shapes always compare as not equal.

PCF comparisons
---------------

Comparison operators also work on PCF tensors. Two PCFs are equal if they have
the same breakpoints and values::

   F = sb.random.noisy_sin((3,))
   G = F.copy()

   result = F == G    # BoolTensor: [True, True, True]
   F.array_equal(G)   # True
