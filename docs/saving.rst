==================
Saving and loading
==================

stablebear provides a binary format for efficiently saving and loading tensors. All tensor types are supported, including PCF, numeric, point cloud, barcode, symmetric matrix tensors, etc.

Saving
======

Use :py:func:`~stablebear.save` to write a tensor to a file::

   import stablebear as sb
   from stablebear.random import noisy_sin

   X = noisy_sin((100,), n_points=50)
   sb.save(X, 'my_pcfs.sb')

You can also pass an open file object in binary write mode::

   with open('my_pcfs.sb', 'wb') as f:
       sb.save(X, f)

Pickle support
==============

All tensor types and standalone objects (``Pcf``, ``Barcode``, ``DistanceMatrix``,
``SymmetricMatrix``) support Python's ``pickle`` protocol. This means they work
with ``pickle.dumps``/``pickle.loads``, ``copy.deepcopy``, and multiprocessing::

   import pickle

   data = pickle.dumps(X)
   X_restored = pickle.loads(data)

Pickling uses stablebear's binary format internally, so it is efficient and
preserves dtype and shape.

.. note::

   Many stablebear operations (distance matrices, reductions, etc.) are already
   parallelized internally using multithreading and GPU acceleration. Layering
   Python ``multiprocessing`` on top will most likely *decrease* performance in
   these cases due to process overhead and memory duplication.


Loading
=======

Use :py:func:`~stablebear.load` to read a tensor back::

   X = sb.load('my_pcfs.sb')

The returned tensor will be of the same type and dtype as what was saved. As with ``save``, you can also pass an open file object::

   with open('my_pcfs.sb', 'rb') as f:
       X = sb.load(f)
