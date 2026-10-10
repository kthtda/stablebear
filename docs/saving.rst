==================
Saving and loading
==================

stablebear provides a binary format for saving and loading numeric, PCF,
nested, point-cloud, barcode, distance-matrix, and symmetric-matrix tensors.
Standalone ``Pcf``, ``PointCloud``, ``Barcode``, ``DistanceMatrix``, and
``SymmetricMatrix`` objects use the same API.

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

Standalone objects can be saved and loaded directly::

   import numpy as np

   cloud = sb.PointCloud(np.array([
       [0.0, 0.0],
       [1.0, 0.0],
       [0.0, 1.0],
   ]))
   sb.save(cloud, 'cloud.sb')
   restored_cloud = sb.load('cloud.sb')  # a PointCloud with the same dtype

Pickle support
==============

All tensor types and standalone data objects (``Pcf``, ``PointCloud``,
``Barcode``, ``DistanceMatrix``, and ``SymmetricMatrix``) support Python's
``pickle`` protocol. This means they work with ``pickle.dumps``/``pickle.loads``,
``copy.deepcopy``, and multiprocessing::

   import pickle

   data = pickle.dumps(X)
   X_restored = pickle.loads(data)

Pickling uses the same binary writer and reader as ``save`` and ``load``,
preserving values, dtype, and shape. Nested tensors retain their recursive
structure; indexed point-cloud and distance-matrix tensors retain their
selections without first materializing their samples.

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

Compatibility
=============

.. note::

   Stablebear's policy is backward compatibility: newer releases read files
   and pickles written by older releases. Forward compatibility is not
   guaranteed: older releases may not read files or pickles written by newer
   releases.

.. note::

   The compatibility policy applies to released formats, not intermediate
   encodings from development branches. If you need to save data for long-term
   use, use a released version of Stablebear.

Stablebear retains readers for supported earlier binary formats and legacy
Python pickle representations, including files produced by Stablebear 0.4.7.
New pickles always use the shared binary format. Invalid binary payloads raise
an error rather than being retried as a legacy pickle representation.
