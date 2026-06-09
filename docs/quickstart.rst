===========
Quick start
===========

After installing, import ``stablebear`` and create your first piecewise constant function (PCF) from a NumPy array of ``(time, value)`` pairs:

.. code-block:: python

    import stablebear as sb
    import numpy as np

    # A PCF that equals 1 on [0,2), 3 on [2,5), and 0 on [5,7)
    f = sb.Pcf(np.array([[0, 1],
                         [2, 3],
                         [5, 0]]))

PCFs are callable -- you can evaluate them at any time:

.. code-block:: python

    f(1.0)    # 1.0  (on the interval [0, 2))
    f(3.5)    # 3.0  (on the interval [2, 5))

They also support arithmetic:

.. code-block:: python

    g = f * 2            # scale values by 2
    h = f + g            # pointwise addition
    s = f ** 0.5         # pointwise square root

Tensors: working with collections
----------------------------------

To work with a collection of PCFs, store them in a tensor created with ``sb.zeros``:

.. code-block:: python

    f1 = sb.Pcf(np.array([[0., 5.], [2., 3.], [5., 0.]]))
    f2 = sb.Pcf(np.array([[0., 2.], [4., 7.], [8., 1.], [9., 0.]]))
    f3 = sb.Pcf(np.array([[0., 4.], [2., 3.], [3., 1.], [5., 0.]]))

    X = sb.zeros((3,))
    X[0] = f1
    X[1] = f2
    X[2] = f3

For quick experimentation, generate random data:

.. code-block:: python

    from stablebear.random import noisy_sin, noisy_cos

    sines   = noisy_sin((200,), n_points=100)   # 200 noisy sin functions
    cosines = noisy_cos((10, 50), n_points=30)  # 10 x 50 noisy cosines

Distances, norms, and reductions
---------------------------------

Compute pairwise :math:`L^p` distances and norms:

.. code-block:: python

    D = sb.pdist(X)               # pairwise L1 distance matrix
    norms = sb.lp_norm(X, p=1)    # L1 norm of each PCF

Higher-dimensional tensors support NumPy-style reductions:

.. code-block:: python

    A = sb.zeros((4, 100))
    avg = sb.mean(A, dim=1)       # mean along axis 1 -> shape (4,)

Persistent homology
--------------------

Compute persistent homology from point cloud data and convert the resulting barcodes to PCF summaries:

.. code-block:: python

    from stablebear.persistence import (compute_persistent_homology,
                                        barcode_to_stable_rank,
                                        barcode_to_betti_curve)

    # Point cloud: 50 random points in R^3
    pts = sb.zeros((1,), dtype=sb.pcloud32)
    pts[0] = np.random.rand(50, 3).astype(np.float32)

    barcodes = compute_persistent_homology(pts)    # Ripser
    sr = barcode_to_stable_rank(barcodes)          # stable rank as a PCF
    bc = barcode_to_betti_curve(barcodes)          # Betti curve as a PCF

Saving and loading
-------------------

Save tensors to disk and load them back:

.. code-block:: python

    from stablebear.io import save, load

    save(X, 'my_pcfs.sb')
    X_loaded = load('my_pcfs.sb')

GPU acceleration
-----------------

stablebear automatically uses NVIDIA GPUs when available. You can check GPU status and control execution:

.. code-block:: python

    from stablebear import gpu

    gpu.has_nvidia_gpu()             # True/False
    gpu.nvidia_gpu_count()           # number of available GPUs

See the :doc:`User guide <userguide>` for more detail on each of these topics.
