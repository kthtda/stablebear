=========================================================================================================
stablebear: A computational package for discrete objects in Python and C++
=========================================================================================================

*stablebear* is a Python package with a C++/CUDA backend for GPU-accelerated computations on piecewise constant functions (PCFs) and related discrete objects such as point clouds, persistence barcodes, and distance matrices. The primary audience is practitioners within Topological Data Analysis (TDA) wanting to do statistical analysis on invariants such as *stable rank*, *Euler characteristic curves*, *Betti curves*, and so on.

The core data structures are NumPy-like multidimensional tensors supporting slicing, broadcasting, arithmetic, and reductions. Key operations include pairwise :math:`L_p` distance matrices, :math:`L_2` kernels, norms, and persistent homology (via Ripser). The resulting distance matrices and kernels can be used directly with scikit-learn and other machine learning libraries for clustering, classification, and dimensionality reduction.


.. toctree::
   :hidden:

   getting_started
   userguide

.. toctree::
   :hidden:
   :caption: Reference:

   stablebear

.. toctree::
   :hidden:
   :caption: About:

   acknowledgments

.. toctree::
   :hidden:
   :caption: More:

   internals
   changelog

.. grid:: 2

   .. grid-item-card:: :fas:`rocket` Getting started
      :link: getting_started
      :link-type: doc

      Install stablebear, quick start examples, and tutorials.

   .. grid-item-card:: :fas:`book` User guide
      :link: userguide
      :link-type: doc

      In-depth guides on core concepts, tensors, distances, persistence, and GPU acceleration.

   .. grid-item-card:: :fas:`code` API reference
      :link: stablebear
      :link-type: doc

      Detailed descriptions of the Python API.

   .. grid-item-card:: :fas:`circle-info` About
      :link: acknowledgments
      :link-type: doc

      Citing stablebear, acknowledgments, and licensing.

   .. grid-item-card:: :fas:`gears` Internals
      :link: internals
      :link-type: doc

      Architecture of the C++/CUDA engine: block execution, double buffering, and extensibility.

   .. grid-item-card:: :fas:`clock-rotate-left` Changelog
      :link: changelog
      :link-type: doc

      Release history and version notes.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


