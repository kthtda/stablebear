=======================
Persistent Homology
=======================

This guide covers the persistent homology pipeline in stablebear: going from point cloud data to persistence barcodes to stable rank functions, and using those functions for downstream analysis.

For the *relative approach* -- drawing subsamples of a reference point cloud around chosen query points before computing persistent homology -- see the :doc:`subsampling guide <sampling>`.


Background
==========

**Persistent homology** is a tool from Topological Data Analysis (TDA) that captures the "shape" of data at multiple scales. Given a set of points (a point cloud), persistent homology tracks the appearance and disappearance of topological features -- connected components, loops, voids, etc. -- as a scale parameter increases.

The output is a **persistence barcode**: a collection of intervals :math:`[b_i, d_i)`, where each interval records the birth :math:`b_i` and death :math:`d_i` of a topological feature. Features that persist over a wide range of scales are considered significant, while short-lived features are often regarded as noise.

stablebear provides three functional summaries of persistence barcodes, all of which are piecewise constant functions:

- The (1d) **stable rank** counts, for each threshold :math:`t`, how many bars have length at least :math:`t` :footcite:`Chacholski2020,Gafvert2017,Scolamiero2017`.
- The **Betti curve** counts, for each filtration value :math:`t`, how many bars are alive at :math:`t` (see, e.g., :footcite:`Umeda2017,Chazal2021`).
- The **accumulated persistence function** (APF) sums, for each mean age :math:`m`, the lifetimes of all bars whose midpoint is at most :math:`m` :footcite:`Biscio2019`.

Because these summaries are PCFs, they fit naturally into stablebear's tensor framework, enabling efficient computation of distances and means over collections of barcodes.


The pipeline
============

A typical TDA workflow in stablebear follows three steps:

1. **Point clouds** -- Organize your data into a tensor of point clouds
2. **Barcodes** -- Compute persistent homology to get persistence barcodes
3. **Functional summaries** -- Convert barcodes to PCFs (stable ranks, Betti curves, APFs, etc.) for further analysis

.. code-block:: text

   Point clouds  ──>  Barcodes  ──>  Functional summaries (PCFs)
                                          │
                                          ├──  distances (pdist)
                                          ├──  means (mean)
                                          └──  norms (lp_norm)


Step 1: Point clouds
====================

Point clouds are stored in ``PointCloudTensor``. Each element of the tensor is a point cloud, represented internally as an :math:`n \times d` numeric array where :math:`n` is the number of points and :math:`d` is the ambient dimension.

Create a tensor and assign point clouds as NumPy arrays::

   import stablebear as sb
   import numpy as np

   # A tensor that will hold 5 point clouds
   pclouds = sb.zeros((5,), dtype=sb.pcloud64)

   # Assign random point clouds (varying number of points, 3-dimensional)
   for i in range(5):
       n_points = np.random.randint(20, 100)
       pclouds[i] = np.random.randn(n_points, 3)

Point clouds in the same tensor can have different numbers of points and, in principle, different dimensions, though in practice it is most common for all point clouds to share the same ambient dimension.

Higher-dimensional tensors work as well::

   # A 10 x 20 grid of point clouds
   pclouds = sb.zeros((10, 20), dtype=sb.pcloud64)


Step 2: Computing persistent homology
=======================================

:py:func:`~stablebear.persistence.compute_persistent_homology` takes a tensor of point clouds and returns a tensor of persistence barcodes. Barcode computation is performed using Ripser :footcite:`Bauer2021` under the hood::

   from stablebear import persistence

   bcs = persistence.compute_persistent_homology(pclouds, max_dim=1)

The ``max_dim`` parameter controls the highest homology dimension computed. With ``max_dim=1``, the function computes :math:`H_0` (connected components) and :math:`H_1` (loops).

The output tensor has one extra dimension appended, of size ``max_dim + 1``. For example:

- Input shape ``(5,)`` with ``max_dim=1`` produces output shape ``(5, 2)``
- Input shape ``(10, 20)`` with ``max_dim=2`` produces output shape ``(10, 20, 3)``

To access the :math:`H_n` barcode for a specific point cloud, index with the point cloud's position followed by ``n``::

   bc_H0 = bcs[3, 0]   # H0 barcode of point cloud 3
   bc_H1 = bcs[3, 1]   # H1 barcode of point cloud 3

Each element is a :py:class:`~stablebear.persistence.Barcode` object.

Input flexibility
-----------------

``compute_persistent_homology`` also accepts:

- A single ``FloatTensor`` (interpreted as a single point cloud)
- A plain NumPy array (interpreted as a single point cloud)
- A ``DistanceMatrix`` (precomputed pairwise distances for a single data set)
- A ``DistanceMatrixTensor`` (a tensor of precomputed distance matrices)

::

   # From a NumPy array directly
   points = np.random.randn(50, 3)
   bcs = persistence.compute_persistent_homology(points, max_dim=1)

When the input is a distance matrix, the ``distance_type`` parameter is
ignored because distances are already provided::

   from scipy.spatial.distance import pdist, squareform
   import stablebear as sb
   from stablebear import persistence

   points = np.random.randn(50, 3)
   D = squareform(pdist(points))

   dm = sb.DistanceMatrix(50, dtype=sb.float64)
   for i in range(50):
       for j in range(i):
           dm[i, j] = D[i, j]

   bcs = persistence.compute_persistent_homology(dm, max_dim=1)

Options
-------

The function supports the following options:

- ``distance_type`` -- The distance metric used between points. Currently only ``DistanceType.Euclidean`` (the default). Ignored when the input is a distance matrix.
- ``complex_type`` -- The simplicial complex construction. Currently only ``ComplexType.VietorisRips`` (the default).
- ``reduced`` -- If ``True``, compute reduced homology. If ``False`` (the default), an essential ``[0, inf)`` bar is added to :math:`H_0` representing the single connected component that never dies. This matches the convention used by most TDA textbooks.
- ``verbose`` -- Print progress information (default ``False``).


Step 3: Functional summaries
=============================

Persistence barcodes can be converted to piecewise constant functions for
downstream analysis. Because the results are PCFs, they fit naturally into
stablebear's tensor framework, enabling distances, means, and norms.

Stable ranks
-------------

:py:func:`~stablebear.persistence.barcode_to_stable_rank` converts barcodes into
stable rank PCFs. The stable rank counts, for each threshold :math:`t`, the
number of bars with length (death minus birth) strictly greater than :math:`t`
:footcite:`Chacholski2020`::

   sranks = persistence.barcode_to_stable_rank(bcs)

The output tensor has the same shape as the input.

.. image:: _static/gallery_tda_pipeline_light.png
   :width: 100%
   :class: only-light

.. image:: _static/gallery_tda_pipeline_dark.png
   :width: 100%
   :class: only-dark

.. dropdown:: Show code
   :color: secondary

   .. literalinclude:: _static/gen_plotting_gallery.py
      :language: python
      :start-after: docs snippet start tda_pipeline --
      :end-before: docs snippet end tda_pipeline --

Betti curves
-------------

:py:func:`~stablebear.persistence.barcode_to_betti_curve` converts barcodes into
Betti curves. The Betti curve counts, for each filtration value :math:`t`, the
number of bars alive at :math:`t` (i.e., bars with birth :math:`\leq t <`
death)::

   bettis = persistence.barcode_to_betti_curve(bcs)

The output tensor has the same shape as the input.

.. image:: _static/gallery_betti_pipeline_light.png
   :width: 100%
   :class: only-light

.. image:: _static/gallery_betti_pipeline_dark.png
   :width: 100%
   :class: only-dark

.. dropdown:: Show code
   :color: secondary

   .. literalinclude:: _static/gen_plotting_gallery.py
      :language: python
      :start-after: docs snippet start betti_pipeline --
      :end-before: docs snippet end betti_pipeline --

Accumulated persistence functions
----------------------------------

:py:func:`~stablebear.persistence.barcode_to_accumulated_persistence` converts
barcodes into accumulated persistence functions (APFs) :footcite:`Biscio2019`. For each mean
age :math:`m`, the APF sums the lifetimes of all bars whose midpoint is at
most :math:`m`:

.. math::

   \mathrm{APF}(m) = \sum_{i=1}^{N} \ell_i \, \mathbf{1}(m_i \leq m)

where :math:`N` is the number of bars, :math:`\ell_i = d_i - b_i` is the
lifetime, :math:`m_i = (b_i + d_i)/2` is the midpoint of bar :math:`i`, and
:math:`\mathbf{1}(\cdot)` is the indicator function.

::

   from stablebear import persistence

   apfs = persistence.barcode_to_accumulated_persistence(bcs)

.. image:: _static/gallery_apf_light.png
   :width: 100%
   :class: only-light

.. image:: _static/gallery_apf_dark.png
   :width: 100%
   :class: only-dark

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_plotting_gallery.py
      :language: python
      :start-after: docs snippet start apf --
      :end-before: docs snippet end apf --

An optional ``max_death`` parameter excludes bars whose death time exceeds
a given threshold. This corresponds to Equation (2) in :footcite:`Biscio2019`, where a
filtration cutoff :math:`T` limits the computation to bars with
:math:`d_i \leq T`::

   apfs = persistence.barcode_to_accumulated_persistence(bcs, max_death=25.0)

.. image:: _static/gallery_apf_max_death_light.png
   :width: 100%
   :class: only-light

.. image:: _static/gallery_apf_max_death_dark.png
   :width: 100%
   :class: only-dark

.. dropdown:: Show plotting code
   :color: secondary

   .. literalinclude:: _static/gen_plotting_gallery.py
      :language: python
      :start-after: docs snippet start apf_max_death --
      :end-before: docs snippet end apf_max_death --

Using functional summaries
---------------------------

Since stable ranks and Betti curves are PCFs, they are stored in PCF tensors
and support all of stablebear's standard operations::

   import stablebear as sb
   from stablebear.plotting import plot as plotpcf
   import matplotlib.pyplot as plt

   # Plot the H1 stable ranks
   plotpcf(sranks[:, 1])
   plt.title('H1 stable ranks')
   plt.show()

   # Compute distances between H1 stable ranks
   D = sb.pdist(sranks[:, 1], verbose=False)

   # Compute the mean H1 stable rank
   avg = sb.mean(sranks[:, 1], dim=0)


Complete example
================

The following example creates a multidimensional tensor of random point clouds, computes persistent homology, converts to stable ranks, and visualizes the result::

   import stablebear as sb
   from stablebear import persistence
   from stablebear.plotting import plot as plotpcf
   import numpy as np
   import matplotlib.pyplot as plt

   shape = (10, 20)
   pcloud_dim = 4

   # Create and fill point cloud tensor
   pclouds = sb.zeros(shape, dtype=sb.pcloud64)
   for i in range(shape[0]):
       for j in range(shape[1]):
           n_points = np.random.randint(20, 100)
           pclouds[i, j] = np.random.randn(n_points, pcloud_dim)

   # Compute persistent homology (H0 and H1)
   bcs = persistence.compute_persistent_homology(pclouds, max_dim=1)
   print(bcs.shape)   # (10, 20, 2)

   # Convert to stable ranks
   sranks = persistence.barcode_to_stable_rank(bcs)
   print(sranks.shape) # (10, 20, 2)

   # Plot H1 stable ranks for the first row of point clouds
   plotpcf(sranks[0, :, 1])
   plt.title('H1 stable ranks for pclouds[0, :, :]')
   plt.show()

   # Distance matrix between H1 stable ranks in the first row
   D = sb.pdist(sranks[0, :, 1], verbose=False)


The Barcode class
=================

Individual barcodes are represented by :py:class:`~stablebear.persistence.Barcode`. A barcode can be constructed from an :math:`n \times 2` NumPy array of ``(birth, death)`` pairs::

   from stablebear.persistence import Barcode

   bc = Barcode(np.array([[0.0, 1.5],
                           [0.2, 3.0],
                           [0.5, 0.8]]))

You can also convert a single barcode to a stable rank::

   from stablebear.persistence import barcode_to_stable_rank

   sr = barcode_to_stable_rank(bc)
   # sr is a Pcf


References
==========


.. footbibliography::
