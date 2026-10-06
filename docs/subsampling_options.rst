==========================
Shared subsampling options
==========================

These options apply to both :func:`~stablebear.random.subsample` and
:func:`~stablebear.random.subsample_relative`. See :doc:`subsampling` for
uniform sampling and :doc:`relative_subsampling` for filter-based weights.

Replacement and eligible points
===============================

Which points are eligible depends on the sampler:

* ``subsample`` uses all logical points in each input point cloud or matrix.
* ``subsample_relative`` uses points with positive weight for each
  query/distribution pair.

By default, ``replace=True`` allows the same point to be selected more than
once per sample. With ``replace=False``, each eligible point is selected at
most once.
Points are distinguished by their input indices, even when their coordinates
are identical. Uniform sampling gives eligible points equal weight; relative
sampling uses its prepared point weights.
Without replacement, each draw uses the weights of the remaining points.

``allow_partial`` chooses what happens when fewer than ``n_points`` points
are eligible:

.. list-table:: Partial-sample policies
   :header-rows: 1
   :widths: 31 23 23 23

   * - ``allow_partial``
     - Some eligible points, without replacement
     - Some eligible points, with replacement
     - No eligible points
   * - ``"keep"`` (default)
     - Return all eligible points in sampled order
     - Draw ``n_points``
     - Return an empty sample
   * - ``"no"``
     - Raise ``ValueError``
     - Draw ``n_points``
     - Raise ``ValueError``
   * - ``"drop"``
     - Return an empty sample
     - Return an empty sample
     - Return an empty sample

``"drop"`` requires at least ``n_points`` eligible points **even with
replacement**. It returns an empty sample in the corresponding output cell;
it does not remove the cell or shift the indices of other samples. With
enough eligible points, all three policies draw ``n_points``. The policy is
applied separately to each input dataset, or to each query and distribution
combination for relative subsampling. Duplicate removal happens afterward and
may further reduce the sample size.

For compatibility with existing calls, ``allow_partial=True`` means
``"keep"`` and ``allow_partial=False`` means ``"no"``. Omitting the option
now uses ``"keep"``; pass ``"no"`` to request the previous strict default.

For example, a point cloud containing two points is too small for the
threshold of three, even when repeated draws are allowed::

   import stablebear as sb

   cloud = sb.PointCloud([
       [0.0, 0.0],
       [1.0, 0.0]
   ])
   samples = sb.random.subsample(
       cloud, n_points=3, n_samples=2,
       replace=True, allow_partial="drop"
   )
   assert samples.shape == (2,)
   assert samples[0].shape == (0, 2)
   assert samples[1].shape == (0, 2)

For relative subsampling, the same policy uses the eligible region around
each query. Here the first query has two eligible points, but the second has
only one::

   reference = sb.PointCloud([
       [0.0],
       [1.0],
       [4.0],
   ])
   samples = sb.random.subsample_relative(
       reference, query=[0, 2], n_points=2,
       distribution=sb.distributions.Uniform(end=1.5),
       replace=True, allow_partial="drop",
   )
   assert samples[0, 0].shape == (2, 1)
   assert samples[1, 0].shape == (0, 1)

Duplicate removal
=================

``discard_duplicates=False`` by default. Set it to True to keep only the
first occurrence of each selected point after drawing, without redraws:

* Point clouds remove coordinate-identical points, including points with
  distinct source indices but the same coordinates.
* Distance matrices remove repeated source indices. Distinct vertices whose
  distance happens to be zero are retained.

Duplicate removal can shorten a sample even when enough points were eligible.
Eligible points are counted by index, not by distinct coordinate values.

Coordinate equality uses ordinary numeric equality: signed zeros compare
equal, as do infinities with the same sign; opposite-sign infinities are
distinct. NaN-containing points are never considered duplicates, even when
the same source index is drawn repeatedly. This matters for uniform
subsampling; relative subsampling requires finite coordinates. Infinite matrix
distances give zero sampling weight.

Generators
==========

Pass a :class:`~stablebear.random.Generator` as
``generator=sb.random.Generator(seed=...)`` for explicit seed control.
When omitted, both functions use the global generator. Each output sample
has its own random stream, independent of worker scheduling. See
:doc:`random` for seeding and reproducibility.

Inspecting selected indices
===========================

``samples.indices`` returns an independent ``NestedTensor`` of uint64 source
indices, with one index array per sample, arranged along the same axes as
``samples``. For example,
``samples.indices[0]`` selects the first sample from a single uniform input;
``samples.indices[0, 0]`` selects the first query's first relative sample when
using one distribution. Indices appear in drawn order.

Changing the returned indices does not change the samples. Assigning to
``samples`` or a view of it, or modifying a sample's coordinates or distances,
makes ``samples.indices`` return ``None`` for the entire result and its shared
views. To retain a copy of the original selections, assign
``selected_indices = samples.indices`` before making such changes.
Tensors without stored selections also return ``None``.
See :doc:`indexing` for indexed tensor operations.
