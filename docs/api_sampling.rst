stablebear.sampling
===================

Relative-approach subsampling: for each query point, draw subsamples of a
reference point cloud with per-query-point probabilities formed by a *filter*
and a *distribution*. The resulting subsamples feed directly into
:func:`stablebear.persistence.compute_persistent_homology`.

:func:`~stablebear.sampling.subsample` and the built-in distributions
:class:`~stablebear.sampling.distributions.Gaussian` and
:class:`~stablebear.sampling.distributions.Uniform` are re-exported at the top
level, so the common case needs only a single import:

.. code-block:: python

    import stablebear as sb

    # Default distribution (Gaussian(mean=0.0, sigma=1.0)) on Euclidean distance:
    subs = sb.subsample(reference, query, sample_size=30, n_instances=2000)

    # Concentrate probability near distance 2.0 from each query point:
    subs = sb.subsample(reference, query, sample_size=30, n_instances=2000,
                        distribution=sb.Gaussian(mean=2.0, sigma=0.5))

    # Or sample uniformly from an annulus between radii 1.0 and 3.0:
    subs = sb.subsample(reference, query, sample_size=30, n_instances=2000,
                        distribution=sb.Uniform(inner=1.0, outer=3.0))

.. automodule:: stablebear.sampling
   :no-members:

subsample
---------

.. automodule:: stablebear.sampling.subsample
   :members:
   :undoc-members:
   :show-inheritance:

distributions
-------------

.. automodule:: stablebear.sampling.distributions
   :members:
   :undoc-members:
   :show-inheritance:
