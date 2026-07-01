"""Built-in sampling distributions.

A distribution is a lightweight spec: it holds its parameters and forwards them
to the matching precision-specific C++ sampler for the fused fast path used by
:func:`stablebear.sampling.subsample_relative`, which only accepts these
built-ins. The distance-to-weight math lives solely in the C++ functor; these
classes carry no Python copy of it.
"""


class Uniform:
    r"""Uniform weight over a band of filter values :math:`[\text{low}, \text{high}]`,

    .. math::
        D(v) = \begin{cases} 1 & \text{if } \text{low} \le v \le \text{high} \\
                             0 & \text{otherwise.} \end{cases}

    Applied to the Euclidean distance, this samples uniformly from a region
    defined by distance to the query point:

    * a **disk** of radius :math:`r` (every reference point within :math:`r` of the
      query equally likely) -- ``Uniform(high=r)``;
    * an **annulus** between radii :math:`r_1` and :math:`r_2` --
      ``Uniform(low=r1, high=r2)``;
    * plain uniform sampling of the whole reference cloud -- ``Uniform()`` (the
      default, ``low=0``, ``high=`` :math:`\infty`). This case is independent of
      the query point, so a single query point suffices.

    Parameters
    ----------
    low : float, optional
        Lower band edge :math:`\text{low} \ge 0`, by default 0.0.
    high : float, optional
        Upper band edge. If ``None`` (the default) it is :math:`+\infty`, so every
        point at distance :math:`\ge` ``low`` is included. Must be strictly greater
        than ``low``.
    """

    def __init__(self, low=0.0, high=None):
        low = float(low)
        if low < 0:
            raise ValueError("low must be non-negative.")
        high = float("inf") if high is None else float(high)
        if not (high > low):
            raise ValueError("high must be strictly greater than low.")
        self.low = low
        self.high = high

    def _sample_subsets(self, backend, reference, query, sample_size, n_instances, replace, gen):
        return backend.sample_subsets_uniform(
            reference, query, self.low, self.high, sample_size, n_instances, replace, gen
        )

    def _sample_subsets_distmat(self, backend, source, query, sample_size, n_instances, replace, gen):
        return backend.sample_subsets_distmat_uniform(
            source, query, self.low, self.high, sample_size, n_instances, replace, gen
        )



class Gaussian:
    r"""Unnormalized Gaussian of the filter value,

    .. math::
        D(v) = \exp\!\left(-\tfrac{1}{2}\left(\frac{v - \mu}{\sigma}\right)^2\right).

    Applied to the Euclidean distance, this concentrates sampling probability on
    reference points whose distance to the query point is near :math:`\mu`.

    Parameters
    ----------
    mean : float, optional
        Center :math:`\mu`, by default 0.0.
    sigma : float, optional
        Standard deviation :math:`\sigma`, by default 1.0.
    """

    def __init__(self, mean=0.0, sigma=1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mean = float(mean)
        self.sigma = float(sigma)

    def _sample_subsets(self, backend, reference, query, sample_size, n_instances, replace, gen):
        return backend.sample_subsets_gaussian(
            reference, query, self.mean, self.sigma, sample_size, n_instances, replace, gen
        )

    def _sample_subsets_distmat(self, backend, source, query, sample_size, n_instances, replace, gen):
        return backend.sample_subsets_distmat_gaussian(
            source, query, self.mean, self.sigma, sample_size, n_instances, replace, gen
        )
__all__ = ["Gaussian", "Uniform"]
