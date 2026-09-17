"""Built-in sampling distributions.

A distribution is a lightweight spec: it holds its parameters and builds the
matching precision-specific C++ object for the fused fast path used by
:func:`stablebear.sampling.subsample_relative`, which only accepts these
built-ins. The distance-to-weight math lives solely in the C++ functor; these
classes carry no Python copy of it.
"""

import math


class Uniform:
    r"""Uniform weight over a band of filter values :math:`[\text{low}, \text{high}]`,

    .. math::
        D(v) = \begin{cases} 1 & \text{if } \text{low} \le v \le \text{high} \\
                             0 & \text{otherwise.} \end{cases}

    Applied to the Euclidean distance, this samples uniformly from a region
    defined by distance to the query point.

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
        # Inverted comparisons so that NaN parameters fail validation too.
        low = float(low)
        if not (low >= 0):
            raise ValueError("low must be non-negative.")
        high = float("inf") if high is None else float(high)
        if not (high > low):
            raise ValueError("high must be strictly greater than low.")
        self.low = low
        self.high = high

    def _native(self, backend):
        """The matching C++ object for the given precision backend."""
        return backend.Uniform(self.low, self.high)


class Gaussian:
    r"""Unnormalized Gaussian of the filter value,

    .. math::
        D(v) = \exp\!\left(-\tfrac{1}{2}\left(\frac{v - \mu}{\sigma}\right)^2\right).

    Applied to the Euclidean distance, this concentrates sampling probability on
    reference points whose distance to the query point is near :math:`\mu`.

    Parameters
    ----------
    mean : float, optional
        Center :math:`\mu`, by default 0.0. Must be finite.
    sigma : float, optional
        Standard deviation :math:`\sigma`, by default 1.0. Must be positive
        and finite.
    """

    def __init__(self, mean=0.0, sigma=1.0):
        mean = float(mean)
        sigma = float(sigma)
        # Non-finite parameters would not error downstream — they silently
        # produce all-empty (mean=nan/inf) or plain-uniform (sigma=inf)
        # sampling — so reject them here, like Uniform does. The inverted
        # comparison makes a NaN sigma fail validation too.
        if not math.isfinite(mean):
            raise ValueError("mean must be finite.")
        if not (sigma > 0) or math.isinf(sigma):
            raise ValueError("sigma must be positive and finite.")
        self.mean = mean
        self.sigma = sigma

    def _native(self, backend):
        """The matching C++ object for the given precision backend."""
        return backend.Gaussian(self.mean, self.sigma)


__all__ = ["Gaussian", "Uniform"]
