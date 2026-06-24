import numpy as np


class _BuiltinDistribution:
    """Base for distributions that have a fused C++ ``distance`` fast path.

    A built-in distribution is both:

    * callable as ``distribution(values) -> weights`` (used when the filter is a
      custom Python callable and weights are computed in Python), and
    * able to invoke its matching backend method via :meth:`_fused_call` (used
      when the filter is the built-in ``"distance"`` and everything runs in C++).
    """

    def _fused_call(self, backend, reference, query, sample_size, n_instances, replace, gen):
        raise NotImplementedError


class Uniform(_BuiltinDistribution):
    r"""Uniform weight over a distance band :math:`[\text{inner}, \text{outer}]`,

    .. math::
        D(v) = \begin{cases} 1 & \text{if } \text{inner} \le v \le \text{outer} \\
                             0 & \text{otherwise.} \end{cases}

    With the default ``"distance"`` filter this samples uniformly from a region
    defined by distance to the query point:

    * a **disk** of radius :math:`r` (every reference point within :math:`r` of
      the query equally likely) — ``Uniform(outer=r)``;
    * a **circle**/annulus between radii :math:`r_1` and :math:`r_2` —
      ``Uniform(inner=r1, outer=r2)``;
    * plain uniform sampling of the whole reference cloud — ``Uniform()``
      (the default, ``inner=0``, ``outer=`` :math:`\infty`). This case is
      independent of the query point, so a single query point suffices.

    Parameters
    ----------
    inner : float, optional
        Inner radius :math:`\text{inner} \ge 0`, by default 0.0.
    outer : float, optional
        Outer radius. If ``None`` (the default) it is :math:`+\infty`, so every
        point at distance :math:`\ge` ``inner`` is included. Must be strictly
        greater than ``inner``.
    """

    def __init__(self, inner=0.0, outer=None):
        inner = float(inner)
        if inner < 0:
            raise ValueError("inner must be non-negative.")
        outer = float("inf") if outer is None else float(outer)
        if outer <= inner:
            raise ValueError("outer must be strictly greater than inner.")
        self.inner = inner
        self.outer = outer

    def __call__(self, values):
        v = np.asarray(values)
        return ((v >= self.inner) & (v <= self.outer)).astype(v.dtype)

    def _fused_call(self, backend, reference, query, sample_size, n_instances, replace, gen):
        return backend.sample_subsets_distance_uniform(
            reference, query, self.inner, self.outer, sample_size, n_instances, replace, gen
        )


class Gaussian(_BuiltinDistribution):
    r"""Unnormalized Gaussian of the filter value,

    .. math::
        D(v) = \exp\!\left(-\tfrac{1}{2}\left(\frac{v - \mu}{\sigma}\right)^2\right).

    With the default ``"distance"`` filter this concentrates sampling probability
    on reference points whose distance to the query point is near :math:`\mu`.

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

    def __call__(self, values):
        d = (np.asarray(values) - self.mean) / self.sigma
        return np.exp(-0.5 * d * d)

    def _fused_call(self, backend, reference, query, sample_size, n_instances, replace, gen):
        return backend.sample_subsets_distance_gaussian(
            reference, query, self.mean, self.sigma, sample_size, n_instances, replace, gen
        )
