#    Copyright 2024-2026 Bjorn Wehlin
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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


class Identity(_BuiltinDistribution):
    r"""Use the filter value directly as the sampling weight, :math:`D(v) = v`."""

    def __call__(self, values):
        return values

    def _fused_call(self, backend, reference, query, sample_size, n_instances, replace, gen):
        return backend.sample_subsets_distance_identity(
            reference, query, sample_size, n_instances, replace, gen
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
