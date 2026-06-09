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

from . import _sb_cpp as cpp
from .tensor_create import zeros
from .typing import _validate_dtype, pcf32, pcf64


class Generator:
    """Seedable random number generator for stablebear.

    Parameters
    ----------
    seed : int, optional
        Seed for deterministic generation. If ``None``, a non-deterministic
        seed is used.
    """

    def __init__(self, seed=None):
        if seed is None:
            self._gen = cpp.RandomGenerator()
        else:
            self._gen = cpp.RandomGenerator(seed)

    def seed(self, seed):
        """Re-seed the generator."""
        self._gen.seed(seed)


def seed(s):
    """Seed the global random number generator.

    Parameters
    ----------
    s : int
        Seed value.
    """
    cpp.seed(s)


def _get_backend(dtype):
    dtype = _validate_dtype(dtype, [pcf32, pcf64])

    if dtype == pcf32:
        return cpp.Random_f32_f32
    elif dtype == pcf64:
        return cpp.Random_f64_f64


def noisy_sin(shape, n_points=20, dtype=pcf32, generator=None):
    r"""Generate a tensor of noisy :math:`\sin(2\pi t)` PCFs.

    Each generated PCF has the form

    .. math::
        f(t) = \sin(2\pi t) + \varepsilon(t)

    where :math:`\varepsilon(t) \sim \mathcal{N}(0, 0.1)` is sampled
    independently at each breakpoint. The breakpoints are drawn uniformly
    from :math:`[0, 1]` and sorted, with the first breakpoint fixed at
    :math:`t = 0` and the last value set to :math:`0`.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output tensor.
    n_points : int, optional
        Number of breakpoints per PCF, by default 20.
    dtype : type, optional
        ``pcf32`` or ``pcf64``, by default ``pcf32``.
    generator : Generator, optional
        Random number generator. If ``None``, the global generator is used.

    Returns
    -------
    PcfTensor
        Tensor of noisy sine PCFs with the given shape.
    """
    backend = _get_backend(dtype)

    A = zeros(shape, dtype=dtype)
    gen = generator._gen if generator is not None else None
    backend.noisy_sin(A._data, n_points, gen)

    return A


def noisy_cos(shape, n_points=20, dtype=pcf32, generator=None):
    r"""Generate a tensor of noisy :math:`\cos(2\pi t)` PCFs.

    Each generated PCF has the form

    .. math::
        f(t) = \cos(2\pi t) + \varepsilon(t)

    where :math:`\varepsilon(t) \sim \mathcal{N}(0, 0.1)` is sampled
    independently at each breakpoint. The breakpoints are drawn uniformly
    from :math:`[0, 1]` and sorted, with the first breakpoint fixed at
    :math:`t = 0` and the last value set to :math:`0`.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output tensor.
    n_points : int, optional
        Number of breakpoints per PCF, by default 20.
    dtype : type, optional
        ``pcf32`` or ``pcf64``, by default ``pcf32``.
    generator : Generator, optional
        Random number generator. If ``None``, the global generator is used.

    Returns
    -------
    PcfTensor
        Tensor of noisy cosine PCFs with the given shape.
    """
    backend = _get_backend(dtype)

    A = zeros(shape, dtype=dtype)
    gen = generator._gen if generator is not None else None
    backend.noisy_cos(A._data, n_points, gen)

    return A
