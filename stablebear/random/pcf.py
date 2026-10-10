from .. import _sb_cpp as cpp
from ..tensor_create import zeros
from ..typing import _validate_dtype, pcf32, pcf64
from .generator import _unwrap


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
    backend.noisy_sin(A._data, n_points, _unwrap(generator))

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
    backend.noisy_cos(A._data, n_points, _unwrap(generator))

    return A
