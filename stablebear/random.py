from . import _sb_cpp as cpp
from .tensor_create import zeros
from .typing import _validate_dtype, pcf32, pcf64


class Generator:
    """Seedable random number generator for stablebear.

    The generator advances its internal state on every sampling call, so
    repeated draws from a single generator produce different — but, for a
    given seed, fully reproducible — results. This mirrors the conventions of
    :class:`numpy.random.Generator` and :class:`torch.Generator`::

        g = Generator(seed=5)
        a = noisy_sin((3,), generator=g)
        b = noisy_sin((3,), generator=g)
        # a and b differ; rerunning with a fresh Generator(5) reproduces both.

    State advancing, sub-stream derivation, and auto-seeding all live in the
    C++ engine (``sb::RandomGenerator``); this class is a thin wrapper.

    Parameters
    ----------
    seed : int, optional
        Seed for deterministic generation. If ``None``, a non-deterministic
        seed is drawn by the engine.
    """

    def __init__(self, seed=None):
        if seed is None:
            self._gen = cpp.RandomGenerator()
        else:
            self._gen = cpp.RandomGenerator(int(seed))

    def seed(self, seed):
        """Re-seed the generator and reset its internal stream counter."""
        self._gen.seed(int(seed))


def _unwrap(generator):
    """Unwrap *generator* to the underlying C++ generator to draw from.

    ``None`` stays ``None``, which the backend reads as the process-wide global
    generator (``sb::default_generator()``, reseeded by :func:`seed`). Either
    way the generator advances itself once per sampling call (reserving a fresh
    block of seed slots), so consecutive draws are independent yet reproducible.
    """
    return None if generator is None else generator._gen


def seed(s):
    """Seed the global random number generator.

    Parameters
    ----------
    s : int
        Seed value.
    """
    cpp.seed(int(s))


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
