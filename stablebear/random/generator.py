from .. import _sb_cpp as cpp


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
    Anything else raises ``TypeError``.
    """
    if generator is None:
        return None
    if not isinstance(generator, Generator):
        raise TypeError("generator must be a stablebear.random.Generator or None")
    return generator._gen


def seed(s):
    """Seed the global random number generator.

    Parameters
    ----------
    s : int
        Seed value.
    """
    cpp.seed(int(s))
