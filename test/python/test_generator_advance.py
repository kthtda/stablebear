"""Regression tests for issue #86: a Generator used to NOT advance its state
between calls, so two consecutive draws from one generator produced identical
tensors (a silent footgun for any repeated-sampling / Monte-Carlo loop)."""

import stablebear as sb
from stablebear.point_process import sample_poisson


def test_consecutive_draws_differ():
    """The core fix: two draws from the same generator must differ."""
    g = sb.random.Generator(seed=5)
    a = sb.random.noisy_sin((3,), n_points=4, generator=g)
    b = sb.random.noisy_sin((3,), n_points=4, generator=g)
    assert not a.array_equal(b)


def test_sequence_is_reproducible():
    """Advancing must still be deterministic: the same seed replays the same
    sequence of draws."""
    g1 = sb.random.Generator(seed=5)
    a1 = sb.random.noisy_sin((3,), n_points=4, generator=g1)
    b1 = sb.random.noisy_sin((3,), n_points=4, generator=g1)

    g2 = sb.random.Generator(seed=5)
    a2 = sb.random.noisy_sin((3,), n_points=4, generator=g2)
    b2 = sb.random.noisy_sin((3,), n_points=4, generator=g2)

    assert a1.array_equal(a2)
    assert b1.array_equal(b2)


def test_reseed_resets_sequence():
    g = sb.random.Generator(seed=5)
    a = sb.random.noisy_sin((3,), n_points=4, generator=g)
    _ = sb.random.noisy_sin((3,), n_points=4, generator=g)
    g.seed(5)
    a_again = sb.random.noisy_sin((3,), n_points=4, generator=g)
    assert a.array_equal(a_again)


def test_global_generator_advances():
    sb.random.seed(7)
    a = sb.random.noisy_sin((3,), n_points=4)
    b = sb.random.noisy_sin((3,), n_points=4)
    assert not a.array_equal(b)


def test_global_seed_replays_sequence():
    sb.random.seed(7)
    a1 = sb.random.noisy_sin((3,), n_points=4)
    b1 = sb.random.noisy_sin((3,), n_points=4)
    sb.random.seed(7)
    a2 = sb.random.noisy_sin((3,), n_points=4)
    b2 = sb.random.noisy_sin((3,), n_points=4)
    assert a1.array_equal(a2)
    assert b1.array_equal(b2)


def test_poisson_generator_advances():
    g = sb.random.Generator(seed=3)
    p1 = sample_poisson((5,), dim=2, rate=4.0, generator=g)
    p2 = sample_poisson((5,), dim=2, rate=4.0, generator=g)
    # Reproducible across a fresh generator with the same seed...
    g2 = sb.random.Generator(seed=3)
    p1b = sample_poisson((5,), dim=2, rate=4.0, generator=g2)
    assert p1.array_equal(p1b)
    # ...but the two consecutive draws are not identical.
    assert not p1.array_equal(p2)
