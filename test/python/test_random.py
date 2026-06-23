import stablebear as sb


def test_noisy_trig():
    X = sb.random.noisy_sin((10, 20))
    assert X.shape == (10, 20)


def test_global_seed_determinism():
    sb.random.seed(42)
    A = sb.random.noisy_sin((5, 10))

    sb.random.seed(42)
    B = sb.random.noisy_sin((5, 10))

    assert A.array_equal(B)


def test_global_seed_different_seeds_differ():
    sb.random.seed(42)
    A = sb.random.noisy_sin((5, 10))

    sb.random.seed(99)
    B = sb.random.noisy_sin((5, 10))

    assert not A.array_equal(B)


def test_generator_determinism():
    gen = sb.random.Generator(seed=123)
    A = sb.random.noisy_sin((5, 10), generator=gen)

    gen2 = sb.random.Generator(seed=123)
    B = sb.random.noisy_sin((5, 10), generator=gen2)

    assert A.array_equal(B)


def test_generator_noisy_cos():
    gen = sb.random.Generator(seed=77)
    A = sb.random.noisy_cos((3, 4), generator=gen)

    gen2 = sb.random.Generator(seed=77)
    B = sb.random.noisy_cos((3, 4), generator=gen2)

    assert A.array_equal(B)


def test_determinism_pcf64():
    gen = sb.random.Generator(seed=42)
    A = sb.random.noisy_sin((3, 4), dtype=sb.pcf64, generator=gen)

    gen2 = sb.random.Generator(seed=42)
    B = sb.random.noisy_sin((3, 4), dtype=sb.pcf64, generator=gen2)

    assert A.array_equal(B)
