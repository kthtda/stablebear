import numpy as np

import stablebear as sb
from stablebear.point_process import sample_poisson


def test_basic_shape():
    gen = sb.random.Generator(seed=42)
    X = sample_poisson((3, 4), generator=gen)
    assert X.shape == (3, 4)


def test_each_element_has_correct_dimension():
    gen = sb.random.Generator(seed=42)
    X = sample_poisson((5,), dim=3, rate=10.0, generator=gen)
    checked = 0
    for i in range(5):
        pc = np.asarray(X[i])
        if pc.size > 0:
            assert pc.shape[1] == 3
            checked += 1
    assert checked > 0, "All point clouds were empty (rate=10 makes this astronomically unlikely)"


def test_seeded_determinism():
    gen1 = sb.random.Generator(seed=123)
    A = sample_poisson((4, 5), rate=10.0, generator=gen1)

    gen2 = sb.random.Generator(seed=123)
    B = sample_poisson((4, 5), rate=10.0, generator=gen2)

    assert A.array_equal(B)


def test_different_seeds_differ():
    gen1 = sb.random.Generator(seed=42)
    A = sample_poisson((4, 5), rate=10.0, generator=gen1)

    gen2 = sb.random.Generator(seed=99)
    B = sample_poisson((4, 5), rate=10.0, generator=gen2)

    assert not A.array_equal(B)


def test_mean_count_close_to_expected():
    rate = 50.0
    gen = sb.random.Generator(seed=7)
    X = sample_poisson((200,), dim=2, rate=rate, generator=gen)

    counts = [None] * 200
    for i in range(200):
        pc = np.asarray(X[i])
        counts[i] = pc.shape[0]

    mean_count = np.mean(counts)
    # Expected count = rate * volume = 50 * 1 = 50
    assert abs(mean_count - rate) < 5.0, f"Mean count {mean_count} too far from {rate}"


def test_custom_bounds():
    lo = [2.0, 3.0]
    hi = [5.0, 7.0]
    gen = sb.random.Generator(seed=42)
    X = sample_poisson((50,), dim=2, rate=20.0, lo=lo, hi=hi, generator=gen)

    checked = 0
    for i in range(50):
        pc = np.asarray(X[i])
        if pc.size > 0:
            assert np.all(pc[:, 0] >= 2.0)
            assert np.all(pc[:, 0] <= 5.0)
            assert np.all(pc[:, 1] >= 3.0)
            assert np.all(pc[:, 1] <= 7.0)
            checked += 1
    assert checked > 0


def test_pcloud32():
    gen = sb.random.Generator(seed=42)
    X = sample_poisson((3,), rate=10.0, dtype=sb.pcloud32, generator=gen)
    assert X.dtype == sb.pcloud32
    assert X.shape == (3,)


def test_lo_greater_than_hi_raises():
    import pytest
    gen = sb.random.Generator(seed=42)
    # First dimension invalid
    with pytest.raises(ValueError):
        sample_poisson((3,), dim=2, lo=[1.0, 0.0], hi=[0.0, 1.0], generator=gen)
    # Second dimension invalid (first is fine)
    with pytest.raises(ValueError):
        sample_poisson((3,), dim=2, lo=[0.0, 1.0], hi=[1.0, 0.0], generator=gen)


def test_lo_hi_wrong_length_raises():
    import pytest
    gen = sb.random.Generator(seed=42)
    with pytest.raises(ValueError):
        sample_poisson((3,), dim=3, lo=[0.0, 0.0], hi=[1.0, 1.0], generator=gen)
