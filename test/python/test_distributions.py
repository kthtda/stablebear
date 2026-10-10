import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


def test_gaussian_weight_is_the_normalized_density():
    mean, std = 2, 0.5
    gaussian = sb.distributions.Gaussian(mean, std)

    # The mean, one std on either side, and two std above.
    weights = gaussian.weight([2, 2.5, 1.5, 3])

    peak = 1 / (std * np.sqrt(2 * np.pi))
    assert weights == pytest.approx([
        peak * np.exp(-(2 - mean)**2 / (2 * std**2)),
        peak * np.exp(-(2.5 - mean)**2 / (2 * std**2)),
        peak * np.exp(-(1.5 - mean)**2 / (2 * std**2)),
        peak * np.exp(-(3 - mean)**2 / (2 * std**2)),
    ])


def test_bounded_uniform_weight_is_the_density_on_a_half_open_interval():
    start, end = -1, 1
    uniform = sb.distributions.Uniform(start, end)

    # Below the interval, at the start, inside, and at the end.
    weights = uniform.weight([-1.5, -1, 0, 1])

    # The start is included and the end excluded.
    density = 1 / (end - start)
    npt.assert_array_equal(weights, [0, density, density, 0])


def test_mixture_normalizes_the_coefficient_of_a_single_component():
    mixture = sb.distributions.Mixture([sb.distributions.Uniform(0, 1)], [4])

    assert mixture.coefficients == (1.0,)

def test_mixture_normalizes_its_coefficients():
    mixture = sb.distributions.Mixture(
        [sb.distributions.Uniform(0, 1), sb.distributions.Uniform(5, 6)],
        [1, 3],
    )

    assert mixture.coefficients == (0.25, 0.75)


def test_mixture_normalizes_the_coefficients_of_every_component():
    mixture = sb.distributions.Mixture(
        [sb.distributions.Uniform(0, 1), sb.distributions.Uniform(5, 6), sb.distributions.Uniform(10, 11)],
        [1, 2, 5],
    )

    assert mixture.coefficients == (0.125, 0.25, 0.625)

def test_mixture_normalizes_coefficients_passed_by_keyword():
    mixture = sb.distributions.Mixture(
        [sb.distributions.Uniform(0, 1), sb.distributions.Uniform(5, 6)],
        coefficients=[2, 6],
    )

    assert mixture.coefficients == (0.25, 0.75)


def test_mixture_normalizes_coefficients_whose_sum_overflows_double():
    # 1e308 + 1e308 is infinite in double. On x86-64 the long double sum
    # cannot overflow, so there this checks the contract only.
    mixture = sb.distributions.Mixture(
        [sb.distributions.Uniform(0, 1), sb.distributions.Uniform(5, 6)],
        [1e308, 1e308],
    )

    assert mixture.coefficients == (0.5, 0.5)


def test_distributions_with_equal_parameters_are_equal_and_hash_equally():
    # Keyword and positional parameters, the default interval [0, 1), and
    # coefficients that normalize to the same values.
    pairs = [
        (sb.distributions.Gaussian(mean=2, std=0.5), sb.distributions.Gaussian(2, 0.5)),
        (sb.distributions.Uniform(), sb.distributions.Uniform(0, 1)),
        (
            sb.distributions.Mixture([sb.distributions.Uniform(0, 1), sb.distributions.Uniform(5, 6)], [1, 3]),
            sb.distributions.Mixture([sb.distributions.Uniform(0, 1), sb.distributions.Uniform(5, 6)], [2, 6]),
        ),
    ]

    for first, second in pairs:
        assert first == second
        assert hash(first) == hash(second)


def test_distributions_with_different_parameters_are_unequal():
    # Swapped mean and std.
    assert sb.distributions.Gaussian(2, 0.5) != sb.distributions.Gaussian(0.5, 2)
    # Comparison with a non-distribution returns unequal instead of raising.
    assert sb.distributions.Gaussian(0, 1) != 0


def test_mixture_weight_does_not_depend_on_the_other_values():
    mixture = sb.distributions.Mixture(
        [sb.distributions.Uniform(0, 2), sb.distributions.Uniform(10, 11)],
        [1, 3],
    )

    # No value lies in the second component's interval [10, 11).
    weights = mixture.weight([0, 1, 2])

    # Only the first component contributes, with coefficient 1/4 and density 1/2.
    # Normalizing over the values, or giving the second component's coefficient
    # to the first, would give [0.5, 0.5, 0].
    npt.assert_array_equal(weights, [0.25 * 0.5, 0.25 * 0.5, 0])


def test_unbounded_uniform_weight_is_one_on_its_support():
    uniform = sb.distributions.Uniform(1, np.inf)

    # Below the interval, at the start, and far inside.
    weights = uniform.weight([0.5, 1, 1e300])

    # 1/(end - start) would be zero on the whole support.
    npt.assert_array_equal(weights, [0, 1, 1])


@pytest.mark.parametrize(("mixture", "expected"), [
    pytest.param(
        sb.distributions.Mixture(
            [sb.distributions.Gaussian(0, 0.5), sb.distributions.Uniform(0, 2.5)],
            [1, 3],
        ),
        "0.25 * Gaussian(0, 0.5) + 0.75 * Uniform(0, 2.5)",
        id="flat",
    ),
    pytest.param(
        sb.distributions.Mixture(
            [
                sb.distributions.Mixture(
                    [sb.distributions.Uniform(0, 1), sb.distributions.Uniform(1, 2)],
                    [1, 1],
                ),
                sb.distributions.Gaussian(0, 1),
            ],
            [1, 0],
        ),
        "1 * (0.5 * Uniform(0, 1) + 0.5 * Uniform(1, 2)) + 0 * Gaussian(0, 1)",
        id="nested-with-zero-term",
    ),
])
def test_mixture_str_shows_normalized_coefficients(mixture, expected):
    assert str(mixture) == expected


def test_mixture_repr_names_every_parameter():
    mixture = sb.distributions.Mixture(
        [sb.distributions.Gaussian(0, 0.5), sb.distributions.Uniform(0, 2.5)],
        [1, 3],
    )

    assert repr(mixture) == (
        "Mixture(distributions=(Gaussian(mean=0.0, std=0.5), Uniform(start=0.0, end=2.5)), "
        "coefficients=(0.25, 0.75))"
    )


@pytest.mark.parametrize(("attribute", "value"), [
    pytest.param("mean", 2, id="mean"),
    pytest.param("std", 0.5, id="std"),
])
def test_distributions_are_immutable(attribute, value):
    gaussian = sb.distributions.Gaussian(2, 0.5)

    with pytest.raises(AttributeError, match="Gaussian is immutable"):
        setattr(gaussian, attribute, 5)

    assert getattr(gaussian, attribute) == value


@pytest.mark.parametrize("value", [
    pytest.param(0, id="python-int"),
    pytest.param(np.array(0.0), id="0-d-array"),
])
def test_scalar_weight_is_a_python_float(value):
    weight = sb.distributions.Gaussian(0, 1).weight(value)

    # numpy.float64 subclasses float, so isinstance would also accept it.
    assert type(weight) is float
    assert weight == pytest.approx(0.3989422804)


def test_array_weight_keeps_its_shape_as_float64():
    values = np.array([
        [0, 2],
        [3, 1.5],
    ], dtype=np.float32)

    weights = sb.distributions.Uniform(0, 2).weight(values)

    assert weights.dtype == np.float64
    npt.assert_array_equal(weights, [
        [0.5, 0],
        [0, 0.5],
    ])
