"""Distributions on the real line with weight evaluation.

Gaussian and bounded Uniform weights are normalized densities. Unbounded
Uniform intervals use weight one on their support. Mixture coefficients sum
to one independently of the supplied values. Use ``weight(values)`` with
finite scalar or array-like values to evaluate weights.
"""

import math
from abc import ABC, abstractmethod

import numpy as np

from . import _sb_cpp as cpp
from ._validation import _real

__all__ = ["Distribution", "Gaussian", "Uniform", "Mixture"]


def _evaluate(native, values):
    values = np.asarray(values)
    if values.dtype.kind not in "biuf":
        raise TypeError("values must be real numbers")
    result = native.evaluate(values)
    return result.item() if values.ndim == 0 else result


def _wrap(native):
    """Return the Python distribution wrapping a native distribution."""
    return _WRAPPERS[type(native)]._from_native(native)


class Distribution(ABC):
    """Abstract base class for stablebear's built-in distributions.

    Distributions are immutable. Parameters live in the native object and are
    read back through properties.
    """

    __slots__ = ("_native",)

    def __init__(self, native):
        object.__setattr__(self, "_native", native)

    @classmethod
    def _from_native(cls, native):
        instance = object.__new__(cls)
        Distribution.__init__(instance, native)
        return instance

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    @abstractmethod
    def _fields(self):
        """Return ``(name, value)`` pairs of the constructor parameters."""

    def __repr__(self):
        arguments = ", ".join(f"{name}={value!r}" for name, value in self._fields())
        return f"{type(self).__name__}({arguments})"

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return self._fields() == other._fields()

    def __hash__(self):
        return hash((type(self), self._fields()))

    @abstractmethod
    def weight(self, values):
        """Evaluate weights at finite real values.

        Parameters
        ----------
        values : float or array_like
            Finite real values, of any shape.

        Returns
        -------
        float or numpy.ndarray
            Scalar input returns a float; array-like input returns a float64
            array with the same shape. Values are not normalized over the array.
        """

    @abstractmethod
    def plot_range(self):
        """Return suggested finite plotting bounds as ``(lower, upper)``."""
        raise NotImplementedError


class Gaussian(Distribution):
    """Gaussian distribution with mean ``mean`` and standard deviation ``std``.

    Parameters
    ----------
    mean : float, optional
        Finite center of the weighting function, by default 0.
    std : float, optional
        Finite positive standard deviation, by default 1.

    Notes
    -----
    Weights are the normalized density
    ``exp(-0.5 * ((value - mean)/std)**2) / (std * sqrt(2*pi))``.
    """

    __slots__ = ()

    def __init__(self, mean=0.0, std=1.0):
        super().__init__(cpp._Gaussian(_real(mean, "mean"), _real(std, "std")))

    @property
    def mean(self):
        """Mean of the distribution."""
        return self._native.mean

    @property
    def std(self):
        """Standard deviation of the distribution."""
        return self._native.std

    def _fields(self):
        return ("mean", self.mean), ("std", self.std)

    def weight(self, values):
        """Evaluate the normalized Gaussian density at finite real values.

        See :meth:`Distribution.weight` for parameters and return values.
        """
        return _evaluate(self._native, values)

    def __str__(self):
        """Return a compact label, such as ``Gaussian(0, 0.5)``."""
        return f"Gaussian({self.mean:g}, {self.std:g})"

    def plot_range(self):
        """Return ``(mean - 3*std, mean + 3*std)`` for plotting."""
        return self.mean - 3 * self.std, self.mean + 3 * self.std


class Uniform(Distribution):
    """Uniform distribution on ``[start, end)``, with unbounded intervals allowed.

    Parameters
    ----------
    start : float, optional
        Inclusive lower bound, by default 0. May be negative infinity.
    end : float, optional
        Exclusive upper bound, greater than ``start``, by default 1.
        May be positive infinity.

    Notes
    -----
    A bounded interval has density ``1 / (end - start)`` on its support and
    zero elsewhere. If either endpoint is infinite, the weight is one on the
    interval; this special case does not define a normalized probability density.
    """

    __slots__ = ()

    def __init__(self, start=0.0, end=1.0):
        super().__init__(cpp._Uniform(_real(start, "start"), _real(end, "end")))

    @property
    def start(self):
        """Inclusive lower bound of the interval."""
        return self._native.start

    @property
    def end(self):
        """Exclusive upper bound of the interval."""
        return self._native.end

    def _fields(self):
        return ("start", self.start), ("end", self.end)

    def weight(self, values):
        """Evaluate the interval density, or unit weight if unbounded, at finite real values.

        See :meth:`Distribution.weight` for parameters and return values.
        """
        return _evaluate(self._native, values)

    def __str__(self):
        """Return a compact label, such as ``Uniform(0, 2.5)``."""
        return f"Uniform({self.start:g}, {self.end:g})"

    def plot_range(self):
        """Return ``(start, end)`` widened by 10% of the width on each side.

        An infinite endpoint is first replaced to give a unit-width interval:
        ``Uniform(a, inf)`` uses ``(a, a + 1)``, ``Uniform(-inf, b)`` uses
        ``(b - 1, b)``, and ``Uniform(-inf, inf)`` uses ``(0, 1)``.
        """
        start, end = self.start, self.end
        if math.isinf(start) and math.isinf(end):
            start, end = 0.0, 1.0
        elif math.isinf(end):
            end = start + 1
        elif math.isinf(start):
            start = end - 1
        margin = 0.1 * (end - start)
        if math.isfinite(margin):
            return start - margin, end + margin
        return start, end


class Mixture(Distribution):
    """A mixture of distributions with coefficients normalized to sum to one.

    Parameters
    ----------
    distributions : sequence of Distribution
        Nonempty component sequence, which may include nested mixtures.
    coefficients : sequence of float
        Mixing coefficients, one per component. Must be finite and nonnegative,
        with at least one positive. Normalized to sum to one at construction.

    Notes
    -----
    The ``weight`` method sums component weights using the stored coefficients.
    Evaluation is independent of the other supplied values. A component that is
    zero at an evaluation point contributes zero there; the other coefficients
    remain unchanged. A positive-coefficient unbounded Uniform component,
    including one in a nested mixture, makes the mixture unnormalized as a
    continuous density.
    Functionality that normalizes weights over a set of values, such as
    relative subsampling, normalizes the combined weights once; it does not
    normalize each component over those values.
    Components and coefficients are returned as tuples.
    """

    __slots__ = ()

    def __init__(self, distributions, coefficients):
        distributions = tuple(distributions)
        if not all(isinstance(d, Distribution) for d in distributions):
            raise TypeError("Mixture components must be Distribution instances")
        coefficients = [_real(c, "coefficient") for c in coefficients]
        if len(coefficients) != len(distributions):
            raise ValueError("Mixture needs one coefficient per distribution")
        super().__init__(cpp._Mixture([(c, d._native) for c, d in zip(coefficients, distributions)]))

    @property
    def distributions(self):
        """Component distributions, including zero-coefficient components."""
        return tuple(_wrap(native) for _, native in self._native.terms)

    @property
    def coefficients(self):
        """Mixing coefficients, normalized to sum to one."""
        return tuple(coefficient for coefficient, _ in self._native.terms)

    def _fields(self):
        return ("distributions", self.distributions), ("coefficients", self.coefficients)

    def weight(self, values):
        """Evaluate the coefficient-weighted sum of component weights at finite real values.

        See :meth:`Distribution.weight` for parameters and return values.
        """
        return _evaluate(self._native, values)

    def __str__(self):
        """Return the mixture as a weighted sum of its components.

        For example, ``0.25 * Gaussian(0, 0.5) + 0.75 * Uniform(0, 2.5)``.
        Nested mixtures are parenthesized.
        """
        terms = []
        for distribution, coefficient in zip(self.distributions, self.coefficients):
            label = f"({distribution})" if isinstance(distribution, Mixture) else str(distribution)
            terms.append(f"{coefficient:g} * {label}")
        return " + ".join(terms)

    def plot_range(self):
        """Return the range spanning all positive-coefficient components.

        Uses each component's ``plot_range()``, including nested mixtures.
        """
        ranges = [component.plot_range()
                  for component, coefficient in zip(self.distributions, self.coefficients)
                  if coefficient > 0]
        return min(lower for lower, _ in ranges), max(upper for _, upper in ranges)


_WRAPPERS = {cpp._Gaussian: Gaussian, cpp._Uniform: Uniform, cpp._Mixture: Mixture}
