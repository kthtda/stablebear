"""Shared argument validation for Python API wrappers."""

import operator
from numbers import Real


def _positive_integer(value, name):
    """Return an index-compatible positive integer, rejecting Python bools."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _boolean(value, name):
    """Return a Python bool without coercing other values."""
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _real(value, name):
    """Return a real scalar as a float, rejecting Python bools."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    return float(value)
