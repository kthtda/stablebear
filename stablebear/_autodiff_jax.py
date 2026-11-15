"""JAX adapter for stablebear._autodiff.

JAX traces functions with abstract values, so each operation runs as a
``pure_callback`` inside a ``custom_vjp``. That also works under ``jax.jit``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


def to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x)


def concrete(x: Any) -> np.ndarray | None:
    """The values of `x`, or None while JAX traces it abstractly."""
    try:
        return np.asarray(x)
    except jax.errors.TracerArrayConversionError:
        return None


def as_tensor(x: Any, like: Any = None) -> jax.Array:
    if like is None:
        return jnp.asarray(x)
    return jnp.asarray(x, dtype=like.dtype)


def gather(values: Any, idx: np.ndarray) -> jax.Array:
    return values[jnp.asarray(idx)]


def concat(xs: list) -> jax.Array:
    return jnp.concatenate(xs)


def sort(x: Any) -> jax.Array:
    return jnp.sort(x)


def apply(primal: Callable[..., Any], vjp: Callable[..., Any], *params: Any, out: Any = None) -> jax.Array:
    if out is None:
        raise TypeError("This operation does not support JAX: it does not declare the shape of its result.")
    shape, dtype = out
    out_spec = jax.ShapeDtypeStruct(shape, jnp.dtype(dtype))
    params = tuple(jnp.asarray(p) for p in params)
    specs = tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in params)

    def run_primal(*values):
        return np.asarray(primal(*values), dtype=out_spec.dtype)

    def run_vjp(grad_out, *values):
        grads = vjp(grad_out, *values)
        return tuple(np.asarray(g, dtype=s.dtype).reshape(s.shape) for g, s in zip(grads, specs))

    @jax.custom_vjp
    def op(*ps):
        return jax.pure_callback(run_primal, out_spec, *ps)

    def fwd(*ps):
        return op(*ps), ps

    def bwd(ps, grad_out):
        return jax.pure_callback(run_vjp, specs, grad_out, *ps)

    op.defvjp(fwd, bwd)
    return op(*params)
