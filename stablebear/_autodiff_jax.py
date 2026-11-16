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


def version(x: Any) -> None:
    """JAX arrays cannot be changed in place."""
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


def cumsum(x: Any) -> jax.Array:
    return jnp.cumsum(x)


def apply(
    primal: Callable[..., Any], vjp: Callable[..., Any], *params: Any, out: Any = None, residuals: Any = ()
) -> jax.Array:
    """``primal(*values)`` returns (result, residuals), and
    ``vjp(grad_out, residuals, *values)`` the gradients of the values.
    ``residuals`` lists the (shape, dtype) of each residual."""
    if out is None:
        raise TypeError("This operation does not support JAX: it does not declare the shape of its result.")
    shape, dtype = out
    out_spec = jax.ShapeDtypeStruct(shape, jnp.dtype(dtype))
    residual_specs = tuple(
        jax.ShapeDtypeStruct(s, jax.dtypes.canonicalize_dtype(d)) for s, d in residuals)
    params = tuple(jnp.asarray(p) for p in params)
    specs = tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in params)
    n_residuals = len(residual_specs)

    def run_primal(*values):
        result, saved = primal(*values)
        return (np.asarray(result, dtype=out_spec.dtype),
                *(np.asarray(r, dtype=s.dtype) for r, s in zip(saved, residual_specs)))

    def run_vjp(grad_out, *rest):
        grads = vjp(grad_out, tuple(rest[:n_residuals]), *rest[n_residuals:])
        return tuple(np.asarray(g, dtype=s.dtype).reshape(s.shape) for g, s in zip(grads, specs))

    def call(*ps):
        return jax.pure_callback(run_primal, (out_spec, *residual_specs), *ps)

    @jax.custom_vjp
    def op(*ps):
        return call(*ps)[0]

    def fwd(*ps):
        result, *saved = call(*ps)
        return result, (ps, tuple(saved))

    def bwd(carried, grad_out):
        ps, saved = carried
        return jax.pure_callback(run_vjp, specs, grad_out, *saved, *ps)

    op.defvjp(fwd, bwd)
    return op(*params)
