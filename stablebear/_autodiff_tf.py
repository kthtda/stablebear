"""TensorFlow adapter for stablebear._autodiff.

Each operation runs as a ``tf.numpy_function`` inside a
``tf.custom_gradient``, which also works inside ``tf.function``.
"""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
import tensorflow as tf

from .base_tensor import _PcfTensorBase
from .functional.pcf import Pcf


def _refuse_conversion(value, dtype=None, name=None, as_ref=False):
    raise TypeError(f"A {type(value).__name__} is not converted to a TensorFlow tensor.")


# TensorFlow converts any object with __array__ in binary operators, which
# would turn ``a * pcf`` into a tensor of the PCF's breakpoints. Refusing the
# conversion makes TensorFlow fall back to Pcf.__rmul__.
tf.register_tensor_conversion_function(Pcf, _refuse_conversion)
tf.register_tensor_conversion_function(_PcfTensorBase, _refuse_conversion)


def to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x.numpy())


def concrete(x: Any) -> np.ndarray | None:
    """The values of `x`, or None for a symbolic tensor inside tf.function."""
    try:
        return to_numpy(x)
    except (AttributeError, NotImplementedError):
        return None


def version(x: Any) -> Any:
    """Tensors cannot be changed in place; a variable can, and has no
    counter of its changes, so each call gives a new version."""
    return None if isinstance(x, tf.Tensor) else object()


def as_tensor(x: Any, like: Any = None) -> tf.Tensor:
    if like is None:
        return tf.constant(x)
    return tf.constant(x, dtype=like.dtype)


def gather(values: Any, idx: np.ndarray) -> tf.Tensor:
    return tf.gather(values, idx)


def concat(xs: list) -> tf.Tensor:
    return tf.concat(xs, axis=0)


def sort(x: Any) -> tf.Tensor:
    return tf.sort(x)


def apply(
    primal: Callable[..., Any], vjp: Callable[..., Any], *params: Any, out: Any = None, residuals: Any = ()
) -> tf.Tensor:
    """``primal(*values)`` returns (result, residuals), and
    ``vjp(grad_out, residuals, *values)`` the gradients of the values.
    ``residuals`` lists the (shape, dtype) of each residual."""
    if out is None:
        raise TypeError("This operation does not support TensorFlow: it does not declare the shape of its result.")
    shape, dtype = out
    tensors = [tf.convert_to_tensor(p) for p in params]
    n_residuals = len(residuals)

    def run_primal(*values):
        result, saved = primal(*values)
        return [np.asarray(result, dtype=dtype), *(np.asarray(r, dtype=d) for r, (_, d) in zip(saved, residuals))]

    def run_vjp(grad_out, *rest):
        values = rest[n_residuals:]
        grads = vjp(grad_out, tuple(rest[:n_residuals]), *values)
        return [np.asarray(g, dtype=v.dtype).reshape(v.shape) for g, v in zip(grads, values)]

    @tf.custom_gradient
    def op(*ps):
        outputs = tf.numpy_function(
            run_primal, list(ps), [tf.as_dtype(dtype), *(tf.as_dtype(d) for _, d in residuals)])
        if not isinstance(outputs, (list, tuple)):  # a single output comes back unwrapped
            outputs = [outputs]
        y, *saved = outputs
        y.set_shape(shape)
        for r, (residual_shape, _) in zip(saved, residuals):
            r.set_shape(residual_shape)

        def grad(grad_out):
            # A gather downstream, as in transform_bars, gives IndexedSlices.
            grad_out = tf.convert_to_tensor(grad_out)
            grads = tf.numpy_function(run_vjp, [grad_out, *saved, *ps], [p.dtype for p in tensors])
            if not isinstance(grads, (list, tuple)):  # a single output comes back unwrapped
                grads = [grads]
            for g, p in zip(grads, tensors):
                g.set_shape(p.shape)
            return grads

        return y, grad

    # tf.custom_gradient's annotations describe the undecorated function.
    return cast(tf.Tensor, op(*tensors))
