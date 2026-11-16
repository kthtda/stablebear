"""The autodiff frameworks behind one interface, for tests that run once per
framework, and tensors that carry their gradients.

``tensor`` makes a float64 framework array, and ``from_numpy`` one of the
dtype of a NumPy array.
"""

import numpy as np

from stablebear._tracked import tracked_class


class _Torch:
    traces_abstractly = False

    def __init__(self):
        import torch

        self.torch = torch
        self.tensor_type = torch.Tensor

    def tensor(self, x):
        return self.torch.tensor(x, dtype=self.torch.float64)

    def from_numpy(self, x):
        return self.torch.from_numpy(x)

    def to_numpy(self, x):
        return x.detach().numpy()

    def sum(self, x):
        return x.sum()

    def concat(self, xs):
        return self.torch.cat(xs)

    def value_and_grad(self, f, x0):
        x = self.tensor(x0).requires_grad_()
        y = f(x)
        y.backward()
        assert x.grad is not None
        return y.item(), x.grad.numpy()

    compiled_value_and_grad = None


class _Jax:
    traces_abstractly = True

    def __init__(self):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = jnp
        self.tensor_type = jax.Array

    def tensor(self, x):
        return self.jnp.asarray(x, dtype=self.jnp.float64)

    def from_numpy(self, x):
        return self.jnp.asarray(x)

    def to_numpy(self, x):
        return np.asarray(x)

    def sum(self, x):
        return x.sum()

    def concat(self, xs):
        return self.jnp.concatenate(xs)

    def value_and_grad(self, f, x0):
        y, g = self.jax.value_and_grad(f)(self.tensor(x0))
        return float(y), np.asarray(g)

    def compiled_value_and_grad(self, f, x0):
        y, g = self.jax.jit(self.jax.value_and_grad(f))(self.tensor(x0))
        return float(y), np.asarray(g)


class _TensorFlow:
    traces_abstractly = False

    def __init__(self):
        import tensorflow as tf

        # Registers the refusal to convert PCFs, which stablebear does on
        # import only when TensorFlow was imported first.
        import stablebear._autodiff_tf  # noqa: F401

        self.tf = tf
        self.tensor_type = tf.Tensor

    def tensor(self, x):
        return self.tf.constant(x, dtype=self.tf.float64)

    def from_numpy(self, x):
        return self.tf.constant(x)

    def to_numpy(self, x):
        return x.numpy()

    def sum(self, x):
        return self.tf.reduce_sum(x)

    def concat(self, xs):
        return self.tf.concat(xs, axis=0)

    def _value_and_grad(self, f, x):
        with self.tf.GradientTape() as tape:
            y = f(x)
        return y, tape.gradient(y, x)

    def value_and_grad(self, f, x0):
        y, g = self._value_and_grad(f, self.tf.Variable(x0, dtype=self.tf.float64))
        return float(y.numpy()), self.tf.convert_to_tensor(g).numpy()

    def compiled_value_and_grad(self, f, x0):
        x = self.tf.Variable(x0, dtype=self.tf.float64)
        y, g = self.tf.function(lambda: self._value_and_grad(f, x))()
        return float(y.numpy()), self.tf.convert_to_tensor(g).numpy()


FRAMEWORKS = {"torch": _Torch, "jax": _Jax, "tensorflow": _TensorFlow}


def tracked(fw, tensor):
    """The tensor as one that carries gradients of the framework: its serial
    content becomes a framework array."""
    content, enumeration = tensor._serial()
    return tracked_class(type(tensor))([fw.from_numpy(content)], enumeration)
