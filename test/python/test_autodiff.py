import numpy as np
import pytest

import stablebear as sb


class _Torch:
    def __init__(self):
        import torch

        self.torch = torch
        self.tensor_type = torch.Tensor

    def tensor(self, x):
        return self.torch.tensor(x, dtype=self.torch.float64)

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
    def __init__(self):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = jnp
        self.tensor_type = jax.Array

    def tensor(self, x):
        return self.jnp.asarray(x, dtype=self.jnp.float64)

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
    def __init__(self):
        import tensorflow as tf

        # Registers the refusal to convert PCFs, which stablebear does on
        # import only when TensorFlow was imported first.
        import stablebear._autodiff_tf  # noqa: F401

        self.tf = tf
        self.tensor_type = tf.Tensor

    def tensor(self, x):
        return self.tf.constant(x, dtype=self.tf.float64)

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
        return float(y.numpy()), g.numpy()

    def compiled_value_and_grad(self, f, x0):
        x = self.tf.Variable(x0, dtype=self.tf.float64)
        y, g = self.tf.function(lambda: self._value_and_grad(f, x))()
        return float(y.numpy()), g.numpy()


_FRAMEWORKS = {"torch": _Torch, "jax": _Jax, "tensorflow": _TensorFlow}


@pytest.fixture(params=list(_FRAMEWORKS))
def fw(request):
    """Parametrize a test over the autodiff frameworks that are installed."""
    pytest.importorskip(request.param)
    return _FRAMEWORKS[request.param]()


def _assert_gradient_matches_finite_differences(fw, f, x0, eps=1e-6):
    x0 = np.asarray(x0, dtype=np.float64)
    _, grad = fw.value_and_grad(f, x0)
    numerical = np.zeros_like(x0)
    for i in np.ndindex(x0.shape):
        step = np.zeros_like(x0)
        step[i] = eps
        numerical[i] = (fw.to_numpy(f(fw.tensor(x0 + step))) - fw.to_numpy(f(fw.tensor(x0 - step)))) / (2 * eps)
    np.testing.assert_allclose(grad, numerical, rtol=1e-6, atol=1e-8)


def _f_and_g():
    f = sb.Pcf(np.array([[0.0, 1.0], [1.0, -2.0], [3.0, 0.5], [4.0, 3.0], [6.0, -1.0], [7.0, 0.0]]))
    g = sb.Pcf(np.array([[0.0, 1.0], [2.0, -0.5], [5.0, 2.0], [7.0, 0.0]]))
    return f, g


def _tensors():
    F = sb.PcfTensor([
        sb.Pcf(np.array([[0.0, 1.0], [1.0, -2.0], [3.0, 0.5], [4.0, 0.0]])),
        sb.Pcf(np.array([[0.0, 3.0], [0.5, 1.0], [2.5, 0.0]])),
        sb.Pcf(np.array([[0.0, -1.0], [2.0, 2.0], [6.0, 0.0]])),
    ])
    G = sb.PcfTensor([sb.Pcf(np.array([[0.0, 1.0], [2.0, -0.5], [5.0, 0.0]]))])
    return F, G


def test_lp_norm_with_framework_p_matches_float_p(fw):
    F, _ = _tensors()
    out = sb.lp_norm(F, p=fw.tensor(2.5))
    assert isinstance(out, fw.tensor_type)
    np.testing.assert_allclose(fw.to_numpy(out), np.asarray(sb.lp_norm(F, p=2.5)))


@pytest.mark.parametrize("p0", [1.2, 2.0, 3.7])
def test_lp_norm_gradient_wrt_p(fw, p0):
    F, _ = _tensors()
    weights = fw.tensor([0.3, -1.0, 2.0])
    _assert_gradient_matches_finite_differences(fw, lambda p: fw.sum(weights * sb.lp_norm(F, p=p)), p0)


@pytest.mark.parametrize("p", [1.0, 2.5])
@pytest.mark.parametrize("expr", [
    lambda f, g, a: f + a * g,
    lambda f, g, a: g * a - f,
    lambda f, g, a: 2.0 * f - (a * g) / 2.0,
])
def test_lp_norm_gradient_through_pcf_arithmetic(fw, p, expr):
    f, g = _f_and_g()
    _assert_gradient_matches_finite_differences(fw, lambda a: fw.sum(sb.lp_norm(expr(f, g, a), p=p)), 0.3)


def test_pcf_arithmetic_with_framework_scalar_matches_float_scalar(fw):
    f, g = _f_and_g()
    h = f + fw.tensor(0.3) * g
    expected = (f + 0.3 * g).to_numpy()
    np.testing.assert_allclose(h(expected[:, 0]), expected[:, 1])
    np.testing.assert_allclose(fw.to_numpy(h._tvalues), h.to_numpy()[:, 1])


@pytest.mark.parametrize("p", [1.0, 2.5])
@pytest.mark.parametrize("expr", [
    lambda F, G, a: F - a * G,
    lambda F, G, a: (G * a + F) / 2.0,
    lambda F, G, a: -(a * F) + G,
])
def test_lp_norm_gradient_through_pcf_tensor_arithmetic(fw, p, expr):
    F, G = _tensors()
    weights = fw.tensor([0.3, -1.0, 2.0])
    _assert_gradient_matches_finite_differences(
        fw, lambda a: fw.sum(weights * sb.lp_norm(expr(F, G, a), p=p)), 0.3)


@pytest.mark.parametrize("p", [1.0, 2.5])
def test_lp_norm_gradient_wrt_pcf_values(fw, p):
    f, _ = _f_and_g()
    times = np.array([0.0, 0.5, 2.0, 4.5, 8.0])

    def loss(c):
        g = sb.Pcf(fw.concat([c, fw.tensor([0.0])]), times=times)
        return fw.sum(sb.lp_norm(f * g, p=p)) + fw.sum(sb.lp_norm(g, p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, [0.7, -1.3, 0.4, 2.0])


@pytest.mark.parametrize("loss", [
    lambda fw, x: fw.sum(sb.lp_norm(_tensors()[0] - x * _tensors()[1], p=2.0)),
    lambda fw, x: fw.sum(sb.lp_norm(sb.Pcf(fw.concat([x * fw.tensor([1.0, -2.0]), fw.tensor([0.0])]),
                                           times=np.array([0.0, 1.0, 3.0])), p=1.5)),
], ids=["arithmetic", "values_and_times"])
def test_compiled_gradient_matches_eager(fw, loss):
    if fw.compiled_value_and_grad is None:
        pytest.skip("No compiled mode to test")
    eager = fw.value_and_grad(lambda x: loss(fw, x), 0.3)
    compiled = fw.compiled_value_and_grad(lambda x: loss(fw, x), 0.3)
    np.testing.assert_allclose(compiled, eager)


def test_breakpoints_as_framework_array_raise(fw):
    with pytest.raises(TypeError, match="times="):
        sb.Pcf(fw.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_framework_values_and_times_of_different_lengths_raise(fw):
    with pytest.raises(ValueError, match="same length"):
        sb.Pcf(fw.tensor([0.0, 0.0]), times=np.array([0.0, 1.0, 2.0]))


def test_gradient_wrt_keras_variable():
    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    if keras.backend.backend() != "tensorflow":
        pytest.skip("Keras runs on another backend")
    import stablebear._autodiff_tf  # noqa: F401

    F, G = _tensors()
    grads = []
    for param in (keras.Variable(0.3, dtype="float64"), tf.Variable(0.3, dtype=tf.float64)):
        with tf.GradientTape() as tape:
            # A Keras variable on the left would convert G itself and fail.
            loss = tf.reduce_sum(sb.lp_norm(F - G * param, p=2.0))
        grads.append(tape.gradient(loss, param).numpy())
    np.testing.assert_allclose(grads[0], grads[1])


def test_serial_content_round_trip():
    F, _ = _tensors()
    content, enumeration = F._serial()
    assert content.shape == (10, 2)
    np.testing.assert_array_equal(enumeration, [[0, 4], [4, 7], [7, 10]])
    G = sb.from_serial_content(content, enumeration)
    for i in range(3):
        np.testing.assert_array_equal(G[i].to_numpy(), F[i].to_numpy())


def test_values_with_times_match_breakpoints():
    g = sb.Pcf(np.array([2.0, -1.0, 0.0]), times=np.array([0.0, 1.0, 3.0]))
    np.testing.assert_array_equal(g.to_numpy(), [[0.0, 2.0], [1.0, -1.0], [3.0, 0.0]])


def test_values_and_times_from_float_tensors():
    times = sb.FloatTensor(np.array([0.0, 1.0, 3.0], dtype=np.float32))
    values = sb.FloatTensor(np.array([2.0, -1.0, 0.0], dtype=np.float32))
    g = sb.Pcf(values, times=times)
    assert g.vtype == sb.float32
    np.testing.assert_array_equal(g.to_numpy(), [[0.0, 2.0], [1.0, -1.0], [3.0, 0.0]])


@pytest.mark.parametrize("values", [np.zeros((3, 2)), np.zeros(2)], ids=["2d", "shorter"])
def test_values_and_times_of_different_shapes_raise(values):
    with pytest.raises(ValueError, match="same length"):
        sb.Pcf(values, times=np.array([0.0, 1.0, 2.0]))
