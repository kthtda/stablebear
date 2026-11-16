import numpy as np
import pytest

import stablebear as sb


class _Torch:
    traces_abstractly = False

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


_TIMES = [0.75, 1.75, 3.5, 5.25]  # shared with no breakpoint of f or F
_VALUES = np.array([0.7, -1.3, 0.4, 2.0, 0.0])


@pytest.mark.parametrize("p", [1.0, 2.5])
def test_lp_norm_gradient_wrt_pcf_times(fw, p):
    def loss(t):
        g = sb.Pcf(_VALUES, times=fw.concat([fw.tensor([0.0]), t]))
        return fw.sum(sb.lp_norm(g, p=p))

    _assert_gradient_matches_finite_differences(fw, loss, _TIMES)


@pytest.mark.parametrize("p", [1.0, 2.5])
@pytest.mark.parametrize("expr", [
    lambda f, F, g: f * g,
    lambda f, F, g: F - g,
    lambda f, F, g: g * 2.0 + f,
])
def test_lp_norm_gradient_wrt_pcf_times_through_arithmetic(fw, p, expr):
    if fw.traces_abstractly:
        pytest.skip("Arithmetic needs concrete times")
    f, _ = _f_and_g()
    F, _ = _tensors()

    def loss(t):
        g = sb.Pcf(_VALUES, times=fw.concat([fw.tensor([0.0]), t]))
        return fw.sum(sb.lp_norm(expr(f, F, g), p=p))

    _assert_gradient_matches_finite_differences(fw, loss, _TIMES)


def test_arithmetic_with_abstract_times_raises():
    jax = pytest.importorskip("jax")
    f, _ = _f_and_g()

    def loss(t):
        g = sb.Pcf(_VALUES, times=jax.numpy.concatenate([jax.numpy.zeros(1), t]))
        return (sb.lp_norm(f * g, p=1.0)).sum()  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(NotImplementedError, match="order of the breakpoints"):
        jax.grad(loss)(jax.numpy.asarray(_TIMES))


@pytest.mark.parametrize("loss", [
    lambda fw, x: fw.sum(sb.lp_norm(_tensors()[0] - x * _tensors()[1], p=2.0)),
    lambda fw, x: fw.sum(sb.lp_norm(sb.Pcf(fw.concat([x * fw.tensor([1.0, -2.0]), fw.tensor([0.0])]),
                                           times=np.array([0.0, 1.0, 3.0])), p=1.5)),
    lambda fw, x: fw.sum(sb.persistence.compute_homological_kernel(
        fw.tensor(_kernel_points()[0]), x * fw.tensor(_kernel_points()[1]))._tvalues),
], ids=["arithmetic", "values_and_times", "homological_kernel"])
def test_compiled_gradient_matches_eager(fw, loss):
    if fw.compiled_value_and_grad is None:
        pytest.skip("No compiled mode to test")
    eager = fw.value_and_grad(lambda x: loss(fw, x), 0.3)
    compiled = fw.compiled_value_and_grad(lambda x: loss(fw, x), 0.3)
    np.testing.assert_allclose(compiled, eager)


def _kernel_points():
    X = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, -1.0], [4.5, 0.5], [7.0, 3.0], [8.0, -2.5]])
    Y = X * np.array([1.0, 0.25])  # dominated by X: a contraction of each pairwise distance
    return X, Y


def test_homological_kernel_of_framework_points_matches_numpy(fw):
    X, Y = _kernel_points()
    out = sb.persistence.compute_homological_kernel(fw.tensor(X), fw.tensor(Y))
    expected = sb.persistence.compute_homological_kernel(X, Y)[0].to_numpy()
    np.testing.assert_allclose(out[0].to_numpy(), expected)
    np.testing.assert_allclose(fw.to_numpy(out._tvalues), expected)


@pytest.mark.parametrize("wrt", ["X", "Y"])
def test_homological_kernel_gradient(fw, wrt):
    X, Y = _kernel_points()
    weights = fw.tensor([[0.5, 1.0], [-1.0, 2.0], [1.5, -0.5], [2.0, 0.25], [-0.75, 1.25]])

    def loss(P):
        Xp, Yp = (P, Y) if wrt == "X" else (X, P)
        return fw.sum(weights * sb.persistence.compute_homological_kernel(Xp, Yp)._tvalues)

    _assert_gradient_matches_finite_differences(fw, loss, X if wrt == "X" else Y)


def test_kernel_backward_reuses_forward(fw, monkeypatch):
    from stablebear.persistence import homological_kernel

    calls = []
    critical_edges = homological_kernel._critical_edges

    def counting(X, Y):
        calls.append(1)
        return critical_edges(X, Y)

    monkeypatch.setattr(homological_kernel, "_critical_edges", counting)
    X, Y = _kernel_points()
    fw.value_and_grad(lambda P: fw.sum(sb.persistence.compute_homological_kernel(P, Y)._tvalues), X)
    assert len(calls) == 1


def test_stable_rank_of_framework_kernel_matches_numpy(fw):
    X, Y = _kernel_points()
    out = sb.persistence.barcode_to_stable_rank(sb.persistence.compute_homological_kernel(fw.tensor(X), fw.tensor(Y)))
    expected = sb.persistence.barcode_to_stable_rank(sb.persistence.compute_homological_kernel(X, Y))
    t = np.linspace(0.0, 4.0, 81)
    np.testing.assert_allclose(out[0](t), expected[0](t))


@pytest.mark.parametrize("p", [1.0, 2.5])
def test_stable_rank_norm_gradient_wrt_points(fw, p):
    X, Y = _kernel_points()

    def loss(P):
        barcodes = sb.persistence.compute_homological_kernel(P, P * fw.tensor([1.0, 0.25]))
        return fw.sum(sb.lp_norm(sb.persistence.barcode_to_stable_rank(barcodes), p=p))

    _assert_gradient_matches_finite_differences(fw, loss, X)


def test_stable_rank_distance_gradient_wrt_points(fw):
    if fw.traces_abstractly:
        pytest.skip("Arithmetic needs concrete times")
    X, _ = _kernel_points()
    target = sb.persistence.barcode_to_stable_rank(
        sb.persistence.compute_homological_kernel(1.5 * X, 1.5 * X * np.array([1.0, 0.25])))

    def loss(P):
        barcodes = sb.persistence.compute_homological_kernel(P, P * fw.tensor([1.0, 0.25]))
        return fw.sum(sb.lp_norm(sb.persistence.barcode_to_stable_rank(barcodes) - target, p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, X)


def _bar_plf():
    return sb.Plf([[0.0, 0.0], [1.3, 0.5], [2.7, 3.0], [4.1, 3.5]])


def test_transform_bars_of_framework_kernel_matches_numpy(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    X, Y = _kernel_points()
    out = sb.persistence.transform_bars(sb.persistence.compute_homological_kernel(fw.tensor(X), fw.tensor(Y)), _bar_plf())
    expected = sb.persistence.transform_bars(sb.persistence.compute_homological_kernel(X, Y), _bar_plf())[0].to_numpy()
    np.testing.assert_allclose(out[0].to_numpy(), expected)
    np.testing.assert_allclose(fw.to_numpy(out._tvalues), expected)


def test_transform_bars_removes_empty_bars_of_framework_kernel(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    X, Y = _kernel_points()
    f = sb.Plf([[0.0, 0.0], [1.0, 1.0], [2.3, 1.0], [5.0, 2.0]])
    out = sb.persistence.transform_bars(sb.persistence.compute_homological_kernel(fw.tensor(X), fw.tensor(Y)), f)
    expected = sb.persistence.transform_bars(sb.persistence.compute_homological_kernel(X, Y), f)[0].to_numpy()
    assert len(expected) == 3
    np.testing.assert_allclose(fw.to_numpy(out._tvalues), expected)


def test_transform_bars_gradient_with_removed_bars(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    X, Y = _kernel_points()
    f = sb.Plf([[0.0, 0.0], [1.0, 1.0], [2.3, 1.0], [5.0, 2.0]])
    weights = fw.tensor([[0.5, 1.0], [-1.0, 2.0], [1.5, -0.5]])

    def loss(P):
        barcodes = sb.persistence.compute_homological_kernel(X, P)
        return fw.sum(weights * sb.persistence.transform_bars(barcodes, f)._tvalues)

    _assert_gradient_matches_finite_differences(fw, loss, Y)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_transform_bars_of_abstract_bars_raises(fw, compiled):
    value_and_grad = fw.compiled_value_and_grad if compiled else fw.value_and_grad
    if value_and_grad is None or not (compiled or fw.traces_abstractly):
        pytest.skip("Bars are concrete")
    X, Y = _kernel_points()

    def loss(x):
        barcodes = sb.persistence.compute_homological_kernel(fw.tensor(X), x * fw.tensor(Y))
        return fw.sum(sb.persistence.transform_bars(barcodes, _bar_plf())._tvalues)

    with pytest.raises(NotImplementedError, match="abstractly"):
        value_and_grad(loss, 0.3)


@pytest.mark.parametrize("wrt", ["X", "Y"])
def test_transform_bars_gradient(fw, wrt):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    X, Y = _kernel_points()
    weights = fw.tensor([[0.5, 1.0], [-1.0, 2.0], [1.5, -0.5], [2.0, 0.25], [-0.75, 1.25]])

    def loss(P):
        Xp, Yp = (P, Y) if wrt == "X" else (X, P)
        barcodes = sb.persistence.compute_homological_kernel(Xp, Yp)
        return fw.sum(weights * sb.persistence.transform_bars(barcodes, _bar_plf())._tvalues)

    _assert_gradient_matches_finite_differences(fw, loss, X if wrt == "X" else Y)


def test_stable_rank_norm_of_transformed_bars_gradient(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    X, _ = _kernel_points()

    def loss(P):
        barcodes = sb.persistence.compute_homological_kernel(P, P * fw.tensor([1.0, 0.25]))
        return fw.sum(sb.lp_norm(sb.persistence.barcode_to_stable_rank(
            sb.persistence.transform_bars(barcodes, _bar_plf())), p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, X)


def _plf_bars():
    return sb.persistence.BarcodeTensor([sb.persistence.Barcode([[0.2, 1.5], [0.5, 3.7], [2.5, 4.0], [1.2, 2.1]])])


def _plf_bar_weights(fw):
    return fw.tensor([[1.0, 2.0], [-1.0, 0.5], [0.3, 0.7], [0.2, 0.9]])


def test_plf_from_framework_values_and_times_matches_numpy(fw):
    f = sb.Plf(fw.tensor([0.0, 1.0, 1.5, 3.0]), times=fw.tensor([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(f.to_numpy(), [[0.0, 0.0], [1.0, 1.0], [2.0, 1.5], [3.0, 3.0]])


def test_plf_from_framework_breakpoints_raises(fw):
    with pytest.raises(TypeError, match="times="):
        sb.Plf(fw.tensor([[0.0, 0.0], [1.0, 1.0]]))


def test_transform_bars_gradient_wrt_plf_values(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")

    def loss(v):
        f = sb.Plf(v, times=np.array([0.0, 1.0, 2.0, 3.0]))
        return fw.sum(_plf_bar_weights(fw) * sb.persistence.transform_bars(_plf_bars(), f)._tvalues)

    _assert_gradient_matches_finite_differences(fw, loss, [0.0, 1.0, 1.5, 3.0])


def test_transform_bars_gradient_wrt_plf_times(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")

    def loss(t):
        f = sb.Plf(np.array([0.0, 1.0, 1.5, 3.0]), times=fw.concat([fw.tensor([0.0]), t]))
        return fw.sum(_plf_bar_weights(fw) * sb.persistence.transform_bars(_plf_bars(), f)._tvalues)

    _assert_gradient_matches_finite_differences(fw, loss, [1.0, 2.0, 3.0])


def test_transform_bars_gradient_wrt_plf_and_points(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    X, Y = _kernel_points()

    def loss(x):
        barcodes = sb.persistence.compute_homological_kernel(fw.tensor(X), x * fw.tensor(Y))
        f = sb.Plf(x * fw.tensor([0.0, 0.5, 3.0, 3.5]), times=np.array([0.0, 1.3, 2.7, 4.1]))
        return fw.sum(sb.persistence.transform_bars(barcodes, f)._tvalues)

    _assert_gradient_matches_finite_differences(fw, loss, 0.9)


def test_transform_bars_gradient_wrt_plf_at_infinite_death(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    bc = sb.persistence.BarcodeTensor([sb.persistence.Barcode([[0.5, np.inf]])])
    # f ends flat, so f(inf) is its last value.
    _, grad = fw.value_and_grad(
        lambda v: fw.sum(sb.persistence.transform_bars(bc, sb.Plf(v, times=np.array([0.0, 1.0, 2.0])))._tvalues),
        np.array([0.0, 2.0, 2.0]))
    np.testing.assert_allclose(grad, [0.5, 0.5, 1.0])


def test_transform_bars_with_abstract_plf_raises():
    jax = pytest.importorskip("jax")

    def loss(v):
        f = sb.Plf(v, times=np.array([0.0, 1.0]))
        return sb.persistence.transform_bars(_plf_bars(), f)._tvalues.sum()

    with pytest.raises(NotImplementedError, match="abstractly"):
        jax.grad(loss)(jax.numpy.array([0.0, 1.0]))


def _barcode_batch():
    Barcode = sb.persistence.Barcode
    return sb.persistence.BarcodeTensor([
        Barcode([[0.2, 1.5], [0.5, 3.7]]),
        Barcode(np.zeros((0, 2))),
        Barcode([[2.5, 4.0], [1.2, 2.1], [0.1, 0.4]]),
        Barcode([[1.1, 1.4]]),
    ]).reshape((2, 2))


def _batch_plf_times():
    return np.array([0.0, 1.0, 2.0, 3.0])


def test_transform_bars_of_batch_with_framework_plf_matches_numpy(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    values = [0.0, 1.0, 1.0, 3.0]
    out = sb.persistence.transform_bars(_barcode_batch(), sb.Plf(fw.tensor(values), times=_batch_plf_times()))
    expected = sb.persistence.transform_bars(_barcode_batch(), sb.Plf(np.array(values), times=_batch_plf_times()))
    assert out.shape == (2, 2)
    for i in np.ndindex(2, 2):
        np.testing.assert_allclose(out[i].to_numpy(), expected[i].to_numpy())
    np.testing.assert_allclose(fw.to_numpy(out._tvalues), expected._serial()[0])


def test_transform_bars_of_batch_gradient_wrt_plf_values(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    # f is flat on [1, 2], so the bar (1.1, 1.4) is removed.
    weights = fw.tensor([[1.0, 2.0], [-1.0, 0.5], [0.3, 0.7], [0.2, 0.9], [-0.4, 1.1]])

    def loss(v):
        f = sb.Plf(fw.concat([v[:2], v[1:2], v[2:]]), times=_batch_plf_times())
        return fw.sum(weights * sb.persistence.transform_bars(_barcode_batch(), f)._tvalues)

    _assert_gradient_matches_finite_differences(fw, loss, [0.0, 1.0, 3.0])


def test_stable_ranks_of_tracked_batch_match_numpy(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    values = [0.0, 1.0, 1.5, 3.0]
    out = sb.persistence.barcode_to_stable_rank(
        sb.persistence.transform_bars(_barcode_batch(), sb.Plf(fw.tensor(values), times=_batch_plf_times())))
    expected = sb.persistence.barcode_to_stable_rank(
        sb.persistence.transform_bars(_barcode_batch(), sb.Plf(np.array(values), times=_batch_plf_times())))
    assert out.shape == (2, 2)
    t = np.linspace(0.0, 4.0, 81)
    for i in np.ndindex(2, 2):
        np.testing.assert_allclose(out[i](t), expected[i](t))


def test_stable_rank_norms_of_tracked_batch_gradient(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")

    def loss(v):
        f = sb.Plf(v, times=_batch_plf_times())
        return fw.sum(sb.lp_norm(sb.persistence.barcode_to_stable_rank(
            sb.persistence.transform_bars(_barcode_batch(), f)), p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, [0.0, 1.0, 1.5, 3.0])


def test_tracked_plf_with_single_barcode_raises(fw):
    f = sb.Plf(fw.tensor([0.0, 1.0]), times=np.array([0.0, 1.0]))
    with pytest.raises(NotImplementedError, match="BarcodeTensor\\(\\[bc\\]\\)"):
        sb.persistence.transform_bars(_plf_bars()[0], f)


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


def _tracked_batch(fw, values):
    return sb.persistence.transform_bars(_barcode_batch(), sb.Plf(values, times=_batch_plf_times()))


def _expected_batch(values):
    return sb.persistence.transform_bars(_barcode_batch(), sb.Plf(np.array(values), times=_batch_plf_times()))


def test_tracked_barcodes_index_and_reshape_like_plain_ones(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    values = [0.0, 1.0, 1.5, 3.0]
    out, expected = _tracked_batch(fw, fw.tensor(values)), _expected_batch(values)
    for select in (
        lambda t: t[1],
        lambda t: t[:, 0],
        lambda t: t[[1, 0]],
        lambda t: t.reshape((4,)),
        lambda t: t.flatten()[::-1],
        lambda t: t.transpose(),
        lambda t: t.expand_dims(0),
        lambda t: t.broadcast_to((3, 2, 2)),
        lambda t: t.copy(),
    ):
        selected, plain = select(out), select(expected)
        assert selected.shape == plain.shape
        assert selected._tvalues is not None
        np.testing.assert_array_equal(fw.to_numpy(selected._tvalues), plain._serial()[0])
        for i in np.ndindex(*plain.shape):
            np.testing.assert_array_equal(selected[i].to_numpy(), plain[i].to_numpy())


def test_gradient_through_indexing_tracked_barcodes(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")

    def loss(v):
        barcodes = _tracked_batch(fw, v).transpose()[1]
        return fw.sum(sb.lp_norm(sb.persistence.barcode_to_stable_rank(barcodes), p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, [0.0, 1.0, 1.5, 3.0])


def test_tracked_barcodes_cannot_be_changed_or_lose_gradients(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    out = _tracked_batch(fw, fw.tensor([0.0, 1.0, 1.5, 3.0]))
    with pytest.raises(TypeError, match="in place"):
        out[0, 0] = sb.persistence.Barcode([[0.0, 1.0]])
    with pytest.raises(TypeError, match="copy"):
        sb.persistence.BarcodeTensor(out)
    with pytest.raises(NotImplementedError, match="astype"):
        out.astype(sb.barcode32)


def test_arithmetic_on_stable_rank_with_equal_lifetimes_matches_numpy(fw):
    if fw.traces_abstractly:
        pytest.skip("Removing bars needs concrete bars")
    # The bars (0.2, 1.2) and (0.5, 1.5) have the same lifetime, so the
    # tracked stable rank has a repeated time.
    barcodes = sb.persistence.BarcodeTensor([sb.persistence.Barcode([[0.2, 1.2], [0.5, 1.5], [1.0, 3.0]])])
    target = sb.PcfTensor([sb.Pcf(np.array([[0.0, 2.0], [0.7, 1.0], [2.5, 0.0]]))])
    times = np.array([0.0, 4.0])
    tracked = sb.persistence.barcode_to_stable_rank(
        sb.persistence.transform_bars(barcodes, sb.Plf(fw.tensor([0.0, 4.0]), times=times)))
    plain = sb.persistence.barcode_to_stable_rank(
        sb.persistence.transform_bars(barcodes, sb.Plf(np.array([0.0, 4.0]), times=times)))
    t = np.linspace(0.0, 3.0, 61)
    np.testing.assert_allclose((tracked - target)[0](t), (plain - target)[0](t))


def test_barcode_serial_content_round_trip():
    bc = _barcode_batch()
    content, enumeration = bc._serial()
    assert content.shape == (6, 2)
    np.testing.assert_array_equal(enumeration, [[[0, 2], [2, 2]], [[2, 5], [5, 6]]])
    rebuilt = sb.persistence.BarcodeTensor._from_serial(content, enumeration)
    for i in np.ndindex(2, 2):
        np.testing.assert_array_equal(rebuilt[i].to_numpy(), bc[i].to_numpy())


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
