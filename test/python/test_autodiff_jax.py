import numpy as np
import pytest

import stablebear as sb

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _tensors():
    F = sb.PcfTensor([
        sb.Pcf(np.array([[0.0, 1.0], [1.0, -2.0], [3.0, 0.5], [4.0, 0.0]])),
        sb.Pcf(np.array([[0.0, 3.0], [0.5, 1.0], [2.5, 0.0]])),
        sb.Pcf(np.array([[0.0, -1.0], [2.0, 2.0], [6.0, 0.0]])),
    ])
    G = sb.PcfTensor([sb.Pcf(np.array([[0.0, 1.0], [2.0, -0.5], [5.0, 0.0]]))])
    return F, G


def test_lp_norm_jax_p_matches_float_p():
    F, _ = _tensors()
    out = sb.lp_norm(F, p=jnp.asarray(2.5))
    assert isinstance(out, jax.Array)
    np.testing.assert_allclose(np.asarray(out), np.asarray(sb.lp_norm(F, p=2.5)))


def test_lp_norm_gradient_wrt_p():
    F, _ = _tensors()
    weights = jnp.asarray([0.3, -1.0, 2.0])
    # check_grads also evaluates with NumPy inputs, which take the untracked path.
    check_grads(lambda p: (weights * sb.lp_norm(F, p=jnp.asarray(p))).sum(), (jnp.asarray(1.7),), order=1, modes=["rev"])


@pytest.mark.parametrize("p", [1.0, 2.5])
@pytest.mark.parametrize("expr", [
    lambda F, G, a: F - a * G,
    lambda F, G, a: (G * a + F) / 2.0,
    lambda F, G, a: -(a * F[1]) + F[2],
])
def test_lp_norm_gradient_through_pcf_arithmetic(p, expr):
    F, G = _tensors()
    weights = jnp.asarray([0.3, -1.0, 2.0])

    def loss(a):
        n = sb.lp_norm(expr(F, G, jnp.asarray(a)), p=p)
        return (weights[: n.shape[0]] * n).sum()

    check_grads(loss, (jnp.asarray(0.3),), order=1, modes=["rev"])


def test_gradient_under_jit():
    F, G = _tensors()

    def loss(a):
        # Tracked through F - a * G, which the static types cannot express.
        return sb.lp_norm(F - a * G, p=2.0).sum()  # pyright: ignore[reportAttributeAccessIssue]

    a = jnp.asarray(0.3)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(a), jax.grad(loss)(a))


def test_lp_norm_gradient_wrt_pcf_values_under_jit():
    F, _ = _tensors()
    f = F[0]
    times = np.array([0.0, 0.5, 2.0, 4.5, 8.0])

    def loss(c):
        g = sb.Pcf(jnp.concatenate([c, jnp.zeros(1)]), times=times)
        # Tracked through g, which the static types cannot express.
        return (sb.lp_norm(f * g, p=1.0) + sb.lp_norm(g, p=2.0)).sum()  # pyright: ignore[reportAttributeAccessIssue]

    c = jnp.asarray([0.7, -1.3, 0.4, 2.0])
    check_grads(lambda c: loss(jnp.asarray(c)), (c,), order=1, modes=["rev"])
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(c), jax.grad(loss)(c))
