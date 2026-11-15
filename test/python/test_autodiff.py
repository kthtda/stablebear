import numpy as np
import pytest

import stablebear as sb

torch = pytest.importorskip("torch")


def _pcfs():
    return sb.PcfTensor([
        sb.Pcf(np.array([[0.0, 4.0], [0.25, 0.0]])),
        sb.Pcf(np.array([[0.0, 1.0], [4.0, 0.0]])),
        sb.Pcf(np.array([[0.0, 2.0], [1.0, 0.5], [3.0, 3.0], [3.5, 0.0]])),
    ])


def test_lp_norm_torch_p_matches_float_p():
    p = torch.tensor(2.5, dtype=torch.float64)
    out = sb.lp_norm(_pcfs(), p=p)
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out.numpy(), np.asarray(sb.lp_norm(_pcfs(), p=2.5)))


@pytest.mark.parametrize("p0", [1.2, 2.0, 3.7])
def test_lp_norm_gradient_wrt_p(p0):
    fs = _pcfs()
    weights = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)
    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: (weights * sb.lp_norm(fs, p=p)).sum(), p)


def _f_and_g():
    f = sb.Pcf(np.array([[0.0, 1.0], [1.0, -2.0], [3.0, 0.5], [4.0, 3.0], [6.0, -1.0], [7.0, 0.0]]))
    g = sb.Pcf(np.array([[0.0, 1.0], [2.0, -0.5], [5.0, 2.0], [7.0, 0.0]]))
    return f, g


@pytest.mark.parametrize("p", [1.0, 2.5])
@pytest.mark.parametrize("expr", [
    lambda f, g, a: f + a * g,
    lambda f, g, a: g * a - f,
    lambda f, g, a: 2.0 * f - (a * g) / 2.0,
])
def test_lp_norm_gradient_through_pcf_arithmetic(p, expr):
    f, g = _f_and_g()
    a = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a: sb.lp_norm(expr(f, g, a), p=p).sum(), a)


def test_pcf_arithmetic_with_tensor_scalar_matches_float_scalar():
    f, g = _f_and_g()
    h = f + torch.tensor(0.3, dtype=torch.float64) * g
    np.testing.assert_allclose(h.to_numpy(), (f + 0.3 * g).to_numpy())
    np.testing.assert_allclose(h._tvalues.numpy(), h.to_numpy()[:, 1])


def _tensors():
    F = sb.PcfTensor([
        sb.Pcf(np.array([[0.0, 1.0], [1.0, -2.0], [3.0, 0.5], [4.0, 0.0]])),
        sb.Pcf(np.array([[0.0, 3.0], [0.5, 1.0], [2.5, 0.0]])),
        sb.Pcf(np.array([[0.0, -1.0], [2.0, 2.0], [6.0, 0.0]])),
    ])
    G = sb.PcfTensor([sb.Pcf(np.array([[0.0, 1.0], [2.0, -0.5], [5.0, 0.0]]))])
    return F, G


def test_serial_content_round_trip():
    F, _ = _tensors()
    content, enumeration = F._serial()
    assert content.shape == (10, 2)
    np.testing.assert_array_equal(enumeration, [[0, 4], [4, 7], [7, 10]])
    G = sb.from_serial_content(content, enumeration)
    for i in range(3):
        np.testing.assert_array_equal(G[i].to_numpy(), F[i].to_numpy())


@pytest.mark.parametrize("p", [1.0, 2.5])
@pytest.mark.parametrize("expr", [
    lambda F, G, a: F - a * G,
    lambda F, G, a: (G * a + F) / 2.0,
    lambda F, G, a: -(a * F) + G,
])
def test_lp_norm_gradient_through_pcf_tensor_arithmetic(p, expr):
    F, G = _tensors()
    weights = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)
    a = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a: (weights * sb.lp_norm(expr(F, G, a), p=p)).sum(), a)
