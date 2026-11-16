import numpy as np
import pytest

import stablebear as sb
from stablebear import _autodiff

from _frameworks import assert_gradient_matches_finite_differences as _assert_gradient_matches_finite_differences
from _frameworks import tracked
from _round_trip import ROUND_TRIPS


# Runs a test for a Pcf and for a Plf, given as its ``function`` argument.
_FUNCTIONS = pytest.mark.parametrize("function", [sb.Pcf, sb.Plf], ids=["pcf", "plf"])


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
    np.testing.assert_allclose(h.to_numpy(), expected)
    breakpoints, _ = sb.PcfTensor([h]).to_serial_content()
    np.testing.assert_allclose(fw.to_numpy(breakpoints), expected)


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
    f, _ = _f_and_g()
    F, _ = _tensors()

    def loss(t):
        g = sb.Pcf(_VALUES, times=fw.concat([fw.tensor([0.0]), t]))
        return fw.sum(sb.lp_norm(expr(f, F, g), p=p))

    _assert_gradient_matches_finite_differences(fw, loss, _TIMES)


def _kernel_points():
    X = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, -1.0], [4.5, 0.5], [7.0, 3.0], [8.0, -2.5]])
    Y = X * np.array([1.0, 0.25])  # dominated by X: a contraction of each pairwise distance
    return X, Y


def test_homological_kernel_of_framework_points_matches_numpy(fw):
    X, Y = _kernel_points()
    out = sb.persistence.compute_homological_kernel(fw.tensor(X), fw.tensor(Y))
    expected = sb.persistence.compute_homological_kernel(X, Y)[0].to_numpy()
    np.testing.assert_allclose(out[0].to_numpy(), expected)
    np.testing.assert_allclose(fw.to_numpy(out.to_serial_content()[0]), expected)


@pytest.mark.parametrize("wrt", ["X", "Y"])
def test_homological_kernel_gradient(fw, wrt):
    X, Y = _kernel_points()
    weights = fw.tensor([[0.5, 1.0], [-1.0, 2.0], [1.5, -0.5], [2.0, 0.25], [-0.75, 1.25]])

    def loss(P):
        Xp, Yp = (P, Y) if wrt == "X" else (X, P)
        return fw.sum(weights * sb.persistence.compute_homological_kernel(Xp, Yp).to_serial_content()[0])

    _assert_gradient_matches_finite_differences(fw, loss, X if wrt == "X" else Y)


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
    X, Y = _kernel_points()
    out = sb.persistence.transform_bars(sb.persistence.compute_homological_kernel(fw.tensor(X), fw.tensor(Y)), _bar_plf())
    expected = sb.persistence.transform_bars(sb.persistence.compute_homological_kernel(X, Y), _bar_plf())[0].to_numpy()
    np.testing.assert_allclose(out[0].to_numpy(), expected)
    np.testing.assert_allclose(fw.to_numpy(out.to_serial_content()[0]), expected)


def test_transform_bars_removes_empty_bars_of_framework_kernel(fw):
    X, Y = _kernel_points()
    f = sb.Plf([[0.0, 0.0], [1.0, 1.0], [2.3, 1.0], [5.0, 2.0]])
    out = sb.persistence.transform_bars(sb.persistence.compute_homological_kernel(fw.tensor(X), fw.tensor(Y)), f)
    expected = sb.persistence.transform_bars(sb.persistence.compute_homological_kernel(X, Y), f)[0].to_numpy()
    assert len(expected) == 3
    np.testing.assert_allclose(fw.to_numpy(out.to_serial_content()[0]), expected)


def test_transform_bars_removing_every_bar_of_framework_kernel(fw):
    X, Y = _kernel_points()
    barcodes = sb.persistence.compute_homological_kernel(fw.tensor(X), fw.tensor(Y))
    # A constant PLF maps every bar to an empty one.
    out = sb.persistence.transform_bars(barcodes, sb.Plf([[0.0, 1.0]]))
    assert out.shape == (1,)
    assert out[0].to_numpy().shape == (0, 2)


def test_transform_bars_gradient_with_removed_bars(fw):
    X, Y = _kernel_points()
    f = sb.Plf([[0.0, 0.0], [1.0, 1.0], [2.3, 1.0], [5.0, 2.0]])
    weights = fw.tensor([[0.5, 1.0], [-1.0, 2.0], [1.5, -0.5]])

    def loss(P):
        barcodes = sb.persistence.compute_homological_kernel(X, P)
        return fw.sum(weights * sb.persistence.transform_bars(barcodes, f).to_serial_content()[0])

    _assert_gradient_matches_finite_differences(fw, loss, Y)


@pytest.mark.parametrize("wrt", ["X", "Y"])
def test_transform_bars_gradient(fw, wrt):
    X, Y = _kernel_points()
    weights = fw.tensor([[0.5, 1.0], [-1.0, 2.0], [1.5, -0.5], [2.0, 0.25], [-0.75, 1.25]])

    def loss(P):
        Xp, Yp = (P, Y) if wrt == "X" else (X, P)
        barcodes = sb.persistence.compute_homological_kernel(Xp, Yp)
        return fw.sum(weights * sb.persistence.transform_bars(barcodes, _bar_plf()).to_serial_content()[0])

    _assert_gradient_matches_finite_differences(fw, loss, X if wrt == "X" else Y)


def test_stable_rank_norm_of_transformed_bars_gradient(fw):
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
    def loss(v):
        f = sb.Plf(v, times=np.array([0.0, 1.0, 2.0, 3.0]))
        return fw.sum(_plf_bar_weights(fw) * sb.persistence.transform_bars(_plf_bars(), f).to_serial_content()[0])

    _assert_gradient_matches_finite_differences(fw, loss, [0.0, 1.0, 1.5, 3.0])


def test_transform_bars_gradient_wrt_plf_times(fw):
    def loss(t):
        f = sb.Plf(np.array([0.0, 1.0, 1.5, 3.0]), times=fw.concat([fw.tensor([0.0]), t]))
        return fw.sum(_plf_bar_weights(fw) * sb.persistence.transform_bars(_plf_bars(), f).to_serial_content()[0])

    _assert_gradient_matches_finite_differences(fw, loss, [1.0, 2.0, 3.0])


def test_transform_bars_gradient_wrt_plf_and_points(fw):
    X, Y = _kernel_points()

    def loss(x):
        barcodes = sb.persistence.compute_homological_kernel(fw.tensor(X), x * fw.tensor(Y))
        f = sb.Plf(x * fw.tensor([0.0, 0.5, 3.0, 3.5]), times=np.array([0.0, 1.3, 2.7, 4.1]))
        return fw.sum(sb.persistence.transform_bars(barcodes, f).to_serial_content()[0])

    _assert_gradient_matches_finite_differences(fw, loss, 0.9)


def test_transform_bars_gradient_wrt_plf_at_infinite_death(fw):
    bc = sb.persistence.BarcodeTensor([sb.persistence.Barcode([[0.5, np.inf]])])
    # f ends flat, so f(inf) is its last value.
    _, grad = fw.value_and_grad(
        lambda v: fw.sum(sb.persistence.transform_bars(bc, sb.Plf(v, times=np.array([0.0, 1.0, 2.0]))).to_serial_content()[0]),
        np.array([0.0, 2.0, 2.0]))
    np.testing.assert_allclose(grad, [0.5, 0.5, 1.0])


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
    values = [0.0, 1.0, 1.0, 3.0]
    out = sb.persistence.transform_bars(_barcode_batch(), sb.Plf(fw.tensor(values), times=_batch_plf_times()))
    expected = sb.persistence.transform_bars(_barcode_batch(), sb.Plf(np.array(values), times=_batch_plf_times()))
    assert out.shape == (2, 2)
    for i in np.ndindex(2, 2):
        np.testing.assert_allclose(out[i].to_numpy(), expected[i].to_numpy())
    np.testing.assert_allclose(fw.to_numpy(out.to_serial_content()[0]), expected._serial()[0])


def test_transform_bars_of_batch_gradient_wrt_plf_values(fw):
    # f is flat on [1, 2], so the bar (1.1, 1.4) is removed.
    weights = fw.tensor([[1.0, 2.0], [-1.0, 0.5], [0.3, 0.7], [0.2, 0.9], [-0.4, 1.1]])

    def loss(v):
        f = sb.Plf(fw.concat([v[:2], v[1:2], v[2:]]), times=_batch_plf_times())
        return fw.sum(weights * sb.persistence.transform_bars(_barcode_batch(), f).to_serial_content()[0])

    _assert_gradient_matches_finite_differences(fw, loss, [0.0, 1.0, 3.0])


def test_stable_ranks_of_tracked_batch_match_numpy(fw):
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
    def loss(v):
        f = sb.Plf(v, times=_batch_plf_times())
        return fw.sum(sb.lp_norm(sb.persistence.barcode_to_stable_rank(
            sb.persistence.transform_bars(_barcode_batch(), f)), p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, [0.0, 1.0, 1.5, 3.0])


def test_transform_bars_of_single_barcode_with_tracked_plf_matches_numpy(fw):
    barcode = _plf_bars()[0]
    values, times = [0.0, 1.0], np.array([0.0, 1.0])

    out = sb.persistence.transform_bars(barcode, sb.Plf(fw.tensor(values), times=times))

    assert isinstance(out, sb.persistence.Barcode)
    assert isinstance(out.to_serial_content()[0], fw.tensor_type)
    expected = sb.persistence.transform_bars(barcode, sb.Plf(np.array(values), times=times))
    assert out.is_isomorphic_to(expected, atol=1e-12, rtol=1e-12)


def test_transform_bars_of_time_series_with_tracked_plf_raises(fw):
    series = sb.timeseries.TimeSeries(_plf_bars(), times=[0.0])
    f = sb.Plf(fw.tensor([0.0, 1.0]), times=np.array([0.0, 1.0]))
    with pytest.raises(NotImplementedError, match="Barcode"):
        sb.persistence.transform_bars(series, f)


def test_breakpoints_as_framework_array_raise(fw):
    with pytest.raises(TypeError, match="times="):
        sb.Pcf(fw.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_framework_values_and_times_of_different_lengths_raise(fw):
    with pytest.raises(ValueError, match="same length"):
        sb.Pcf(fw.tensor([0.0, 0.0]), times=np.array([0.0, 1.0, 2.0]))


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
        assert isinstance(selected.to_serial_content()[0], fw.tensor_type)
        np.testing.assert_array_equal(fw.to_numpy(selected.to_serial_content()[0]), plain._serial()[0])
        for i in np.ndindex(*plain.shape):
            np.testing.assert_array_equal(selected[i].to_numpy(), plain[i].to_numpy())


def test_gradient_through_indexing_tracked_barcodes(fw):
    def loss(v):
        barcodes = _tracked_batch(fw, v).transpose()[1]
        return fw.sum(sb.lp_norm(sb.persistence.barcode_to_stable_rank(barcodes), p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, [0.0, 1.0, 1.5, 3.0])


def test_tracked_barcodes_cannot_lose_gradients(fw):
    out = _tracked_batch(fw, fw.tensor([0.0, 1.0, 1.5, 3.0]))
    with pytest.raises(TypeError, match="copy"):
        sb.persistence.BarcodeTensor(out)
    with pytest.raises(NotImplementedError, match="astype"):
        out.astype(sb.barcode32)


@pytest.mark.parametrize("operation", [
    pytest.param(lambda barcodes: sb.timeseries.TimeSeries(barcodes, times=[0.0, 1.0]), id="time_series"),
    pytest.param(lambda barcodes: sb.NestedTensor([barcodes]), id="nested_tensor"),
])
def test_operation_without_gradients_on_tracked_barcodes_raises(fw, operation):
    out = _tracked_batch(fw, fw.tensor([0.0, 1.0, 1.5, 3.0]))
    with pytest.raises(NotImplementedError, match="detach"):
        operation(out)


def test_detached_tracked_barcodes_are_the_untracked_ones(fw):
    out = _tracked_batch(fw, fw.tensor([0.0, 1.0, 1.5, 3.0]))

    detached = out.detach()

    assert type(detached) is sb.persistence.BarcodeTensor
    assert detached.array_equal(_expected_batch([0.0, 1.0, 1.5, 3.0]))


def test_tracked_barcodes_compare_by_value(fw):
    out = _tracked_batch(fw, fw.tensor([0.0, 1.0, 1.5, 3.0]))
    same = _expected_batch([0.0, 1.0, 1.5, 3.0])
    # Doubling the values of the PLF doubles the bars; the empty barcode stays equal.
    doubled = _expected_batch([0.0, 2.0, 3.0, 6.0])

    assert out.array_equal(same)
    assert not out.array_equal(doubled)
    np.testing.assert_array_equal(np.asarray(out == doubled), [[False, True],
                                                               [False, False]])
    np.testing.assert_array_equal(np.asarray(out != doubled), [[True, False],
                                                               [True, True]])


@pytest.mark.parametrize("round_trip", ROUND_TRIPS)
def test_saved_tracked_barcodes_load_as_untracked_ones(fw, round_trip):
    out = _tracked_batch(fw, fw.tensor([0.0, 1.0, 1.5, 3.0]))

    loaded = round_trip(out)

    assert type(loaded) is sb.persistence.BarcodeTensor
    assert loaded.array_equal(_expected_batch([0.0, 1.0, 1.5, 3.0]))


def test_operations_on_tracked_barcodes_convert_them_once(fw, monkeypatch):
    from stablebear import _sb_cpp as cpp

    calls = []
    from_serial_content = cpp.barcodes_from_serial_content_64

    def counting(content, enumeration):
        calls.append(1)
        return from_serial_content(content, enumeration)

    monkeypatch.setattr(cpp, "barcodes_from_serial_content_64", counting)
    out = _tracked_batch(fw, fw.tensor([0.0, 1.0, 1.5, 3.0]))
    f = sb.Plf([[0.0, 0.0], [3.0, 6.0]])
    sb.persistence.transform_bars(out, f)
    sb.persistence.transform_bars(out, f)
    assert len(calls) == 1


def test_tracked_barcodes_follow_in_place_change_of_torch_bars():
    torch = pytest.importorskip("torch")
    values = torch.tensor([0.0, 1.0, 1.5, 3.0], dtype=torch.float64)
    out = _tracked_batch(None, values)
    assert out.detach().is_isomorphic_to(_expected_batch([0.0, 1.0, 1.5, 3.0]), atol=0.0, rtol=0.0)

    with torch.no_grad():
        out.to_serial_content()[0].mul_(2)

    # Doubling the values of the PLF doubles the transformed bars.
    assert out.detach().is_isomorphic_to(_expected_batch([0.0, 2.0, 3.0, 6.0]), atol=0.0, rtol=0.0)


def _pcf_grid():
    # Breakpoints (t, 10 * t + k) for the PCF k, so each value shows its PCF.
    return sb.PcfTensor([
        sb.Pcf(np.array([[0.0, 1.0], [1.0, 11.0], [2.0, 21.0]])),
        sb.Pcf(np.array([[0.0, 2.0], [3.0, 32.0]])),
        sb.Pcf(np.array([[0.0, 3.0]])),
        sb.Pcf(np.array([[0.0, 4.0], [0.5, 9.0], [1.5, 19.0], [4.0, 44.0]])),
    ]).reshape((2, 2))


def test_tracked_pcfs_index_and_reshape_like_plain_ones(fw):
    plain = _pcf_grid()
    out = tracked(fw, plain)
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
        selected, expected = select(out), select(plain)
        assert isinstance(selected, sb.PcfTensor)
        assert selected.shape == expected.shape
        assert isinstance(selected.to_serial_content()[0], fw.tensor_type)
        assert selected.array_equal(expected)
        for i in np.ndindex(*expected.shape):
            assert selected[i] == expected[i]


def test_gradient_through_indexing_tracked_pcfs(fw):
    F, _ = _tensors()

    def loss(x):
        return fw.sum(sb.lp_norm((x * F)[[2, 0]], p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, 1.5)


def test_tracked_pcfs_cannot_lose_gradients(fw):
    out = tracked(fw, _pcf_grid())
    with pytest.raises(TypeError, match="copy"):
        sb.PcfTensor(out)
    with pytest.raises(NotImplementedError, match="detach"):
        sb.max_time(out)
    assert out.detach().array_equal(_pcf_grid())


def _written_pcf():
    return sb.Pcf(np.array([[0.0, 7.0], [5.0, 57.0]]))


@pytest.mark.parametrize("write", [
    pytest.param(lambda t: t.__setitem__((0, 1), _written_pcf()), id="element"),
    pytest.param(lambda t: t.__setitem__(1, _written_pcf()), id="element_to_row"),
    pytest.param(lambda t: t.__setitem__(0, sb.PcfTensor([_written_pcf(), _written_pcf() * 2.0])), id="tensor"),
    pytest.param(lambda t: t.__setitem__(sb.tensor(np.array([[False, True], [True, False]])), _written_pcf()),
                 id="mask"),
])
def test_write_to_tracked_pcfs_matches_write_to_plain_ones(fw, write):
    out, expected = tracked(fw, _pcf_grid()), _pcf_grid()

    write(out)
    write(expected)

    # Each element read sees the write, before and after it is applied.
    for i in np.ndindex(2, 2):
        assert out[i] == expected[i]
    assert out.array_equal(expected)
    for i in np.ndindex(2, 2):
        assert out[i] == expected[i]
    np.testing.assert_allclose(fw.to_numpy(sb.lp_norm(out, p=1.0)), np.asarray(sb.lp_norm(expected, p=1.0)))


def test_later_write_to_tracked_pcfs_replaces_earlier_one(fw):
    out = tracked(fw, _pcf_grid())

    out[0, 0] = _written_pcf()
    out[0, 0] = _written_pcf() * 2.0

    assert out[0, 0] == _written_pcf() * 2.0
    assert out.detach()[0, 0] == _written_pcf() * 2.0


def test_write_to_copy_of_tracked_pcfs_leaves_them_unchanged(fw):
    out = tracked(fw, _pcf_grid())
    row = out[0]

    row[1] = _written_pcf()

    assert row[1] == _written_pcf()
    assert out.array_equal(_pcf_grid())


def test_gradient_flows_to_written_element_and_not_to_replaced_one(fw):
    F, _ = _tensors()
    g = sb.Pcf(np.array([[0.0, 2.0], [1.0, 4.0], [3.0, 0.0]]))

    def loss(x):
        out = x * F
        out[1] = (x * x) * g
        return fw.sum(sb.lp_norm(out, p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, 1.5)


def test_writes_to_tracked_barcodes_are_applied_in_one_pass(fw, monkeypatch):
    Barcode = sb.persistence.Barcode
    out = tracked(fw, _barcode_batch())
    adapter = _autodiff._framework(out.to_serial_content()[0])
    calls = []
    concat = adapter.concat

    def counting(arrays):
        calls.append(1)
        return concat(arrays)

    monkeypatch.setattr(adapter, "concat", counting)

    out[0, 0] = Barcode([[0.0, 10.0]])
    out[0, 1] = Barcode([[0.0, 20.0], [1.0, 21.0]])
    out[1, 1] = Barcode([[0.0, 30.0]])
    assert len(calls) == 0

    np.testing.assert_array_equal(fw.to_numpy(out.to_serial_content()[0]), [[0.0, 10.0],
                                                              [0.0, 20.0],
                                                              [1.0, 21.0],
                                                              [2.5, 4.0],
                                                              [1.2, 2.1],
                                                              [0.1, 0.4],
                                                              [0.0, 30.0]])
    assert len(calls) == 1


def test_list_with_tracked_pcf_gives_tracked_tensor(fw):
    f = sb.Pcf(np.array([[0.0, 1.0], [1.0, 11.0]]))
    g = sb.Pcf(np.array([[0.0, 2.0], [3.0, 32.0], [4.0, 42.0]]))
    tracked_g = sb.Pcf(fw.tensor([2.0, 32.0, 42.0]), times=np.array([0.0, 3.0, 4.0]))

    out = sb.PcfTensor([[f, tracked_g],
                        [tracked_g, f]])

    assert out.shape == (2, 2)
    assert isinstance(out.to_serial_content()[0], fw.tensor_type)
    assert out.array_equal(sb.PcfTensor([[f, g],
                                         [g, f]]))


def test_gradient_through_list_of_tracked_pcfs(fw):
    f, g = _f_and_g()

    def loss(x):
        return fw.sum(sb.lp_norm(sb.PcfTensor([x * f, g, (x * x) * g]), p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, 1.5)


def _betti_curve(barcodes):
    return sb.persistence.barcode_to_betti_curve(barcodes, verbose=False)


def _accumulated_persistence(barcodes):
    return sb.persistence.barcode_to_accumulated_persistence(barcodes, max_death=float("inf"), verbose=False)


def _accumulated_persistence_of_early_deaths(barcodes):
    # Leaves out the bars of _barcode_batch that die at 3.7 and 4.0.
    return sb.persistence.barcode_to_accumulated_persistence(barcodes, max_death=2.5, verbose=False)


@pytest.mark.parametrize("summary", [_betti_curve, _accumulated_persistence, _accumulated_persistence_of_early_deaths])
def test_summary_of_tracked_batch_matches_untracked(fw, summary):
    out, expected = summary(tracked(fw, _barcode_batch())), summary(_barcode_batch())
    assert isinstance(out, sb.PcfTensor)
    assert out.shape == (2, 2)
    t = np.linspace(0.0, 4.5, 91)
    for i in np.ndindex(2, 2):
        np.testing.assert_allclose(out[i](t), expected[i](t))


def test_betti_curve_of_tracked_barcode_with_infinite_bar_matches_untracked(fw):
    Barcode, BarcodeTensor = sb.persistence.Barcode, sb.persistence.BarcodeTensor
    barcodes = BarcodeTensor([Barcode([[0.0, 2.0], [1.0, np.inf]])])
    out, expected = _betti_curve(tracked(fw, barcodes)), _betti_curve(barcodes)
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    np.testing.assert_array_equal(out[0](t), [1.0, 1.0, 2.0, 2.0, 1.0, 1.0])
    np.testing.assert_array_equal(out[0](t), expected[0](t))


def test_betti_curve_norm_gradient(fw):
    def loss(v):
        barcodes = sb.persistence.transform_bars(_barcode_batch(), sb.Plf(v, times=_batch_plf_times()))
        return fw.sum(sb.lp_norm(_betti_curve(barcodes), p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, [0.0, 1.0, 1.5, 3.0])


@pytest.mark.parametrize("summary", [_accumulated_persistence, _accumulated_persistence_of_early_deaths])
def test_accumulated_persistence_gradient(fw, summary):
    # The function does not return to 0, so its norms are infinite: the loss
    # is the sum of time * value over its breakpoints.
    def loss(v):
        out = summary(sb.persistence.transform_bars(_barcode_batch(), sb.Plf(v, times=_batch_plf_times())))
        breakpoints, _ = out.to_serial_content()
        return fw.sum(breakpoints[:, 0] * breakpoints[:, 1])

    _assert_gradient_matches_finite_differences(fw, loss, [0.0, 1.0, 1.5, 3.0])


_PAIRWISE = [
    pytest.param(lambda fs: sb.pdist(fs, p=1.0, verbose=False), id="pdist_l1"),
    pytest.param(lambda fs: sb.pdist(fs, p=2.5, verbose=False), id="pdist_lp"),
    pytest.param(lambda fs: sb.l2_kernel(fs, verbose=False), id="l2_kernel"),
]


@pytest.mark.parametrize("pairwise", _PAIRWISE)
def test_pairwise_matrix_of_tracked_pcfs_matches_untracked(fw, pairwise):
    F, _ = _tensors()
    out = pairwise(tracked(fw, F))
    assert isinstance(out, fw.tensor_type)
    np.testing.assert_allclose(fw.to_numpy(out), pairwise(F).to_dense())


def _pairwise_weights():
    # Not symmetric, so that both entries of each pair count.
    return np.array([[1.0, 2.0, -1.0],
                     [0.5, 3.0, 4.0],
                     [-2.0, 1.5, 0.25]])


@pytest.mark.parametrize("pairwise", _PAIRWISE)
def test_pairwise_matrix_gradient_wrt_pcf_values(fw, pairwise):
    f, g = _f_and_g()
    times = np.array([0.0, 0.5, 2.0, 4.5, 8.0])

    def loss(c):
        h = sb.Pcf(fw.concat([c, fw.tensor([0.0])]), times=times)
        return fw.sum(fw.tensor(_pairwise_weights()) * pairwise(sb.PcfTensor([f, h, g])))

    _assert_gradient_matches_finite_differences(fw, loss, [0.7, -1.3, 0.4, 2.0])


@pytest.mark.parametrize("pairwise", _PAIRWISE)
def test_pairwise_matrix_gradient_wrt_pcf_times(fw, pairwise):
    f, g = _f_and_g()

    def loss(t):
        h = sb.Pcf(_VALUES, times=fw.concat([fw.tensor([0.0]), t]))
        return fw.sum(fw.tensor(_pairwise_weights()) * pairwise(sb.PcfTensor([f, h, g])))

    _assert_gradient_matches_finite_differences(fw, loss, _TIMES)


@pytest.mark.parametrize("pairwise", _PAIRWISE)
def test_pairwise_matrix_gradient_through_arithmetic(fw, pairwise):
    F, _ = _tensors()

    def loss(x):
        return fw.sum(fw.tensor(_pairwise_weights()) * pairwise(x * F))

    _assert_gradient_matches_finite_differences(fw, loss, 1.5)


def _loop_points():
    # Six points around a loop, with all pairwise distances different.
    return np.array([[0.0, 0.0], [2.0, 0.1], [3.1, 1.5], [2.3, 3.2], [0.3, 2.9], [-0.9, 1.3]])


@pytest.mark.parametrize("reduced", [False, True], ids=["unreduced", "reduced"])
def test_persistent_homology_of_framework_points_matches_numpy(fw, reduced):
    homology = sb.persistence.compute_persistent_homology
    out = homology(fw.tensor(_loop_points()), max_dim=1, reduced=reduced, verbose=False)
    expected = homology(_loop_points(), max_dim=1, reduced=reduced, verbose=False)
    assert isinstance(out, sb.persistence.BarcodeTensor)
    assert out.shape == (2,)
    assert isinstance(out.to_serial_content()[0], fw.tensor_type)
    # The loop gives one bar in dimension 1.
    assert len(out[1]) == 1
    for dim in range(2):
        assert out[dim].is_isomorphic_to(expected[dim], atol=1e-12, rtol=1e-12)


def test_persistent_homology_of_framework_points_with_equal_distances_matches_numpy(fw):
    # A square: its sides are equal, and so are its diagonals.
    square = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]])
    homology = sb.persistence.compute_persistent_homology
    out = homology(fw.tensor(square), max_dim=1, reduced=True, verbose=False)
    expected = homology(square, max_dim=1, reduced=True, verbose=False)
    for dim in range(2):
        assert out[dim].is_isomorphic_to(expected[dim], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dim", [0, 1])
def test_persistent_homology_gradient(fw, dim):
    def loss(P):
        barcodes = sb.persistence.compute_persistent_homology(P, max_dim=1, reduced=True, verbose=False)
        bars = barcodes[dim:dim + 1].to_serial_content()[0]
        return fw.sum(bars[:, 1] * bars[:, 1]) + fw.sum(bars[:, 0])

    _assert_gradient_matches_finite_differences(fw, loss, _loop_points())


def test_stable_rank_norm_of_persistent_homology_gradient(fw):
    def loss(P):
        barcodes = sb.persistence.compute_persistent_homology(P, max_dim=1, reduced=True, verbose=False)
        return fw.sum(sb.lp_norm(sb.persistence.barcode_to_stable_rank(barcodes), p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, _loop_points())


def test_persistent_homology_of_framework_array_of_several_point_clouds_matches_numpy(fw):
    clouds = np.stack([_loop_points(), 2.0 * _loop_points()])
    homology = sb.persistence.compute_persistent_homology

    out = homology(fw.tensor(clouds), max_dim=1, reduced=True, verbose=False)

    expected = homology(clouds, max_dim=1, reduced=True, verbose=False)
    assert isinstance(out.to_serial_content()[0], fw.tensor_type)
    assert out.shape == (2, 2)
    for i in np.ndindex(2, 2):
        assert out[i].is_isomorphic_to(expected[i], atol=1e-12, rtol=1e-12)


def test_arithmetic_on_stable_rank_with_equal_lifetimes_matches_numpy(fw):
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


def _point_clouds():
    # Points (i, 10 * i), numbered through the clouds.
    return sb.PointCloudTensor([
        np.array([[1.0, 10.0]]),
        np.array([[2.0, 20.0], [3.0, 30.0]]),
        np.array([[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]]),
        np.array([[7.0, 70.0]]),
    ]).reshape((2, 2))


def test_point_cloud_serial_content_round_trip():
    clouds = _point_clouds()
    content, enumeration = clouds._serial()
    np.testing.assert_array_equal(content, [[1.0, 10.0],
                                            [2.0, 20.0],
                                            [3.0, 30.0],
                                            [4.0, 40.0],
                                            [5.0, 50.0],
                                            [6.0, 60.0],
                                            [7.0, 70.0]])
    np.testing.assert_array_equal(enumeration, [[[0, 1], [1, 3]],
                                                [[3, 6], [6, 7]]])
    assert sb.PointCloudTensor._from_serial(content, enumeration).array_equal(clouds)


def test_serial_content_of_point_clouds_of_different_dimensions_raises():
    clouds = sb.PointCloudTensor([np.array([[1.0, 10.0]]), np.array([[2.0, 20.0, 200.0]])])
    with pytest.raises(ValueError, match="dimension"):
        clouds._serial()


def test_tracked_point_clouds_index_and_reshape_like_plain_ones(fw):
    plain = _point_clouds()
    out = tracked(fw, plain)
    for select in (
        lambda t: t[1],
        lambda t: t[:, 0],
        lambda t: t[[1, 0]],
        lambda t: t.reshape((4,)),
        lambda t: t.flatten()[::-1],
        lambda t: t.transpose(),
        lambda t: t.copy(),
    ):
        selected, expected = select(out), select(plain)
        assert isinstance(selected, sb.PointCloudTensor)
        assert selected.shape == expected.shape
        assert isinstance(selected.to_serial_content()[0], fw.tensor_type)
        assert selected.array_equal(expected)
        for i in np.ndindex(*expected.shape):
            np.testing.assert_array_equal(np.asarray(selected[i]), np.asarray(expected[i]))


def _distance_matrices():
    # Matrices of 2, 1, 0, and 3 points.
    matrices = sb.zeros((2, 2), dtype=sb.distmat64)
    matrices[0, 0] = sb.DistanceMatrix(np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ]))
    matrices[0, 1] = sb.DistanceMatrix(np.array([[0.0]]))
    matrices[1, 1] = sb.DistanceMatrix(np.array([
        [0.0, 2.0, 3.0],
        [2.0, 0.0, 4.0],
        [3.0, 4.0, 0.0],
    ]))
    return matrices


def _symmetric_matrices():
    # Matrices of 2, 1, 0, and 3 rows.
    matrices = sb.zeros((2, 2), dtype=sb.symmat64)
    matrices[0, 0] = sb.SymmetricMatrix(np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ]))
    matrices[0, 1] = sb.SymmetricMatrix(np.array([[4.0]]))
    matrices[1, 1] = sb.SymmetricMatrix(np.array([
        [5.0, 6.0, 8.0],
        [6.0, 7.0, 9.0],
        [8.0, 9.0, 10.0],
    ]))
    return matrices


def test_distance_matrix_serial_content_round_trip():
    matrices = _distance_matrices()
    content, enumeration, sizes = matrices._serial()
    np.testing.assert_array_equal(content, [[1.0], [2.0], [3.0], [4.0]])
    np.testing.assert_array_equal(enumeration, [[[0, 1], [1, 1]],
                                                [[1, 1], [1, 4]]])
    # A matrix of one point and a matrix of none both have no entries.
    np.testing.assert_array_equal(sizes, [[2, 1],
                                          [0, 3]])
    assert sb.DistanceMatrixTensor._from_serial(content, enumeration, sizes).array_equal(matrices)


def test_symmetric_matrix_serial_content_round_trip():
    matrices = _symmetric_matrices()
    content, enumeration, sizes = matrices._serial()
    np.testing.assert_array_equal(content, [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
    np.testing.assert_array_equal(enumeration, [[[0, 3], [3, 4]],
                                                [[4, 4], [4, 10]]])
    np.testing.assert_array_equal(sizes, [[2, 1],
                                          [0, 3]])
    assert sb.SymmetricMatrixTensor._from_serial(content, enumeration, sizes).array_equal(matrices)


def test_serial_content_of_matrices_with_sizes_that_do_not_match_raises():
    content, enumeration, sizes = _distance_matrices()._serial()
    sizes[1, 1] = 4
    with pytest.raises(ValueError, match="entries"):
        sb.DistanceMatrixTensor._from_serial(content, enumeration, sizes)


def test_serial_content_of_distance_matrices_with_negative_entry_raises():
    content, enumeration, sizes = _distance_matrices()._serial()
    content[0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        sb.DistanceMatrixTensor._from_serial(content, enumeration, sizes)


@pytest.mark.parametrize("make_matrices", [_distance_matrices, _symmetric_matrices], ids=["distance", "symmetric"])
def test_tracked_matrices_index_and_reshape_like_plain_ones(fw, make_matrices):
    plain = make_matrices()
    out = tracked(fw, plain)
    for select in (
        lambda t: t[1],
        lambda t: t[:, 0],
        lambda t: t[[1, 0]],
        lambda t: t.reshape((4,)),
        lambda t: t.flatten()[::-1],
        lambda t: t.transpose(),
        lambda t: t.copy(),
    ):
        selected, expected = select(out), select(plain)
        assert isinstance(selected, type(plain))
        assert selected.shape == expected.shape
        assert isinstance(selected.to_serial_content()[0], fw.tensor_type)
        assert selected.array_equal(expected)
        for i in np.ndindex(*expected.shape):
            np.testing.assert_array_equal(selected[i].to_dense(), expected[i].to_dense())


@pytest.mark.parametrize("make_matrices", [_distance_matrices, _symmetric_matrices], ids=["distance", "symmetric"])
def test_write_to_tracked_matrices_raises(fw, make_matrices):
    out = tracked(fw, make_matrices())
    with pytest.raises(NotImplementedError, match="Writing"):
        out[0, 0] = make_matrices()[1, 1]


def test_serial_content_of_pcf_tensor_round_trips():
    F = _pcf_grid()

    content, enumeration = F.to_serial_content()

    assert content.shape == (10, 2)
    assert enumeration.shape == (2, 2, 2)
    assert sb.from_serial_content(content, enumeration).array_equal(F)


@pytest.mark.parametrize("make_tensor", [
    _pcf_grid, _barcode_batch, _point_clouds, _distance_matrices, _symmetric_matrices])
def test_serial_content_of_tracked_tensor_is_a_framework_array_of_its_values(fw, make_tensor):
    plain = make_tensor()

    content, *rows = tracked(fw, plain).to_serial_content()

    expected_content, *expected_rows = plain.to_serial_content()
    assert isinstance(content, fw.tensor_type)
    np.testing.assert_array_equal(fw.to_numpy(content), expected_content)
    # The enumeration and, for matrices, the sizes.
    assert len(rows) == len(expected_rows)
    for array, expected in zip(rows, expected_rows):
        np.testing.assert_array_equal(array, expected)


def test_gradient_of_loss_on_serial_content(fw):
    F, _ = _tensors()

    def loss(x):
        breakpoints, _ = (x * F).to_serial_content()
        return fw.sum(breakpoints[:, 1] * breakpoints[:, 1])

    _assert_gradient_matches_finite_differences(fw, loss, 1.5)


def _numbers():
    return np.array([[1.0, 2.0],
                     [3.0, 4.0]])


def _other_numbers():
    return np.array([[2.0, 4.0],
                     [8.0, 16.0]])


_BINARY_OPERATIONS = [
    pytest.param(lambda a, b: a + b, id="add"),
    pytest.param(lambda a, b: a - b, id="sub"),
    pytest.param(lambda a, b: a * b, id="mul"),
    pytest.param(lambda a, b: a / b, id="truediv"),
    pytest.param(lambda a, b: a ** b, id="pow"),
    pytest.param(lambda a, b: a // b, id="floordiv"),
]


@pytest.mark.parametrize("operation", _BINARY_OPERATIONS)
@pytest.mark.parametrize("order", ["tensor_first", "framework_first"])
@pytest.mark.parametrize("make_tensor", [
    pytest.param(lambda: sb.FloatTensor(_numbers()), id="float64"),
    pytest.param(lambda: sb.IntTensor(_numbers().astype(np.int64)), id="int64"),
])
def test_arithmetic_of_numeric_tensor_and_framework_array_matches_numpy(fw, operation, order, make_tensor):
    tensor, array = make_tensor(), fw.tensor(_other_numbers())

    out = operation(tensor, array) if order == "tensor_first" else operation(array, tensor)

    expected = (operation(_numbers(), _other_numbers()) if order == "tensor_first"
                else operation(_other_numbers(), _numbers()))
    assert isinstance(out, fw.tensor_type)
    np.testing.assert_allclose(fw.to_numpy(out), expected)


@pytest.mark.parametrize("order", ["tensor_first", "framework_first"])
def test_arithmetic_of_numeric_tensor_and_framework_scalar_matches_numpy(fw, order):
    tensor, scalar = sb.FloatTensor(_numbers()), fw.tensor(2.0)

    out = tensor / scalar if order == "tensor_first" else scalar / tensor

    assert isinstance(out, fw.tensor_type)
    np.testing.assert_allclose(fw.to_numpy(out), _numbers() / 2.0 if order == "tensor_first" else 2.0 / _numbers())


# Without the powers, whose values here are too large for finite differences,
# and the floor division, whose gradient is zero.
@pytest.mark.parametrize("operation", _BINARY_OPERATIONS[:4])
@pytest.mark.parametrize("order", ["tensor_first", "framework_first"])
def test_gradient_through_arithmetic_with_numeric_tensor(fw, operation, order):
    tensor = sb.FloatTensor(_numbers())

    def loss(x):
        return fw.sum(operation(tensor, x) if order == "tensor_first" else operation(x, tensor))

    _assert_gradient_matches_finite_differences(fw, loss, _other_numbers())


def test_gradient_of_norms_divided_by_framework_scalar(fw):
    F, _ = _tensors()

    # The norms of plain PCFs are a FloatTensor.
    def loss(x):
        return fw.sum(sb.lp_norm(F, p=2.0) / x)

    _assert_gradient_matches_finite_differences(fw, loss, 2.0)


@pytest.mark.parametrize("comparison, expected", [
    pytest.param(lambda a, b: a > b, [[False, False], [False, False]], id="gt"),
    pytest.param(lambda a, b: a <= b, [[True, True], [True, True]], id="le"),
    pytest.param(lambda a, b: a == b, [[False, False], [False, False]], id="eq"),
    pytest.param(lambda a, b: a != b, [[True, True], [True, True]], id="ne"),
])
def test_comparison_of_numeric_tensor_and_framework_array(fw, comparison, expected):
    out = comparison(sb.FloatTensor(_numbers()), fw.tensor(_other_numbers()))

    assert isinstance(out, fw.tensor_type)
    np.testing.assert_array_equal(fw.to_numpy(out), expected)


def test_boolean_tensor_compares_with_framework_array(fw):
    flags = sb.tensor(np.array([True, False]))

    out = flags == (fw.tensor([1.0, 1.0]) > 0.0)

    assert isinstance(out, fw.tensor_type)
    np.testing.assert_array_equal(fw.to_numpy(out), [True, False])


@pytest.mark.parametrize("operation", [
    pytest.param(lambda t, x: t.__iadd__(x), id="iadd"),
    pytest.param(lambda t, x: t.__isub__(x), id="isub"),
    pytest.param(lambda t, x: t.__imul__(x), id="imul"),
    pytest.param(lambda t, x: t.__itruediv__(x), id="itruediv"),
    pytest.param(lambda t, x: t.__ipow__(x), id="ipow"),
    pytest.param(lambda t, x: t.__ifloordiv__(x), id="ifloordiv"),
])
def test_in_place_arithmetic_of_numeric_tensor_with_framework_array_raises(fw, operation):
    tensor = sb.FloatTensor(_numbers())
    with pytest.raises(TypeError, match="gradients"):
        operation(tensor, fw.tensor(2.0))
    np.testing.assert_array_equal(np.asarray(tensor), _numbers())


def test_in_place_power_of_pcf_tensor_with_framework_exponent_raises(fw):
    F, _ = _tensors()
    with pytest.raises(TypeError, match="PyTorch"):
        F **= fw.tensor(2.0)


def _delay_embed(delay):
    series = sb.timeseries.TimeSeries([0.0, 10.0, 25.0, 40.0], times=[0.0, 1.0, 2.5, 4.0])
    return sb.timeseries.delay_embed(series, dimension=2, delay=delay, direction="backward")


_NUMBER_ARGUMENTS = [
    pytest.param(lambda p: sb.pdist(_tensors()[0], p=p, verbose=False), id="pdist"),
    pytest.param(lambda p: sb.cdist(_tensors()[0], _tensors()[1], p=p, verbose=False), id="cdist"),
    pytest.param(lambda p: sb.LpDistance(p), id="lp_distance_object"),
    pytest.param(lambda max_death: sb.persistence.barcode_to_accumulated_persistence(
        _barcode_batch(), max_death=max_death, verbose=False), id="max_death"),
    pytest.param(_delay_embed, id="delay"),
]


@pytest.mark.parametrize("operation", _NUMBER_ARGUMENTS)
def test_framework_scalar_where_a_number_is_required_raises(fw, operation):
    with pytest.raises(TypeError, match="must be a number"):
        operation(fw.tensor(2.0))


@pytest.mark.parametrize("operation", _NUMBER_ARGUMENTS)
def test_number_where_a_number_is_required_is_accepted(operation):
    operation(2.0)


def _evaluation_times():
    # Between breakpoints, at a breakpoint, and after the last one.
    return np.array([[0.5, 1.0],
                     [2.25, 9.0]])


def _plain_pcf():
    return sb.Pcf(np.array([[0.0, 10.0], [1.0, 20.0], [3.0, 30.0]]))


def _plain_plf():
    return sb.Plf(np.array([[0.0, 10.0], [1.0, 20.0], [3.0, 60.0]]))


@pytest.mark.parametrize("make_function", [_plain_pcf, _plain_plf], ids=["pcf", "plf"])
def test_function_at_framework_times_matches_numpy(fw, make_function):
    f = make_function()

    out = f(fw.tensor(_evaluation_times()))

    assert isinstance(out, fw.tensor_type)
    np.testing.assert_allclose(fw.to_numpy(out), f(_evaluation_times()))


@_FUNCTIONS
def test_tracked_function_at_times_matches_numpy(fw, function):
    values, times = [10.0, 20.0, 30.0], np.array([0.0, 1.0, 3.0])
    f = function(fw.tensor(values), times=times)

    out = f(_evaluation_times())

    assert isinstance(out, fw.tensor_type)
    np.testing.assert_allclose(fw.to_numpy(out), function(np.array(values), times=times)(_evaluation_times()))
    # A single time gives a single value.
    np.testing.assert_allclose(fw.to_numpy(f(0.5)), function(np.array(values), times=times)(0.5))


def _evaluation_weights():
    return np.array([[1.0, -2.0],
                     [3.0, 0.5]])


@_FUNCTIONS
def test_gradient_of_evaluation_wrt_values(fw, function):
    def loss(v):
        f = function(v, times=np.array([0.0, 1.0, 3.0]))
        return fw.sum(fw.tensor(_evaluation_weights()) * f(_evaluation_times()))

    _assert_gradient_matches_finite_differences(fw, loss, [10.0, 20.0, 30.0])


def test_gradient_of_plf_evaluation_wrt_its_times(fw):
    def loss(t):
        f = sb.Plf(np.array([10.0, 20.0, 60.0]), times=fw.concat([fw.tensor([0.0]), t]))
        return fw.sum(fw.tensor(_evaluation_weights()) * f(_evaluation_times()))

    _assert_gradient_matches_finite_differences(fw, loss, [1.25, 3.0])


def test_gradient_of_plf_evaluation_wrt_where_it_is_evaluated(fw):
    f = _plain_plf()

    def loss(t):
        return fw.sum(fw.tensor(_evaluation_weights()) * f(t))

    _assert_gradient_matches_finite_differences(fw, loss, np.array([[0.5, 1.5],
                                                                   [2.25, 9.0]]))


def test_tracked_pcf_tensor_at_times_matches_numpy(fw):
    F = _pcf_grid()

    out = tracked(fw, F)(np.array([0.25, 1.5, 5.0]))

    assert isinstance(out, fw.tensor_type)
    assert tuple(out.shape) == (2, 2, 3)
    np.testing.assert_allclose(fw.to_numpy(out), F(np.array([0.25, 1.5, 5.0])))


def test_gradient_of_pcf_tensor_evaluation(fw):
    F, _ = _tensors()

    def loss(x):
        return fw.sum((x * F)(np.array([0.25, 1.5, 3.25])))

    _assert_gradient_matches_finite_differences(fw, loss, 1.5)


def test_plf_tensor_at_framework_times_raises(fw):
    F = sb.PlfTensor([_plain_plf()])
    with pytest.raises(TypeError, match="PlfTensor"):
        F(fw.tensor([0.5]))


def _homology_of(points):
    return sb.persistence.compute_persistent_homology(points, max_dim=1, reduced=True, verbose=False)


def test_integer_index_into_tracked_barcodes_gives_tracked_barcode(fw):
    out = _homology_of(fw.tensor(_loop_points()))
    expected = _homology_of(_loop_points())

    barcode = out[1]

    assert isinstance(barcode, sb.persistence.Barcode)
    assert isinstance(barcode.to_serial_content()[0], fw.tensor_type)
    # Its values read as those of a plain barcode.
    assert len(barcode) == len(expected[1])
    np.testing.assert_allclose(barcode.to_numpy(), expected[1].to_numpy())
    assert barcode.is_isomorphic_to(expected[1], atol=1e-12, rtol=1e-12)


def test_gradient_through_integer_index_matches_that_through_a_slice(fw):
    def lengths(barcodes):
        bars, _ = barcodes.to_serial_content()
        return fw.sum(bars[:, 1] - bars[:, 0])

    _, by_index = fw.value_and_grad(lambda P: lengths(_homology_of(P)[1]), _loop_points())
    _, by_slice = fw.value_and_grad(lambda P: lengths(_homology_of(P)[1:2]), _loop_points())

    np.testing.assert_allclose(by_index, by_slice)
    assert np.abs(by_index).max() > 0.1


@pytest.mark.parametrize("summary", [_betti_curve, _accumulated_persistence, lambda barcodes: (
    sb.persistence.barcode_to_stable_rank(barcodes, verbose=False))], ids=["betti", "accumulated", "stable_rank"])
def test_summary_of_tracked_barcode_is_a_tracked_pcf(fw, summary):
    out = _homology_of(fw.tensor(_loop_points()))
    expected = _homology_of(_loop_points())

    curve = summary(out[0])

    assert isinstance(curve, sb.Pcf)
    assert isinstance(sb.PcfTensor([curve]).to_serial_content()[0], fw.tensor_type)
    t = np.linspace(0.0, 4.0, 41)
    np.testing.assert_allclose(fw.to_numpy(curve(t)), summary(expected[0])(t))


def test_iterating_tracked_tensor_gives_tracked_elements(fw):
    out = _homology_of(fw.tensor(_loop_points()))

    barcodes = list(out)

    assert len(barcodes) == 2
    for barcode in barcodes:
        assert isinstance(barcode.to_serial_content()[0], fw.tensor_type)


def test_integer_index_into_tracked_pcfs_and_point_clouds(fw):
    pcf = tracked(fw, _pcf_grid())[0, 1]
    assert isinstance(pcf, sb.Pcf)
    assert pcf == _pcf_grid()[0, 1]
    assert isinstance(sb.PcfTensor([pcf]).to_serial_content()[0], fw.tensor_type)

    # A single point cloud with gradients is the array of its points.
    cloud = tracked(fw, _point_clouds())[1, 0]
    assert isinstance(cloud, fw.tensor_type)
    np.testing.assert_array_equal(fw.to_numpy(cloud), np.asarray(_point_clouds()[1, 0]))


def test_barcode_tensor_of_list_with_tracked_barcode_carries_gradients(fw):
    out = _homology_of(fw.tensor(_loop_points()))
    plain = sb.persistence.Barcode([[0.0, 1.0]])

    barcodes = sb.persistence.BarcodeTensor([out[1], plain])

    assert barcodes.shape == (2,)
    assert isinstance(barcodes.to_serial_content()[0], fw.tensor_type)
    assert barcodes[1].is_isomorphic_to(plain, atol=0.0, rtol=0.0)


def _assert_same_functions(out, expected):
    """The PCFs of two tensors have the same values on a grid of times."""
    t = np.linspace(0.0, 7.0, 71)
    assert out.shape == expected.shape
    np.testing.assert_allclose(np.asarray(out.detach()(t)), np.asarray(expected(t)))


@pytest.mark.parametrize("operation", [
    pytest.param(lambda F, w: F * w, id="mul"),
    pytest.param(lambda F, w: w * F, id="rmul"),
    pytest.param(lambda F, w: F / w, id="truediv"),
    pytest.param(lambda F, w: F + w, id="add"),
    pytest.param(lambda F, w: w - F, id="rsub"),
])
def test_pcf_arithmetic_with_one_framework_number_per_pcf(fw, operation):
    F, _ = _tensors()
    weights = [2.0, -3.0, 0.5]

    out = operation(F, fw.tensor(weights))

    expected = sb.PcfTensor([operation(F[i], weights[i]) for i in range(3)])
    _assert_same_functions(out, expected)


@pytest.mark.parametrize("make_weights", [
    pytest.param(lambda: np.array([2.0, -3.0, 0.5]), id="numpy"),
    pytest.param(lambda: sb.FloatTensor(np.array([2.0, -3.0, 0.5])), id="float_tensor"),
])
def test_tracked_pcf_arithmetic_with_one_number_per_pcf(fw, make_weights):
    F, _ = _tensors()

    out = tracked(fw, F) * make_weights()

    expected = sb.PcfTensor([F[0] * 2.0, F[1] * -3.0, F[2] * 0.5])
    _assert_same_functions(out, expected)


def test_single_pcf_times_framework_vector_gives_one_pcf_per_number(fw):
    f, _ = _f_and_g()

    out = f * fw.tensor([2.0, -3.0])

    _assert_same_functions(out, sb.PcfTensor([f * 2.0, f * -3.0]))


def test_gradient_wrt_one_weight_per_pcf(fw):
    F, G = _tensors()

    def loss(w):
        return fw.sum(sb.lp_norm(F * w - G, p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, [2.0, -3.0, 0.5])


def test_pcf_arithmetic_with_numbers_of_another_shape_raises(fw):
    F, _ = _tensors()
    with pytest.raises(ValueError, match="broadcast"):
        _ = F * fw.tensor([1.0, 2.0])


def _square():
    # A symmetric matrix with a zero diagonal: distances between three points.
    return np.array([[0.0, 3.0, 4.0],
                     [3.0, 0.0, 5.0],
                     [4.0, 5.0, 0.0]])


def _write(make, index):
    def write(array):
        tensor = make()
        tensor[index] = array
        return np.asarray(tensor)
    return write


# What takes a dense array, with an input for it and what it gives for that
# input as a NumPy array.
_DENSE_INPUTS = [
    pytest.param(lambda a: np.asarray(sb.FloatTensor(a)), _square(), id="float_tensor"),
    pytest.param(lambda a: np.asarray(sb.tensor(a)), _square(), id="tensor"),
    pytest.param(lambda a: np.asarray(sb.PointCloud(a)), _square(), id="point_cloud"),
    pytest.param(lambda a: sb.DistanceMatrix(a).to_dense(), _square(), id="distance_matrix"),
    pytest.param(lambda a: sb.SymmetricMatrix(a).to_dense(), _square(), id="symmetric_matrix"),
    pytest.param(lambda a: sb.persistence.Barcode(a).to_numpy(), np.array([[0.0, 1.0], [0.5, 2.0]]), id="barcode"),
    pytest.param(lambda a: sb.from_serial_content(a, np.array([[0, 2]]))[0].to_numpy(),
                 np.array([[0.0, 10.0], [1.0, 20.0]]), id="from_serial_content"),
    pytest.param(lambda a: sb.persistence.linkage_to_barcode(a, reduced=True).to_numpy(),
                 np.array([[0.0, 1.0, 1.5, 2.0], [2.0, 3.0, 4.5, 3.0]]), id="linkage"),
    pytest.param(_write(lambda: sb.FloatTensor(np.zeros((2, 3))), 1), np.array([1.0, 2.0, 3.0]), id="write"),
]


@pytest.mark.parametrize("operation, array", _DENSE_INPUTS)
def test_framework_array_without_gradients_is_read_as_a_dense_array(fw, operation, array):
    np.testing.assert_array_equal(operation(fw.tensor(array)), operation(array))


@pytest.mark.parametrize("operation, array", _DENSE_INPUTS)
def test_framework_array_with_gradients_as_a_dense_array_raises(fw, operation, array):
    with pytest.raises(TypeError, match="cannot carry the gradients"):
        operation(fw.tensor(array).requires_grad_())


def test_allclose_with_framework_array_compares_values(fw):
    array = fw.tensor(_square()).requires_grad_()

    assert sb.allclose(sb.FloatTensor(_square()), array, atol=0.0, rtol=0.0)
    assert not sb.allclose(sb.FloatTensor(_square() + 1.0), array, atol=0.0, rtol=0.0)


def test_values_with_times_match_breakpoints():
    g = sb.Pcf(np.array([2.0, -1.0, 0.0]), times=np.array([0.0, 1.0, 3.0]))
    np.testing.assert_array_equal(g.to_numpy(), [[0.0, 2.0], [1.0, -1.0], [3.0, 0.0]])


@pytest.mark.parametrize("function", [sb.Pcf, sb.Plf], ids=["pcf", "plf"])
def test_function_of_framework_values_detaches_to_a_plain_one(fw, function):
    f = function(fw.tensor([10.0, 20.0]), times=np.array([0.0, 1.0]))
    assert isinstance(f, function)

    detached = f.detach()

    assert type(detached) is function
    assert detached == function(np.array([[0.0, 10.0],
                                          [1.0, 20.0]]))


@pytest.mark.parametrize("function", [sb.Pcf, sb.Plf], ids=["pcf", "plf"])
def test_function_follows_in_place_change_of_torch_values(function):
    torch = pytest.importorskip("torch")
    values = torch.tensor([10.0, 20.0], dtype=torch.float64)
    f = function(values, times=np.array([0.0, 1.0]))
    np.testing.assert_array_equal(f.to_numpy(), [[0.0, 10.0],
                                                 [1.0, 20.0]])

    with torch.no_grad():
        values.mul_(2)

    np.testing.assert_array_equal(f.to_numpy(), [[0.0, 20.0],
                                                 [1.0, 40.0]])


def _tracked_function(fw, function):
    return function(fw.tensor([10.0, 20.0]), times=np.array([0.0, 1.0]))


@_FUNCTIONS
@pytest.mark.parametrize("operation", [
    pytest.param(lambda function, f: f.astype(sb.float32), id="astype"),
    pytest.param(lambda function, f: function(f), id="copy"),
])
def test_operation_that_would_drop_gradients_of_tracked_function_raises(fw, function, operation):
    f = _tracked_function(fw, function)
    with pytest.raises(NotImplementedError, match="detach"):
        operation(function, f)


def test_integrating_tracked_pcf_raises(fw):
    f = _tracked_function(fw, sb.Pcf)
    with pytest.raises(NotImplementedError, match="detach"):
        f.integrate()


@_FUNCTIONS
def test_tracked_function_gives_its_values(fw, function):
    f = _tracked_function(fw, function)
    plain = function(np.array([[0.0, 10.0],
                               [1.0, 20.0]]))

    np.testing.assert_array_equal(f.to_numpy(), plain.to_numpy())
    np.testing.assert_array_equal(np.asarray(f), np.asarray(plain))
    assert f.size == 2
    assert f == plain
    assert str(f) == str(plain)


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


_SUMS = pytest.mark.parametrize("reduce", [sb.sum, sb.mean], ids=["sum", "mean"])


@_SUMS
@pytest.mark.parametrize("make_pcfs, dim", [
    pytest.param(_pcf_grid, 0, id="first_axis"),
    pytest.param(_pcf_grid, 1, id="last_axis"),
    pytest.param(_pcf_grid, -1, id="negative_axis"),
    pytest.param(lambda: _pcf_grid().flatten(), 0, id="one_axis"),
])
def test_sum_of_tracked_pcfs_matches_untracked(fw, reduce, make_pcfs, dim):
    out = reduce(tracked(fw, make_pcfs()), dim=dim)

    assert isinstance(out, sb.PcfTensor)
    assert isinstance(out.to_serial_content()[0], fw.tensor_type)
    _assert_same_functions(out, reduce(make_pcfs(), dim=dim))


@_SUMS
def test_sum_of_selected_tracked_pcfs_matches_untracked(fw, reduce):
    out = reduce(tracked(fw, _pcf_grid())[::-1, [1, 0]], dim=1)

    _assert_same_functions(out, reduce(_pcf_grid()[::-1, [1, 0]], dim=1))


@_SUMS
def test_sum_of_tracked_pcfs_along_missing_axis_raises(fw, reduce):
    with pytest.raises(IndexError, match="out of range"):
        reduce(tracked(fw, _pcf_grid()), dim=2)


@_SUMS
def test_norm_of_sum_gradient_wrt_pcf_values(fw, reduce):
    f, h = _f_and_g()
    times = np.array([0.0, 0.5, 2.0, 4.5, 8.0])

    def loss(c):
        g = sb.Pcf(fw.concat([c, fw.tensor([0.0])]), times=times)
        return fw.sum(sb.lp_norm(reduce(sb.PcfTensor([f, g, h]), dim=0), p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, [0.7, -1.3, 0.4, 2.0])


@_SUMS
def test_norm_of_sum_gradient_wrt_pcf_times(fw, reduce):
    f, h = _f_and_g()

    def loss(t):
        g = sb.Pcf(_VALUES, times=fw.concat([fw.tensor([0.0]), t]))
        return fw.sum(sb.lp_norm(reduce(sb.PcfTensor([f, g, h]), dim=0), p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, _TIMES)


def _positive_pcfs():
    # PCFs with positive values before their last one, 0.
    return sb.PcfTensor([
        sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 0.5], [4.0, 0.0]])),
        sb.Pcf(np.array([[0.0, 3.0], [0.5, 1.5], [2.5, 0.0]])),
    ])


@pytest.mark.parametrize("exponent", [2, 3.0, 0.5])
def test_power_of_tracked_pcfs_matches_untracked(fw, exponent):
    out = tracked(fw, _positive_pcfs()) ** exponent

    assert isinstance(out, sb.PcfTensor)
    assert isinstance(out.to_serial_content()[0], fw.tensor_type)
    _assert_same_functions(out, _positive_pcfs() ** exponent)


def test_power_of_tracked_pcf_matches_untracked(fw):
    f = _positive_pcfs()[0]

    out = tracked(fw, _positive_pcfs())[0] ** 2

    assert isinstance(out, sb.Pcf)
    t = np.linspace(0.0, 5.0, 51)
    np.testing.assert_allclose(fw.to_numpy(out(t)), (f ** 2)(t))


def test_power_of_pcfs_with_framework_exponent_matches_float_exponent(fw):
    out = _positive_pcfs() ** fw.tensor(1.5)

    _assert_same_functions(out, _positive_pcfs() ** 1.5)


def test_power_of_pcfs_with_one_framework_exponent_per_pcf(fw):
    F = _positive_pcfs()

    out = F ** fw.tensor([1.5, 2.0])

    _assert_same_functions(out, sb.PcfTensor([F[0] ** 1.5, F[1] ** 2.0]))


def test_gradient_through_square_of_pcfs(fw):
    F, G = _tensors()

    def loss(x):
        return fw.sum(sb.lp_norm((F * x - G) ** 2, p=1.0))

    _assert_gradient_matches_finite_differences(fw, loss, 1.5)


def test_gradient_wrt_exponent_of_pcfs(fw):
    def loss(a):
        return fw.sum(sb.lp_norm(_positive_pcfs() ** a, p=2.0))

    _assert_gradient_matches_finite_differences(fw, loss, 1.5)


@pytest.mark.parametrize("p", [1, 2.5])
def test_lp_distance_of_tracked_pcfs_matches_untracked(fw, p):
    F = _positive_pcfs()
    tracked_pcfs = tracked(fw, F)

    out = sb.lp_distance(tracked_pcfs[0], F[1], p=p)

    assert isinstance(out, fw.tensor_type)
    assert tuple(out.shape) == ()
    np.testing.assert_allclose(fw.to_numpy(out), sb.lp_distance(F[0], F[1], p=p))


def test_lp_distance_with_framework_p_matches_float_p(fw):
    f, g = _f_and_g()

    out = sb.lp_distance(f, g, p=fw.tensor(2.5))

    assert isinstance(out, fw.tensor_type)
    np.testing.assert_allclose(fw.to_numpy(out), sb.lp_distance(f, g, p=2.5))


@pytest.mark.parametrize("p", [1.0, 2.5])
def test_lp_distance_gradient_wrt_pcf_values(fw, p):
    f, _ = _f_and_g()
    times = np.array([0.0, 0.5, 2.0, 4.5, 8.0])

    def loss(c):
        g = sb.Pcf(fw.concat([c, fw.tensor([0.0])]), times=times)
        return sb.lp_distance(f, g, p=p) + sb.lp_distance(g, f * 2.0, p=p)

    _assert_gradient_matches_finite_differences(fw, loss, [0.7, -1.3, 0.4, 2.0])


def test_lp_distance_gradient_wrt_p(fw):
    f, g = _f_and_g()

    _assert_gradient_matches_finite_differences(fw, lambda p: sb.lp_distance(f, g, p=p), 2.5)


def test_lp_distance_with_framework_p_below_one_raises(fw):
    f, g = _f_and_g()
    with pytest.raises(ValueError, match=">= 1"):
        sb.lp_distance(f, g, p=fw.tensor(0.5))


_JOINABLE = pytest.mark.parametrize("make_tensor", [
    pytest.param(lambda: _pcf_grid(), id="pcfs"),
    pytest.param(lambda: _barcode_batch(), id="barcodes"),
    pytest.param(lambda: _point_clouds(), id="point_clouds"),
    pytest.param(lambda: _distance_matrices(), id="distance_matrices"),
])


def _assert_tracked_copy_of(fw, out, expected):
    assert isinstance(out, type(expected))
    assert isinstance(out.to_serial_content()[0], fw.tensor_type)
    assert out.shape == expected.shape
    assert out.array_equal(expected)


@_JOINABLE
@pytest.mark.parametrize("join", [
    pytest.param(lambda a, b: sb.concatenate([a, b], axis=0), id="concatenate"),
    pytest.param(lambda a, b: sb.concatenate([a, b[:, :1], a], axis=1), id="concatenate_three"),
    pytest.param(lambda a, b: sb.stack([a, b], axis=0), id="stack"),
    pytest.param(lambda a, b: sb.stack([a, b], axis=-1), id="stack_last"),
])
@pytest.mark.parametrize("which", ["first", "second", "both"])
def test_join_of_tracked_tensors_matches_untracked(fw, make_tensor, join, which):
    a, b = make_tensor(), make_tensor()[::-1]

    out = join(tracked(fw, a) if which != "second" else a, tracked(fw, b) if which != "first" else b)

    _assert_tracked_copy_of(fw, out, join(a, b))


@_JOINABLE
@pytest.mark.parametrize("divide", [
    pytest.param(lambda t: sb.split(t, 2, axis=1), id="split_sections"),
    pytest.param(lambda t: sb.split(t, [1], axis=0), id="split_indices"),
    pytest.param(lambda t: sb.array_split(t.flatten(), 3, axis=0), id="array_split"),
])
def test_split_of_tracked_tensor_matches_untracked(fw, make_tensor, divide):
    out = divide(tracked(fw, make_tensor()))

    expected = divide(make_tensor())
    assert len(out) == len(expected)
    for part, expected_part in zip(out, expected):
        _assert_tracked_copy_of(fw, part, expected_part)


def test_concatenation_of_tracked_tensors_of_different_types_raises(fw):
    with pytest.raises(TypeError, match="one type"):
        sb.concatenate([tracked(fw, _pcf_grid()), _barcode_batch()], axis=0)


def test_concatenation_of_tracked_tensors_of_different_shapes_raises(fw):
    with pytest.raises(ValueError):
        sb.concatenate([tracked(fw, _pcf_grid()), _pcf_grid().flatten()], axis=0)


def test_gradient_through_concatenation_and_split(fw):
    F, G = _tensors()

    def loss(x):
        both = sb.concatenate([F * x, G, G * x * x], axis=0)
        first, second = sb.split(sb.stack([both, both * x], axis=0), [2], axis=1)
        return fw.sum(sb.lp_norm(first, p=2.0)) + fw.sum(sb.lp_norm(second - G, p=1.0))

    _assert_gradient_matches_finite_differences(fw, loss, 1.5)


@pytest.mark.parametrize("p", [1, 2.5])
@pytest.mark.parametrize("which", ["X", "Y", "both"])
def test_cdist_of_tracked_pcfs_matches_untracked(fw, p, which):
    X, (F, _) = _pcf_grid(), _tensors()

    out = sb.cdist(tracked(fw, X) if which != "Y" else X, tracked(fw, F) if which != "X" else F, p=p, verbose=False)

    assert isinstance(out, fw.tensor_type)
    assert tuple(out.shape) == (2, 2, 3)
    np.testing.assert_allclose(fw.to_numpy(out), np.asarray(sb.cdist(X, F, p=p, verbose=False)))


def _cdist_weights():
    return np.array([[1.0, 2.0, -1.0],
                     [0.5, 3.0, 4.0]])


@pytest.mark.parametrize("p", [1.0, 2.5])
@pytest.mark.parametrize("side", ["X", "Y"])
def test_cdist_gradient_wrt_pcf_values(fw, p, side):
    f, g = _f_and_g()
    F, _ = _tensors()
    times = np.array([0.0, 0.5, 2.0, 4.5, 8.0])

    def loss(c):
        h = sb.PcfTensor([f, sb.Pcf(fw.concat([c, fw.tensor([0.0])]), times=times)])
        distances = sb.cdist(h, F, p=p, verbose=False) if side == "X" else sb.cdist(F, h, p=p, verbose=False).T
        return fw.sum(fw.tensor(_cdist_weights()) * distances)

    _assert_gradient_matches_finite_differences(fw, loss, [0.7, -1.3, 0.4, 2.0])


def test_cdist_gradient_wrt_pcf_times(fw):
    f, g = _f_and_g()
    F, _ = _tensors()

    def loss(t):
        h = sb.PcfTensor([f, sb.Pcf(_VALUES, times=fw.concat([fw.tensor([0.0]), t]))])
        return fw.sum(fw.tensor(_cdist_weights()) * sb.cdist(h, F, p=2.0, verbose=False))

    _assert_gradient_matches_finite_differences(fw, loss, _TIMES)


def _reordered_barcode_batch():
    # The barcodes of _barcode_batch with their bars in another order.
    Barcode = sb.persistence.Barcode
    return sb.persistence.BarcodeTensor([
        Barcode([[0.5, 3.7], [0.2, 1.5]]),
        Barcode(np.zeros((0, 2))),
        Barcode([[0.1, 0.4], [2.5, 4.0], [1.2, 2.1]]),
        Barcode([[1.1, 1.4]]),
    ]).reshape((2, 2))


@pytest.mark.parametrize("which", ["first", "second", "both"])
def test_tracked_barcode_tensor_is_isomorphic_to_its_values(fw, which):
    first = tracked(fw, _barcode_batch()) if which != "second" else _barcode_batch()
    second = tracked(fw, _reordered_barcode_batch()) if which != "first" else _reordered_barcode_batch()

    assert first.is_isomorphic_to(second, atol=0.0, rtol=0.0)
    assert not first.is_isomorphic_to(second[::-1], atol=0.0, rtol=0.0)


def test_barcode_is_isomorphic_to_the_values_of_a_tracked_barcode(fw):
    plain, reordered = _barcode_batch(), tracked(fw, _reordered_barcode_batch())

    assert plain[0, 0].is_isomorphic_to(reordered[0, 0], atol=0.0, rtol=0.0)
    assert not plain[0, 0].is_isomorphic_to(reordered[1, 0], atol=0.0, rtol=0.0)
