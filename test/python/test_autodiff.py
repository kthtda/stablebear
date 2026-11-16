import numpy as np
import pytest

import stablebear as sb
from stablebear import _autodiff

from _frameworks import assert_gradient_matches_finite_differences as _assert_gradient_matches_finite_differences
from _frameworks import tracked
from _round_trip import ROUND_TRIPS


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
    np.testing.assert_allclose(fw.to_numpy(h._tvalues), expected[:, 1])


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
    np.testing.assert_allclose(fw.to_numpy(out._tvalues), expected)


@pytest.mark.parametrize("wrt", ["X", "Y"])
def test_homological_kernel_gradient(fw, wrt):
    X, Y = _kernel_points()
    weights = fw.tensor([[0.5, 1.0], [-1.0, 2.0], [1.5, -0.5], [2.0, 0.25], [-0.75, 1.25]])

    def loss(P):
        Xp, Yp = (P, Y) if wrt == "X" else (X, P)
        return fw.sum(weights * sb.persistence.compute_homological_kernel(Xp, Yp)._tvalues)

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
    np.testing.assert_allclose(fw.to_numpy(out._tvalues), expected)


def test_transform_bars_removes_empty_bars_of_framework_kernel(fw):
    X, Y = _kernel_points()
    f = sb.Plf([[0.0, 0.0], [1.0, 1.0], [2.3, 1.0], [5.0, 2.0]])
    out = sb.persistence.transform_bars(sb.persistence.compute_homological_kernel(fw.tensor(X), fw.tensor(Y)), f)
    expected = sb.persistence.transform_bars(sb.persistence.compute_homological_kernel(X, Y), f)[0].to_numpy()
    assert len(expected) == 3
    np.testing.assert_allclose(fw.to_numpy(out._tvalues), expected)


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
        return fw.sum(weights * sb.persistence.transform_bars(barcodes, f)._tvalues)

    _assert_gradient_matches_finite_differences(fw, loss, Y)


@pytest.mark.parametrize("wrt", ["X", "Y"])
def test_transform_bars_gradient(fw, wrt):
    X, Y = _kernel_points()
    weights = fw.tensor([[0.5, 1.0], [-1.0, 2.0], [1.5, -0.5], [2.0, 0.25], [-0.75, 1.25]])

    def loss(P):
        Xp, Yp = (P, Y) if wrt == "X" else (X, P)
        barcodes = sb.persistence.compute_homological_kernel(Xp, Yp)
        return fw.sum(weights * sb.persistence.transform_bars(barcodes, _bar_plf())._tvalues)

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
        return fw.sum(_plf_bar_weights(fw) * sb.persistence.transform_bars(_plf_bars(), f)._tvalues)

    _assert_gradient_matches_finite_differences(fw, loss, [0.0, 1.0, 1.5, 3.0])


def test_transform_bars_gradient_wrt_plf_times(fw):
    def loss(t):
        f = sb.Plf(np.array([0.0, 1.0, 1.5, 3.0]), times=fw.concat([fw.tensor([0.0]), t]))
        return fw.sum(_plf_bar_weights(fw) * sb.persistence.transform_bars(_plf_bars(), f)._tvalues)

    _assert_gradient_matches_finite_differences(fw, loss, [1.0, 2.0, 3.0])


def test_transform_bars_gradient_wrt_plf_and_points(fw):
    X, Y = _kernel_points()

    def loss(x):
        barcodes = sb.persistence.compute_homological_kernel(fw.tensor(X), x * fw.tensor(Y))
        f = sb.Plf(x * fw.tensor([0.0, 0.5, 3.0, 3.5]), times=np.array([0.0, 1.3, 2.7, 4.1]))
        return fw.sum(sb.persistence.transform_bars(barcodes, f)._tvalues)

    _assert_gradient_matches_finite_differences(fw, loss, 0.9)


def test_transform_bars_gradient_wrt_plf_at_infinite_death(fw):
    bc = sb.persistence.BarcodeTensor([sb.persistence.Barcode([[0.5, np.inf]])])
    # f ends flat, so f(inf) is its last value.
    _, grad = fw.value_and_grad(
        lambda v: fw.sum(sb.persistence.transform_bars(bc, sb.Plf(v, times=np.array([0.0, 1.0, 2.0])))._tvalues),
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
    np.testing.assert_allclose(fw.to_numpy(out._tvalues), expected._serial()[0])


def test_transform_bars_of_batch_gradient_wrt_plf_values(fw):
    # f is flat on [1, 2], so the bar (1.1, 1.4) is removed.
    weights = fw.tensor([[1.0, 2.0], [-1.0, 0.5], [0.3, 0.7], [0.2, 0.9], [-0.4, 1.1]])

    def loss(v):
        f = sb.Plf(fw.concat([v[:2], v[1:2], v[2:]]), times=_batch_plf_times())
        return fw.sum(weights * sb.persistence.transform_bars(_barcode_batch(), f)._tvalues)

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
        assert selected._tvalues is not None
        np.testing.assert_array_equal(fw.to_numpy(selected._tvalues), plain._serial()[0])
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
        out._tvalues.mul_(2)

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
        assert selected._ttimes is not None and selected._tvalues is not None
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
        sb.cdist(out.flatten(), out.flatten(), p=1, verbose=False)
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
    adapter = _autodiff._framework(out._tvalues)
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

    np.testing.assert_array_equal(fw.to_numpy(out._tvalues), [[0.0, 10.0],
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
    assert out._tvalues is not None
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
        return fw.sum(out._ttimes * out._tvalues)

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
    assert out._tvalues is not None
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
        bars = barcodes[dim:dim + 1]._tvalues
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
    assert out._tvalues is not None
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
        assert selected._tvalues is not None
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
        assert selected._tvalues is not None
        assert selected.array_equal(expected)
        for i in np.ndindex(*expected.shape):
            np.testing.assert_array_equal(selected[i].to_dense(), expected[i].to_dense())


@pytest.mark.parametrize("make_matrices", [_distance_matrices, _symmetric_matrices], ids=["distance", "symmetric"])
def test_write_to_tracked_matrices_raises(fw, make_matrices):
    out = tracked(fw, make_matrices())
    with pytest.raises(NotImplementedError, match="Writing"):
        out[0, 0] = make_matrices()[1, 1]


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


_FUNCTIONS = pytest.mark.parametrize("function", [sb.Pcf, sb.Plf], ids=["pcf", "plf"])


def _tracked_function(fw, function):
    return function(fw.tensor([10.0, 20.0]), times=np.array([0.0, 1.0]))


@_FUNCTIONS
@pytest.mark.parametrize("operation", [
    pytest.param(lambda function, f: f(0.5), id="evaluate"),
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
