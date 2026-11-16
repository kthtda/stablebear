from typing import overload

import numpy as np

from .. import _autodiff
from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..base_tensor import PcfTensor, _get_backend
from ..functional.pcf import Pcf
from ..functional.plf import Plf
from ..timeseries.series import TimeSeries
from ..typing import barcode32, barcode64, float32, float64
from .barcode import Barcode
from .ph_tensor import BarcodeTensor

cpp_p = cpp.persistence

_BACKEND_MAP = {
    barcode32: cpp_p.PersistenceBarcodeSummary32,
    barcode64: cpp_p.PersistenceBarcodeSummary64,
}


def _barcode_to_pcf(bc, task_method, verbose=False, **kwargs):
    """Shared implementation for barcode-to-PCF conversions."""
    if isinstance(bc, TimeSeries):
        if bc.dtype not in _BACKEND_MAP:
            raise TypeError(f"Expected a time series of barcodes, got {bc.dtype}")
        task = getattr(_BACKEND_MAP[bc.dtype], task_method)(bc._data, **kwargs)
        _run_task(lambda: task, verbose=verbose)
        return TimeSeries._from_cpp(task.result())

    backend, X = _get_backend(bc, _BACKEND_MAP)

    task = getattr(backend, task_method)(X._data, **kwargs)
    _run_task(lambda: task, verbose=verbose)
    out = PcfTensor(task.result())

    if isinstance(X, Barcode):
        # A single barcode gives a 0-d result holding its PCF.
        return out[()]

    if len(out.shape) == 2 and out.shape[0] == 1:
        out = out[0, :]

    return out


def _tracked_stable_rank(bc: BarcodeTensor) -> PcfTensor:
    """The stable rank of a BarcodeTensor whose bars are a framework tensor.

    With n bars, the stable rank is n on [0, l_1), then n - 1 on [l_1, l_2), and
    so on down to 0 after l_n, for the sorted lifetimes l_1 <= ... <= l_n. So its
    times are framework operations on the bars, which the framework
    differentiates (sort included), and its values are constant. Equal
    lifetimes give repeated times, which describe the same function.
    """
    bars = bc._tvalues
    adapter = _autodiff._framework(bars)
    n = int(bars.shape[0])
    lifetimes = bars[:, 1] - bars[:, 0]
    times = adapter.concat([adapter.as_tensor(np.zeros(1), like=bars), adapter.sort(lifetimes)])
    content = np.column_stack([
        _autodiff._concrete_column(times, n + 1, adapter), np.arange(n, -1, -1)
    ]).astype(np.float32 if str(bars.dtype).endswith("float32") else np.float64)
    out = _autodiff._pcfs_from_content(content, np.array([[0, n + 1]]))
    out._ttimes = times
    return out


@overload
def barcode_to_stable_rank(bc: Barcode, verbose: bool = False) -> Pcf: ...


@overload
def barcode_to_stable_rank(bc: BarcodeTensor, verbose: bool = False) -> PcfTensor: ...


@overload
def barcode_to_stable_rank(bc: TimeSeries, verbose: bool = False) -> TimeSeries: ...


def barcode_to_stable_rank(
    bc: Barcode | BarcodeTensor | TimeSeries, verbose=False
):
    r"""Convert barcodes to stable rank functions.

    The stable rank of a barcode is the PCF that counts, for each
    :math:`t \geq 0`, the number of bars with length (death minus birth)
    strictly greater than :math:`t` :footcite:`Chacholski2020`.

    Parameters
    ----------
    bc : Barcode, BarcodeTensor, or TimeSeries
        A single barcode or a container of barcodes.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Pcf, PcfTensor, or TimeSeries
        A single ``Pcf`` if the input is a single ``Barcode``, otherwise the
        same kind of container as the input, with the same shape.
    """
    if isinstance(bc, BarcodeTensor) and bc._tvalues is not None:
        return _tracked_stable_rank(bc)
    return _barcode_to_pcf(
        bc, "spawn_barcode_to_stable_rank_task", verbose
    )


@overload
def barcode_to_betti_curve(bc: Barcode, verbose: bool = False) -> Pcf: ...


@overload
def barcode_to_betti_curve(bc: BarcodeTensor, verbose: bool = False) -> PcfTensor: ...


@overload
def barcode_to_betti_curve(bc: TimeSeries, verbose: bool = False) -> TimeSeries: ...


def barcode_to_betti_curve(
    bc: Barcode | BarcodeTensor | TimeSeries, verbose=False
):
    r"""Convert barcodes to Betti curves.

    The Betti curve is the PCF that counts, for each filtration value
    :math:`t \geq 0`, the number of bars alive at :math:`t`
    (i.e., bars with birth :math:`\leq t <` death) :footcite:`Umeda2017,Chazal2021`.

    Parameters
    ----------
    bc : Barcode, BarcodeTensor, or TimeSeries
        A single barcode or a container of barcodes.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Pcf, PcfTensor, or TimeSeries
        A single ``Pcf`` if the input is a single ``Barcode``, otherwise the
        same kind of container as the input, with the same shape.
    """
    return _barcode_to_pcf(
        bc, "spawn_barcode_to_betti_curve_task", verbose
    )


@overload
def barcode_to_accumulated_persistence(bc: Barcode, max_death: float = float("inf"), verbose: bool = False) -> Pcf: ...


@overload
def barcode_to_accumulated_persistence(bc: BarcodeTensor, max_death: float = float("inf"), verbose: bool = False) -> PcfTensor: ...


@overload
def barcode_to_accumulated_persistence(bc: TimeSeries, max_death: float = float("inf"), verbose: bool = False) -> TimeSeries: ...


def barcode_to_accumulated_persistence(
    bc: Barcode | BarcodeTensor | TimeSeries,
    max_death: float = float("inf"),
    verbose: bool = False,
):
    r"""Convert barcodes to accumulated persistence functions.

    The accumulated persistence function (APF) is defined as

    .. math::

       \mathrm{APF}(t) = \sum_{i=1}^{N} \ell_i \, \mathbf{1}_{m_i \leq t}

    where :math:`N` is the number of bars, :math:`\ell_i = d_i - b_i` is the
    lifetime of bar :math:`i`, and :math:`m_i = (b_i + d_i) / 2` is its
    midpoint :footcite:`Biscio2019`.

    When ``max_death`` is finite, only bars with :math:`d_i \leq`
    ``max_death`` are included (Equation 2 in the paper).

    Parameters
    ----------
    bc : Barcode, BarcodeTensor, or TimeSeries
        A single barcode or a container of barcodes.
    max_death : float, optional
        If finite, exclude bars whose death time exceeds this value.
        By default ``inf`` (all finite bars included).
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Pcf, PcfTensor, or TimeSeries
        A single ``Pcf`` if the input is a single ``Barcode``, otherwise the
        same kind of container as the input, with the same shape.
    """
    return _barcode_to_pcf(
        bc, "spawn_barcode_to_accumulated_persistence_task", verbose,
        max_death=max_death
    )


def _plf_slopes(f: Plf, t: np.ndarray) -> np.ndarray:
    """The slope of f at each time in t: that of the segment starting at or
    before it, the last segment continued after the last breakpoint, and 0 at
    infinite times, where f takes its limit."""
    pts = f.to_numpy().astype(np.float64)
    if len(pts) == 1:
        return np.zeros(t.shape)
    segment = np.diff(pts[:, 1]) / np.diff(pts[:, 0])
    segment = np.append(segment, segment[-1])
    slopes = segment[np.searchsorted(pts[:, 0], t, side="right") - 1]
    return np.where(np.isinf(t), 0.0, slopes)


def _transformed_bars_vjp(grad_out, bc, f):
    return (grad_out * _plf_slopes(f, np.asarray(bc[0], dtype=np.float64)),)


def _transformed_bars_out(bc, f):
    return tuple(bc._tvalues.shape), np.float32 if bc.dtype == barcode32 else np.float64


@_autodiff.differentiable("bc", vjp=_transformed_bars_vjp, out=_transformed_bars_out)
def _transformed_bars(bc: BarcodeTensor, f: Plf) -> np.ndarray:
    """The (m, 2) transformed bars of a (1,) BarcodeTensor, empty ones included."""
    return np.asarray(_transform_bars(bc, f, verbose=False, keep_empty=True)[0])


def _tracked_transform_bars(bc: BarcodeTensor, f: Plf) -> BarcodeTensor:
    """transform_bars of a BarcodeTensor whose bars are a framework tensor."""
    bars = _transformed_bars(bc, f)
    adapter = _autodiff._framework(bars)
    concrete = adapter.concrete(bars)
    if concrete is None:
        raise NotImplementedError(
            "transform_bars of bars that JAX or tf.function traces abstractly is not supported: "
            "which bars it removes depends on their values.")
    keep = np.flatnonzero(concrete[:, 0] < concrete[:, 1])
    return BarcodeTensor._from_tracked_bars(adapter.gather(bars, keep))


@overload
def transform_bars(bc: Barcode, f: Plf, verbose: bool = False) -> Barcode: ...


@overload
def transform_bars(bc: BarcodeTensor, f: Plf, verbose: bool = False) -> BarcodeTensor: ...


@overload
def transform_bars(bc: TimeSeries, f: Plf, verbose: bool = False) -> TimeSeries: ...


def transform_bars(bc: Barcode | BarcodeTensor | TimeSeries, f: Plf, verbose: bool = False):
    r"""Transform the endpoints of every bar by a PLF.

    Each bar :math:`(b, d)` becomes :math:`(f(b), f(d))`. An infinite
    endpoint becomes the limit of :math:`f`: :math:`\infty` or
    :math:`-\infty` if the last segment of :math:`f` rises or falls, or its
    last value if the segment is flat. Bars with :math:`f(b) \geq f(d)`,
    which :math:`f` can give where it is flat or falls, are removed.

    If the bars carry gradients, as those of
    :func:`~stablebear.persistence.homological_kernel.compute_homological_kernel` of PyTorch, JAX, or TensorFlow
    arrays do, so do the transformed bars. At a breakpoint of :math:`f`,
    the derivative is the slope of the segment after it. Which bars are
    removed depends on their values, which JAX and ``tf.function`` do not
    provide while they trace, so there this raises ``NotImplementedError``.

    Parameters
    ----------
    bc : Barcode, BarcodeTensor, or TimeSeries
        A single barcode or a container of barcodes.
    f : Plf
        The function to apply to the endpoints. It is converted to the
        precision of the barcodes.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Barcode, BarcodeTensor, or TimeSeries
        The same kind of object as `bc`, with the same shape, less the
        removed bars.

    Raises
    ------
    ValueError
        If an endpoint is negative (a PLF is defined on
        :math:`[0, \infty)`).
    NotImplementedError
        If the bars carry gradients that JAX or ``tf.function`` traces.

    Examples
    --------
    >>> import stablebear as sb
    >>> bc = sb.persistence.Barcode([[0.0, 1.0], [0.5, float("inf")]])
    >>> f = sb.Plf([[0.0, 0.0], [1.0, 2.0]])
    >>> sb.persistence.transform_bars(bc, f).to_numpy()
    array([[ 0.,  2.],
           [ 1., inf]])
    """
    if not isinstance(f, Plf):
        raise TypeError(f"Expected a Plf, got {type(f).__name__}.")
    if isinstance(bc, BarcodeTensor) and bc._tvalues is not None:
        return _tracked_transform_bars(bc, f)
    return _transform_bars(bc, f, verbose=verbose, keep_empty=False)


def _transform_bars(bc, f, verbose, keep_empty):
    if isinstance(bc, TimeSeries):
        if bc.dtype not in _BACKEND_MAP:
            raise TypeError(f"Expected a time series of barcodes, got {bc.dtype}")
        backend, data = _BACKEND_MAP[bc.dtype], bc._data
    else:
        backend, X = _get_backend(bc, _BACKEND_MAP)
        data = X._data
    f = f.astype(float32 if bc.dtype == barcode32 else float64)

    task = backend.spawn_transform_bars_task(data, f._data, keep_empty=keep_empty)
    _run_task(lambda: task, verbose=verbose)
    if isinstance(bc, TimeSeries):
        return TimeSeries._from_cpp(task.result())
    if isinstance(bc, Barcode):
        return BarcodeTensor(task.result())[()]
    return BarcodeTensor(task.result())
