from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..base_tensor import PcfTensor, _get_backend
from ..timeseries.series import TimeSeries
from ..typing import barcode32, barcode64
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
    return _barcode_to_pcf(
        bc, "spawn_barcode_to_stable_rank_task", verbose
    )


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
