from .. import _sb_cpp as cpp
from ..async_task import _run_task
from ..functional.pcf import Pcf
from ..base_tensor import Tensor, _get_backend
from ..tensor_create import zeros
from ..typing import barcode32, barcode64, pcf32, pcf64
from .barcode import Barcode
from .ph_tensor import BarcodeTensor

cpp_p = cpp.persistence

_BACKEND_MAP = {
    barcode32: cpp_p.PersistenceBarcodeSummary32,
    barcode64: cpp_p.PersistenceBarcodeSummary64,
}

_BARCODE_TO_PCF_DTYPE = {barcode32: pcf32, barcode64: pcf64}


def _barcode_to_pcf(bc, single_method, task_method, verbose=False, **kwargs):
    """Shared implementation for barcode-to-PCF conversions."""

    backend, X = _get_backend(bc, _BACKEND_MAP)

    if isinstance(X, Barcode):
        return Pcf(getattr(backend, single_method)(X._data, **kwargs))
    elif isinstance(X, Tensor):
        pcf_dtype = _BARCODE_TO_PCF_DTYPE[X.dtype]
        out = zeros((1,), dtype=pcf_dtype)

        _run_task(
            lambda: getattr(backend, task_method)(X._data, out._data, **kwargs),
            verbose=verbose,
        )

        # The backend writes a result of the same shape as the input tensor;
        # the single-Barcode convenience case is handled by the branch above,
        # so the batch shape (including a genuine leading-1 axis) is preserved.
        return out


def barcode_to_stable_rank(
    bc: Barcode | BarcodeTensor, verbose=False
):
    r"""Convert barcodes to stable rank functions.

    The stable rank of a barcode is the PCF that counts, for each
    :math:`t \geq 0`, the number of bars with length (death minus birth)
    strictly greater than :math:`t` :footcite:`Chacholski2020`.

    Parameters
    ----------
    bc : Barcode or BarcodeTensor
        A single barcode or a tensor of barcodes.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Pcf or PcfTensor
        A single ``Pcf`` if the input is a single ``Barcode``, otherwise a
        ``PcfTensor`` with the same shape as the input.
    """
    return _barcode_to_pcf(
        bc, "barcode_to_stable_rank", "spawn_barcode_to_stable_rank_task", verbose
    )


def barcode_to_betti_curve(
    bc: Barcode | BarcodeTensor, verbose=False
):
    r"""Convert barcodes to Betti curves.

    The Betti curve is the PCF that counts, for each filtration value
    :math:`t \geq 0`, the number of bars alive at :math:`t`
    (i.e., bars with birth :math:`\leq t <` death) :footcite:`Umeda2017,Chazal2021`.

    Parameters
    ----------
    bc : Barcode or BarcodeTensor
        A single barcode or a tensor of barcodes.
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Pcf or PcfTensor
        A single ``Pcf`` if the input is a single ``Barcode``, otherwise a
        ``PcfTensor`` with the same shape as the input.
    """
    return _barcode_to_pcf(
        bc, "barcode_to_betti_curve", "spawn_barcode_to_betti_curve_task", verbose
    )


def barcode_to_accumulated_persistence(
    bc: Barcode | BarcodeTensor,
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
    bc : Barcode or BarcodeTensor
        A single barcode or a tensor of barcodes.
    max_death : float, optional
        If finite, exclude bars whose death time exceeds this value.
        By default ``inf`` (all finite bars included).
    verbose : bool, optional
        Show progress information, by default False.

    Returns
    -------
    Pcf or PcfTensor
        A single ``Pcf`` if the input is a single ``Barcode``, otherwise a
        ``PcfTensor`` with the same shape as the input.
    """
    return _barcode_to_pcf(
        bc, "barcode_to_accumulated_persistence",
        "spawn_barcode_to_accumulated_persistence_task", verbose,
        max_death=max_death
    )
