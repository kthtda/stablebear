import numpy as np

from .. import _sb_cpp as cpp
from ..tensor_create import zeros
from ..typing import _validate_dtype, pcloud32, pcloud64

cpp_pp = cpp.point_process


def _get_backend(dtype):
    dtype = _validate_dtype(dtype, [pcloud32, pcloud64])

    if dtype == pcloud32:
        return cpp_pp.Poisson32
    elif dtype == pcloud64:
        return cpp_pp.Poisson64


def sample_poisson(shape, dim=2, rate=1.0, lo=None, hi=None, generator=None, dtype=pcloud64):
    r"""Sample a homogeneous spatial Poisson point process.

    Each element of the returned tensor is a point cloud of shape
    :math:`(N_i, \text{dim})` where :math:`N_i \sim \text{Poisson}(\lambda V)`
    and points are drawn uniformly in the hyperrectangle :math:`[\text{lo}, \text{hi}]`.
    Here :math:`V` is the volume of the region.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output tensor.
    dim : int, optional
        Spatial dimension, by default 2.
    rate : float, optional
        Intensity :math:`\lambda` of the Poisson process, by default 1.0.
    lo : array_like, optional
        Lower bounds per spatial dimension. Default is zeros.
    hi : array_like, optional
        Upper bounds per spatial dimension. Default is ones.
    generator : Generator, optional
        Random number generator. If ``None``, the global generator is used.
    dtype : type, optional
        ``pcloud32`` or ``pcloud64``, by default ``pcloud64``.

    Returns
    -------
    PointCloudTensor
        Tensor of point clouds with the given shape.
    """
    backend = _get_backend(dtype)

    float_dtype = np.float32 if dtype == pcloud32 else np.float64

    if lo is None:
        lo = np.zeros(dim, dtype=float_dtype)
    else:
        lo = np.asarray(lo, dtype=float_dtype)

    if hi is None:
        hi = np.ones(dim, dtype=float_dtype)
    else:
        hi = np.asarray(hi, dtype=float_dtype)

    if lo.shape != (dim,) or hi.shape != (dim,):
        raise ValueError(f"lo and hi must have shape ({dim},)")

    from ..random import _unwrap

    A = zeros(shape, dtype=dtype)
    backend.sample_poisson(
        A._data, dim, float_dtype(rate), lo.tolist(), hi.tolist(),
        _unwrap(generator))

    return A
