from .. import _sb_cpp as cpp
from ..distance_matrix import DistanceMatrixTensor
from ..base_tensor import PointCloudTensor, _get_backend
from ..typing import distmat32, distmat64, pcloud32, pcloud64
from .ph_tensor import BarcodeTensor

cpp_p = cpp.persistence


def _compute_barcodes_euclidean_pcloud_ripser(
    X: PointCloudTensor,
    out: BarcodeTensor,
    max_dim: int = 1,
    reduced_homology: bool = False,
):
    backend, X = _get_backend(
        X, {pcloud32: cpp_p.PersistenceRipser32, pcloud64: cpp_p.PersistenceRipser64}
    )

    return backend.spawn_ripser_pcloud_euclidean_task(X._data, out._data, max_dim, reduced_homology)


def _compute_barcodes_distmat_ripser(
    X: DistanceMatrixTensor,
    out: BarcodeTensor,
    max_dim: int = 1,
    reduced_homology: bool = False,
):
    backend, X = _get_backend(
        X, {distmat32: cpp_p.PersistenceRipser32, distmat64: cpp_p.PersistenceRipser64}
    )

    return backend.spawn_ripser_distmat_task(X._data, out._data, max_dim, reduced_homology)
