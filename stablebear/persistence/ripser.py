from .. import _sb_cpp as cpp
from ..distance_matrix import DistanceMatrix, DistanceMatrixTensor
from ..base_tensor import _get_backend
from ..point_cloud import PointCloud, PointCloudTensor
from ..typing import distmat32, distmat64, float32, float64, pcloud32, pcloud64

cpp_p = cpp.persistence


def _compute_barcodes_euclidean_pcloud_ripser(
    X: PointCloud | PointCloudTensor,
    max_dim: int = 1,
    reduced_homology: bool = False,
):
    backend, X = _get_backend(
        X,
        {
            float32: cpp_p.PersistenceRipser32,
            float64: cpp_p.PersistenceRipser64,
            pcloud32: cpp_p.PersistenceRipser32,
            pcloud64: cpp_p.PersistenceRipser64,
        },
    )
    data = X._cpp_point_cloud() if isinstance(X, PointCloud) else X._data

    return backend.spawn_ripser_pcloud_euclidean_task(
        data, max_dim, reduced_homology
    )


def _compute_barcodes_distmat_ripser(
    X: DistanceMatrix | DistanceMatrixTensor,
    max_dim: int = 1,
    reduced_homology: bool = False,
):
    backend, X = _get_backend(
        X,
        {
            float32: cpp_p.PersistenceRipser32,
            float64: cpp_p.PersistenceRipser64,
            distmat32: cpp_p.PersistenceRipser32,
            distmat64: cpp_p.PersistenceRipser64,
        },
    )

    data = X._current_matrix() if isinstance(X, DistanceMatrix) else X._data

    return backend.spawn_ripser_distmat_task(data, max_dim, reduced_homology)
