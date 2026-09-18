#include "py_ripser.hpp"
#include "../py_async_support.hpp"

#include <sbear/tensor.hpp>
#include <sbear/distance_matrix.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/compute_persistence.hpp>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PyRipserBindings
  {
  public:
    static std::unique_ptr<sb::StoppableTask<void>> spawn_ripser_pcloud_euclidean_task(const sb::Tensor<sb::PointCloud<T>>& pclouds, sb::Tensor<sb::ph::Barcode<T>>& out, size_t maxDim, bool reducedHomology)
    {
      return sb_py::execute_stoppable_task<sb::ph::RipserTask<T>>(pclouds, out, maxDim, reducedHomology);
    }

    static std::unique_ptr<sb::StoppableTask<void>> spawn_ripser_pcloud_euclidean_task(const sb::PointCloud<T>& pointCloud, sb::Tensor<sb::ph::Barcode<T>>& out, size_t maxDim, bool reducedHomology)
    {
      sb::Tensor<sb::PointCloud<T>> pointClouds({1});
      pointClouds(0) = pointCloud.copy();
      return sb_py::execute_stoppable_task<sb::ph::RipserTask<T>>(
        std::move(pointClouds), out, maxDim, reducedHomology);
    }

    static std::unique_ptr<sb::StoppableTask<void>> spawn_ripser_distmat_task(const sb::Tensor<sb::DistanceMatrix<T>>& dmats, sb::Tensor<sb::ph::Barcode<T>>& out, size_t maxDim, bool reducedHomology)
    {
      return sb_py::execute_stoppable_task<sb::ph::RipserDistMatTask<T>>(dmats, out, maxDim, reducedHomology);
    }

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PyRipserBindings>(m, ("PersistenceRipser" + suffix).c_str())
        .def_static(
          "spawn_ripser_pcloud_euclidean_task",
          py::overload_cast<const sb::Tensor<sb::PointCloud<T>>&, sb::Tensor<sb::ph::Barcode<T>>&, size_t, bool>(
            &PyRipserBindings::spawn_ripser_pcloud_euclidean_task))
        .def_static(
          "spawn_ripser_pcloud_euclidean_task",
          py::overload_cast<const sb::PointCloud<T>&, sb::Tensor<sb::ph::Barcode<T>>&, size_t, bool>(
            &PyRipserBindings::spawn_ripser_pcloud_euclidean_task))
        .def_static("spawn_ripser_distmat_task", &PyRipserBindings::spawn_ripser_distmat_task)
      ;
    }
  };

}

namespace sb_py
{
  void register_persistence_ripser(pybind11::module_ &m)
  {
    PyRipserBindings<sb::float32_t>::register_bindings(m, "32");
    PyRipserBindings<sb::float64_t>::register_bindings(m, "64");
  }
}
