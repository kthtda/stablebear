#include "py_ripser.hpp"
#include "../py_async_support.hpp"
#include "../py_tensor_property_variants.hpp"

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
    template <sb::TensorProperties Properties>
    static std::unique_ptr<sb::StoppableTask<void>> spawn_ripser_pcloud_euclidean_task(
        const sb::Tensor<sb::PointCloud<T>, Properties>& pclouds,
        sb::Tensor<sb::ph::Barcode<T>>& out, size_t maxDim,
        bool reducedHomology)
    {
      return sb_py::execute_stoppable_task<sb::ph::RipserTask<T, Properties>>(
          pclouds, out, maxDim, reducedHomology);
    }

    static std::unique_ptr<sb::StoppableTask<void>> spawn_ripser_single_pcloud_euclidean_task(const sb::PointCloud<T>& pointCloud, sb::Tensor<sb::ph::Barcode<T>>& out, size_t maxDim, bool reducedHomology)
    {
      sb::Tensor<sb::PointCloud<T>> pointClouds({1});
      pointClouds(0) = pointCloud.copy();
      return sb_py::execute_stoppable_task<sb::ph::RipserTask<T>>(
        std::move(pointClouds), out, maxDim, reducedHomology);
    }

    template <sb::TensorProperties Properties>
    static std::unique_ptr<sb::StoppableTask<void>> spawn_ripser_distmat_task(
        const sb::Tensor<sb::DistanceMatrix<T>, Properties>& dmats,
        sb::Tensor<sb::ph::Barcode<T>>& out,
        size_t maxDim, bool reducedHomology)
    {
      return sb_py::execute_stoppable_task<
        sb::ph::RipserDistMatTask<T, Properties>>(
        dmats, out, maxDim, reducedHomology);
    }

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PyRipserBindings> cls(
        m, ("PersistenceRipser" + suffix).c_str());
      sb_py::bind_tensor_property_variants(
        [&]<sb::TensorProperties Properties>() {
          cls.def_static(
            "spawn_ripser_pcloud_euclidean_task",
            &PyRipserBindings::template spawn_ripser_pcloud_euclidean_task<Properties>);
          cls.def_static(
            "spawn_ripser_distmat_task",
            &PyRipserBindings::template spawn_ripser_distmat_task<Properties>);
        });
      cls.def_static(
          "spawn_ripser_pcloud_euclidean_task",
          &PyRipserBindings::spawn_ripser_single_pcloud_euclidean_task);
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
