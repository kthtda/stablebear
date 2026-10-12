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
    template <typename ElemT, sb::TensorProperties Properties>
    static auto spawn_persistence_task(
        const sb::Tensor<ElemT, Properties>& input, size_t maxDim, bool reducedHomology)
    {
      auto task = sb::ph::make_persistence_task(input, maxDim, reducedHomology);
      task->start_async(sb::default_executor());
      return task;
    }

    /// A single cloud or matrix runs as a 0-d tensor, so the result has shape
    /// (maxDim + 1,).
    template <typename ElemT>
    static auto spawn_single_persistence_task(
        const ElemT& element, size_t maxDim, bool reducedHomology)
    {
      return spawn_persistence_task(
          sb::Tensor<ElemT>(std::vector<size_t>{}, element), maxDim, reducedHomology);
    }

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PyRipserBindings> cls(
        m, ("PersistenceRipser" + suffix).c_str());
      sb_py::bind_tensor_property_variants(
        [&]<sb::TensorProperties Properties>() {
          const std::string variant = Properties == sb::TensorProperty::Indexed ? "Indexed" : "";
          sb_py::register_result_task<sb::ph::PersistenceTask<sb::PointCloud<T>, T, Properties>>(
            m, "_PersistenceTaskPointCloud" + suffix + variant);
          sb_py::register_result_task<sb::ph::PersistenceTask<sb::DistanceMatrix<T>, T, Properties>>(
            m, "_PersistenceTaskDistanceMatrix" + suffix + variant);
          cls.def_static(
            "spawn_ripser_pcloud_euclidean_task",
            &PyRipserBindings::template spawn_persistence_task<sb::PointCloud<T>, Properties>);
          cls.def_static(
            "spawn_ripser_distmat_task",
            &PyRipserBindings::template spawn_persistence_task<sb::DistanceMatrix<T>, Properties>);
        });
      cls.def_static(
          "spawn_ripser_pcloud_euclidean_task",
          &PyRipserBindings::template spawn_single_persistence_task<sb::PointCloud<T>>);
      cls.def_static(
          "spawn_ripser_distmat_task",
          &PyRipserBindings::template spawn_single_persistence_task<sb::DistanceMatrix<T>>);
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
