#include "py_ripser.hpp"
#include "../py_async_support.hpp"
#include "../py_ufunc_containers.hpp"

#include <sbear/tensor.hpp>
#include <sbear/distance_matrix.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/compute_persistence.hpp>
#include <sbear/timeseries.hpp>

#include <type_traits>

namespace nb = nanobind;

namespace
{

  template <typename T>
  class PyRipserBindings
  {
  public:
    /// Persistence of every element of @p input, a tensor or a time series;
    /// see make_persistence_task.
    template <typename Container>
    static auto spawn_persistence_task(const Container& input, size_t maxDim, bool reducedHomology)
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

    /// Bind one spawn function, overloaded on every container of ElemT and,
    /// for clouds and matrices, on a single element, with its task classes.
    template <typename ElemT>
    static void register_spawn(nb::module_& m, nb::class_<PyRipserBindings>& cls,
                               const std::string& spawnName, const std::string& taskName)
    {
      sb_py::bind_ufunc_containers<ElemT>([&]<typename Container>(const std::string& variant) {
        sb_py::register_result_task<sb::ph::PersistenceTask<Container>>(m, taskName + variant);
        cls.def_static(spawnName.c_str(), &PyRipserBindings::template spawn_persistence_task<Container>);
      });
      if constexpr (!std::is_floating_point_v<ElemT>)
      {
        cls.def_static(spawnName.c_str(), &PyRipserBindings::template spawn_single_persistence_task<ElemT>);
      }
    }

    static void register_bindings(nb::module_& m, const std::string& suffix)
    {
      nb::class_<PyRipserBindings> cls(
        m, ("PersistenceRipser" + suffix).c_str());
      register_spawn<sb::PointCloud<T>>(
          m, cls, "spawn_ripser_pcloud_euclidean_task", "_PersistenceTaskPointCloud" + suffix);
      register_spawn<sb::DistanceMatrix<T>>(
          m, cls, "spawn_ripser_distmat_task", "_PersistenceTaskDistanceMatrix" + suffix);
      register_spawn<T>(
          m, cls, "spawn_ripser_coordinates_euclidean_task", "_PersistenceTaskCoordinates" + suffix);
    }
  };

}

namespace sb_py
{
  void register_persistence_ripser(nb::module_ &m)
  {
    PyRipserBindings<sb::float32_t>::register_bindings(m, "32");
    PyRipserBindings<sb::float64_t>::register_bindings(m, "64");
  }
}
