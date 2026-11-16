#include "py_ripser.hpp"
#include "../py_async_support.hpp"
#include "../py_ufunc_containers.hpp"
#include "../py_np_support.hpp"

#include <sbear/tensor.hpp>
#include <sbear/distance_matrix.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/compute_persistence.hpp>
#include <sbear/timeseries.hpp>

#include <sbear/point_cloud.hpp>

#include <string>
#include <type_traits>
#include <vector>

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

    /// (bars, enumeration, critical_edges) of one point cloud or distance
    /// matrix: the bars of
    /// all dimensions as an (m, 2) array, the [start, stop) rows of each
    /// dimension as a (maxDim + 1, 2) array, and per bar the points {a, b,
    /// c, d} whose distances are its birth, between a and b, and its death,
    /// between c and d, with -1 where an endpoint has no edge.
    template <typename ElemT>
    static nb::tuple persistence_critical_edges(const ElemT& points, size_t maxDim, bool reducedHomology)
    {
      std::vector<sb::ph::Barcode<T>> barcodes;
      std::vector<std::vector<sb::ph::detail::CriticalEdges>> edges;
      {
        nb::gil_scoped_release release;
        sb::ph::persistence_with_critical_edges(points, maxDim, reducedHomology, barcodes, edges);
      }
      size_t total = 0;
      for (const auto& barcode : barcodes)
      {
        total += barcode.bars().size();
      }
      auto bars = sb_py::new_array<T>({total, 2});
      auto enumeration = sb_py::new_array<int64_t>({barcodes.size(), 2});
      auto critical = sb_py::new_array<int64_t>({total, 4});
      size_t row = 0;
      for (size_t dim = 0; dim < barcodes.size(); ++dim)
      {
        enumeration.data()[2 * dim] = static_cast<int64_t>(row);
        const auto& dimBars = barcodes[dim].bars();
        for (size_t k = 0; k < dimBars.size(); ++k, ++row)
        {
          bars.data()[2 * row] = dimBars[k].birth;
          bars.data()[2 * row + 1] = dimBars[k].death;
          for (size_t j = 0; j < 4; ++j)
          {
            critical.data()[4 * row + j] = static_cast<int64_t>(edges[dim][k][j]);
          }
        }
        enumeration.data()[2 * dim + 1] = static_cast<int64_t>(row);
      }
      return nb::make_tuple(bars, enumeration, critical);
    }

    static void register_bindings(nb::module_& m, const std::string& suffix)
    {
      nb::class_<PyRipserBindings> cls(
        m, ("PersistenceRipser" + suffix).c_str());
      // For a point cloud, with Euclidean distances, or a distance matrix,
      // whose critical edges are pairs of its rows.
      cls.def_static("persistence_critical_edges",
                     &PyRipserBindings::template persistence_critical_edges<sb::PointCloud<T>>);
      cls.def_static("persistence_critical_edges",
                     &PyRipserBindings::template persistence_critical_edges<sb::DistanceMatrix<T>>);
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
