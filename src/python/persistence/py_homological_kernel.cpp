#include "py_homological_kernel.hpp"
#include "../py_async_support.hpp"
#include "../py_tensor_property_variants.hpp"
#include "../py_np_support.hpp"

#include <sbear/distance_matrix.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/compute_homological_kernel.hpp>
#include <sbear/tensor.hpp>

namespace nb = nanobind;

namespace
{

  // Naming seam: the Python API calls the two inputs X and Y, the C++ layer
  // names them after the metrics they carry — X induces d (the dominating
  // metric), Y induces d' (the dominated one, d' <= d pointwise).
  template <typename T>
  class PyHomologicalKernelBindings
  {
  public:
    template <sb::TensorProperties Properties, sb::TensorProperties PrimeProperties>
    static std::unique_ptr<sb::StoppableTask<void>> spawn_homological_kernel_pcloud_task(
        const sb::Tensor<sb::PointCloud<T>, Properties> &pclouds,
        const sb::Tensor<sb::PointCloud<T>, PrimeProperties> &pcloudsPrime,
        sb::Tensor<sb::ph::Barcode<T>> &out)
    {
      return sb_py::execute_stoppable_task<sb::ph::HomologicalKernelImpl<
          sb::PointCloud<T>, T, Properties, PrimeProperties>>(
          pclouds, pcloudsPrime, out);
    }

    static std::unique_ptr<sb::StoppableTask<void>> spawn_homological_kernel_single_pcloud_task(
        const sb::PointCloud<T> &pointCloud, const sb::PointCloud<T> &pointCloudPrime,
        sb::Tensor<sb::ph::Barcode<T>> &out)
    {
      sb::Tensor<sb::PointCloud<T>> pointClouds({1});
      sb::Tensor<sb::PointCloud<T>> pointCloudsPrime({1});
      pointClouds(0) = pointCloud.copy();
      pointCloudsPrime(0) = pointCloudPrime.copy();
      return sb_py::execute_stoppable_task<sb::ph::HomologicalKernelImpl<sb::PointCloud<T>, T>>(
          std::move(pointClouds), std::move(pointCloudsPrime), out);
    }

    template <sb::TensorProperties Properties, sb::TensorProperties PrimeProperties>
    static std::unique_ptr<sb::StoppableTask<void>> spawn_homological_kernel_distmat_task(
        const sb::Tensor<sb::DistanceMatrix<T>, Properties> &dmats,
        const sb::Tensor<sb::DistanceMatrix<T>, PrimeProperties> &dmatsPrime,
        sb::Tensor<sb::ph::Barcode<T>> &out)
    {
      return sb_py::execute_stoppable_task<sb::ph::HomologicalKernelImpl<
          sb::DistanceMatrix<T>, T, Properties, PrimeProperties>>(
          dmats, dmatsPrime, out);
    }

    /// (bars, critical_edges) of one pair of point clouds: bars as an (m, 2)
    /// array and, per bar, the point indices {a, b, c, e} of the edges whose
    /// lengths in Y and X are its birth and death.
    static nb::tuple homological_kernel_critical_edges(
        const sb::PointCloud<T> &pointCloud, const sb::PointCloud<T> &pointCloudPrime)
    {
      sb::ph::Barcode<T> barcode;
      std::vector<std::array<size_t, 4>> edges;
      {
        nb::gil_scoped_release release;
        sb::ph::homological_kernel_with_critical_edges(pointCloud, pointCloudPrime, barcode, edges);
      }
      auto bars = sb_py::new_array<T>({barcode.bars().size(), 2});
      auto critical = sb_py::new_array<int64_t>({edges.size(), 4});
      for (size_t i = 0; i < edges.size(); ++i)
      {
        bars.data()[2 * i] = barcode.bars()[i].birth;
        bars.data()[2 * i + 1] = barcode.bars()[i].death;
        for (size_t j = 0; j < 4; ++j)
        {
          critical.data()[4 * i + j] = static_cast<int64_t>(edges[i][j]);
        }
      }
      return nb::make_tuple(bars, critical);
    }

    static void register_bindings(nb::module_ &m, const std::string &suffix)
    {
      nb::class_<PyHomologicalKernelBindings> cls(
        m, ("HomologicalKernel" + suffix).c_str());
      sb_py::bind_tensor_property_pairs(
        [&]<sb::TensorProperties Properties,
            sb::TensorProperties PrimeProperties>() {
          cls.def_static(
            "spawn_homological_kernel_pcloud_task",
            &PyHomologicalKernelBindings::template spawn_homological_kernel_pcloud_task<
              Properties, PrimeProperties>);
          cls.def_static(
            "spawn_homological_kernel_distmat_task",
            &PyHomologicalKernelBindings::template spawn_homological_kernel_distmat_task<
              Properties, PrimeProperties>);
        });
      cls.def_static(
        "spawn_homological_kernel_pcloud_task",
        &PyHomologicalKernelBindings::spawn_homological_kernel_single_pcloud_task);
      cls.def_static(
        "homological_kernel_critical_edges", &PyHomologicalKernelBindings::homological_kernel_critical_edges);
    }
  };

} // namespace

namespace sb_py
{
  void register_persistence_homological_kernel(nb::module_ &m)
  {
    PyHomologicalKernelBindings<sb::float32_t>::register_bindings(m, "32");
    PyHomologicalKernelBindings<sb::float64_t>::register_bindings(m, "64");
  }
} // namespace sb_py
