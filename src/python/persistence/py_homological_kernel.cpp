#include "py_homological_kernel.hpp"
#include "../py_async_support.hpp"

#include <sbear/distance_matrix.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/compute_homological_kernel.hpp>
#include <sbear/tensor.hpp>

namespace py = pybind11;

namespace
{

  // Naming seam: the Python API calls the two inputs X and Y, the C++ layer
  // names them after the metrics they carry — X induces d (the dominating
  // metric), Y induces d' (the dominated one, d' <= d pointwise).
  template <typename T>
  class PyHomologicalKernelBindings
  {
  public:
    static std::unique_ptr<sb::StoppableTask<void>> spawn_homological_kernel_pcloud_task(
        const sb::Tensor<sb::PointCloud<T>> &pclouds, const sb::Tensor<sb::PointCloud<T>> &pcloudsPrime,
        sb::Tensor<sb::ph::Barcode<T>> &out)
    {
      return sb_py::execute_stoppable_task<sb::ph::HomologicalKernelImpl<sb::PointCloud<T>, T>>(
          pclouds, pcloudsPrime, out);
    }

    static std::unique_ptr<sb::StoppableTask<void>> spawn_homological_kernel_pcloud_task(
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

    static std::unique_ptr<sb::StoppableTask<void>> spawn_homological_kernel_distmat_task(
        const sb::Tensor<sb::DistanceMatrix<T>> &dmats, const sb::Tensor<sb::DistanceMatrix<T>> &dmatsPrime,
        sb::Tensor<sb::ph::Barcode<T>> &out)
    {
      return sb_py::execute_stoppable_task<sb::ph::HomologicalKernelImpl<sb::DistanceMatrix<T>, T>>(
          dmats, dmatsPrime, out);
    }

    static void register_bindings(py::module_ &m, const std::string &suffix)
    {
      py::class_<PyHomologicalKernelBindings>(m, ("HomologicalKernel" + suffix).c_str())
          .def_static(
              "spawn_homological_kernel_pcloud_task",
              py::overload_cast<
                  const sb::Tensor<sb::PointCloud<T>> &, const sb::Tensor<sb::PointCloud<T>> &,
                  sb::Tensor<sb::ph::Barcode<T>> &
                  >(&PyHomologicalKernelBindings::spawn_homological_kernel_pcloud_task)
          )
          .def_static(
              "spawn_homological_kernel_pcloud_task",
              py::overload_cast<
                  const sb::PointCloud<T> &, const sb::PointCloud<T> &,
                  sb::Tensor<sb::ph::Barcode<T>> &
                  >(&PyHomologicalKernelBindings::spawn_homological_kernel_pcloud_task)
          )
          .def_static(
              "spawn_homological_kernel_distmat_task",
              &PyHomologicalKernelBindings::spawn_homological_kernel_distmat_task);
    }
  };

} // namespace

namespace sb_py
{
  void register_persistence_homological_kernel(pybind11::module_ &m)
  {
    PyHomologicalKernelBindings<sb::float32_t>::register_bindings(m, "32");
    PyHomologicalKernelBindings<sb::float64_t>::register_bindings(m, "64");
  }
} // namespace sb_py
