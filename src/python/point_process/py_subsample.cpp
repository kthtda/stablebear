#include "py_subsample.hpp"

#include <sbear/point_process/subsample.hpp>

#include <sbear/executor.hpp>
#include <sbear/random_generator.hpp>

namespace py = pybind11;

namespace
{
  template <typename T>
  sb::Tensor<sb::PointCloud<T>> subsample(
      const sb::Tensor<sb::PointCloud<T>>& points, size_t nPoints, size_t nSamples, bool replace,
      bool allowPartial, bool discardDuplicates, sb::DefaultRandomGenerator* gen)
  {
    auto& generator = gen == nullptr ? sb::default_generator() : *gen;
    py::gil_scoped_release release;
    return sb::pp::subsample(
        points, nPoints, nSamples, replace, allowPartial, discardDuplicates, generator, sb::default_executor());
  }
}

void sb_py::register_point_process_subsample(py::module_& m)
{
  m.def(
      "subsample32", &subsample<sb::float32_t>, py::arg("points"), py::arg("n_points"), py::arg("n_samples"),
      py::arg("replace"), py::arg("allow_partial"), py::arg("discard_duplicates"),
      py::arg("generator").none(true) = py::none());
  m.def(
      "subsample64", &subsample<sb::float64_t>, py::arg("points"), py::arg("n_points"), py::arg("n_samples"),
      py::arg("replace"), py::arg("allow_partial"), py::arg("discard_duplicates"),
      py::arg("generator").none(true) = py::none());
}
