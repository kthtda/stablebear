#include "py_subsample.hpp"

#include <sbear/point_process/subsample.hpp>

#include <sbear/executor.hpp>
#include <sbear/random_generator.hpp>

namespace py = pybind11;

namespace
{
  template <typename T, sb::TensorProperties Properties>
  sb::Tensor<sb::PointCloud<T>, sb::TensorProperty::Indexed> subsample(
      const sb::Tensor<sb::PointCloud<T>, Properties>& points, size_t nPoints, size_t nSamples, bool replace,
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
  auto bind = [&]<typename T, sb::TensorProperties Properties>(const char* name) {
    m.def(name, &subsample<T, Properties>,
        py::arg("points"), py::arg("n_points"), py::arg("n_samples"),
        py::arg("replace"), py::arg("allow_partial"), py::arg("discard_duplicates"),
        py::arg("generator").none(true) = py::none());
  };
  bind.operator()<sb::float32_t, sb::TensorProperty::None>("subsample32");
  bind.operator()<sb::float32_t, sb::TensorProperty::Indexed>("subsample32");
  bind.operator()<sb::float64_t, sb::TensorProperty::None>("subsample64");
  bind.operator()<sb::float64_t, sb::TensorProperty::Indexed>("subsample64");
}
