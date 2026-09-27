#include "py_subsample.hpp"
#include "../py_tensor_property_variants.hpp"

#include <sbear/point_process/subsample.hpp>

#include <sbear/executor.hpp>
#include <sbear/random_generator.hpp>

namespace py = pybind11;

namespace
{
  template <typename ElementT, sb::TensorProperties Properties>
  sb::Tensor<ElementT, sb::TensorProperty::Indexed> subsample(
      const sb::Tensor<ElementT, Properties>& data, size_t nPoints, size_t nSamples, bool replace,
      bool allowPartial, bool discardDuplicates, sb::DefaultRandomGenerator* gen)
  {
    auto& generator = gen == nullptr ? sb::default_generator() : *gen;
    py::gil_scoped_release release;
    return sb::pp::subsample(
        data, nPoints, nSamples, replace, allowPartial, discardDuplicates, generator, sb::default_executor());
  }
}

void sb_py::register_point_process_subsample(py::module_& m)
{
  auto bind = [&]<typename ElementT, sb::TensorProperties Properties>(const char* name) {
    m.def(name, &subsample<ElementT, Properties>,
        py::arg("data"), py::arg("n_points"), py::arg("n_samples"),
        py::arg("replace"), py::arg("allow_partial"), py::arg("discard_duplicates"),
        py::arg("generator").none(true) = py::none());
  };
  auto bindElement = [&]<typename ElementT>(const char* name) {
    sb_py::bind_tensor_property_variants(
      [&]<sb::TensorProperties Properties>() {
        bind.operator()<ElementT, Properties>(name);
      });
  };
  bindElement.operator()<sb::PointCloud<sb::float32_t>>("subsample32");
  bindElement.operator()<sb::PointCloud<sb::float64_t>>("subsample64");
  bindElement.operator()<sb::DistanceMatrix<sb::float32_t>>("subsample32");
  bindElement.operator()<sb::DistanceMatrix<sb::float64_t>>("subsample64");
}
