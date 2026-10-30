#include "py_subsample.hpp"
#include "../py_tensor_property_variants.hpp"

#include <sbear/point_process/subsample.hpp>

#include <sbear/executor.hpp>
#include <sbear/random_generator.hpp>

#include <string>

namespace nb = nanobind;

namespace
{
  template <typename ElementT, sb::TensorProperties Properties>
  std::unique_ptr<sb::pp::SubsampleTask<ElementT, Properties>> spawn_subsample(
      const sb::Tensor<ElementT, Properties>& data, size_t nPoints, size_t nSamples, bool replace,
      sb::sampling::PartialPolicy partialPolicy, bool discardDuplicates, sb::DefaultRandomGenerator* gen)
  {
    auto& generator = gen == nullptr ? sb::default_generator() : *gen;
    nb::gil_scoped_release release;
    auto task = std::make_unique<sb::pp::SubsampleTask<ElementT, Properties>>(
        data, nPoints, nSamples, replace, partialPolicy, discardDuplicates, generator);
    task->start_async(sb::default_executor());
    return task;
  }
}

void sb_py::register_point_process_subsample(nb::module_& m)
{
  nb::enum_<sb::sampling::PartialPolicy>(m, "_PartialPolicy")
    .value("no", sb::sampling::PartialPolicy::Disallow)
    .value("keep", sb::sampling::PartialPolicy::Keep)
    .value("drop", sb::sampling::PartialPolicy::Drop);
  // One task class per element type and tensor property variant.
  auto bind = [&]<typename ElementT, sb::TensorProperties Properties>(const char* name, const std::string& taskName) {
    using Task = sb::pp::SubsampleTask<ElementT, Properties>;
    nb::class_<Task, sb::StoppableTask<void>>(m, taskName.c_str())
      .def("result", &Task::result, nb::call_guard<nb::gil_scoped_release>());
    m.def(name, &spawn_subsample<ElementT, Properties>,
        nb::arg("data"), nb::arg("n_points"), nb::arg("n_samples"),
        nb::arg("replace"), nb::arg("allow_partial"), nb::arg("discard_duplicates"),
        nb::arg("generator").none() = nb::none());
  };
  auto bindElement = [&]<typename ElementT>(const char* name, const std::string& elementName) {
    sb_py::bind_tensor_property_variants(
      [&]<sb::TensorProperties Properties>() {
        const bool indexed = Properties == sb::TensorProperty::Indexed;
        bind.operator()<ElementT, Properties>(name, "_SubsampleTask" + elementName + (indexed ? "Indexed" : ""));
      });
  };
  bindElement.operator()<sb::PointCloud<sb::float32_t>>("subsample32", "PointCloud32");
  bindElement.operator()<sb::PointCloud<sb::float64_t>>("subsample64", "PointCloud64");
  bindElement.operator()<sb::DistanceMatrix<sb::float32_t>>("subsample32", "DistanceMatrix32");
  bindElement.operator()<sb::DistanceMatrix<sb::float64_t>>("subsample64", "DistanceMatrix64");
}
