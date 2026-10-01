#include "py_subsample.hpp"

#include <pybind11/stl.h>
#include <sbear/point_process/subsample_relative.hpp>

namespace py = pybind11;

namespace
{
  template <typename ElementT>
  void bind_sampler(py::module_& m, const char* name)
  {
    using T = typename ElementT::value_type;
    using Task = sb::pp::RelativeSubsampleTask<ElementT>;
    py::class_<Task, sb::StoppableTask<void>>(m, (std::string("_") + name + "Task").c_str())
      .def("result", &Task::result, py::call_guard<py::gil_scoped_release>())
      .def("empty_regions", &Task::empty_regions);
    m.def(name, [](const ElementT& reference, std::optional<sb::pp::RelativeQuery<T>> query,
                   std::vector<sb::sampling::Distribution> distributions,
                   size_t nPoints, size_t nSamples, bool distributionAxis, bool replace,
                   bool allowPartial, bool discardDuplicates, sb::DefaultRandomGenerator* generator) {
      py::gil_scoped_release release;
      auto task = std::make_unique<Task>(reference, query ? std::move(*query) : sb::pp::RelativeQuery<T>{}, std::move(distributions),
        nPoints, nSamples, distributionAxis, replace, allowPartial, discardDuplicates,
        generator ? *generator : sb::default_generator());
      task->start_async(sb::default_executor());
      return task;
    }, py::arg("reference"), py::arg("query"), py::arg("distributions"),
       py::arg("n_points"), py::arg("n_samples"), py::arg("distribution_axis"),
       py::arg("replace"), py::arg("allow_partial"), py::arg("discard_duplicates"),
       py::arg("generator").none(true));
  }
}

void sb_py::register_relative_subsample(py::module_& m)
{
  bind_sampler<sb::PointCloud<sb::float32_t>>(m, "subsample_relative_pcloud32");
  bind_sampler<sb::PointCloud<sb::float64_t>>(m, "subsample_relative_pcloud64");
  bind_sampler<sb::DistanceMatrix<sb::float32_t>>(m, "subsample_relative_distmat32");
  bind_sampler<sb::DistanceMatrix<sb::float64_t>>(m, "subsample_relative_distmat64");
}
