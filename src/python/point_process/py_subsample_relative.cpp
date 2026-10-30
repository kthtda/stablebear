#include "py_subsample.hpp"

#include <sbear/point_process/subsample_relative.hpp>

namespace nb = nanobind;

namespace
{
  template <typename ElementT>
  void bind_sampler(nb::module_& m, const char* name)
  {
    using T = typename ElementT::value_type;
    using Task = sb::pp::RelativeSubsampleTask<ElementT>;
    nb::class_<Task, sb::StoppableTask<void>>(m, (std::string("_") + name + "Task").c_str())
      .def("result", &Task::result, nb::call_guard<nb::gil_scoped_release>());
    m.def(name, [](const ElementT& reference, std::optional<sb::pp::RelativeQuery<T>> query,
                   std::vector<sb::random::Distribution> distributions,
                   size_t nPoints, size_t nSamples, bool distributionAxis, bool replace,
                   sb::sampling::PartialPolicy partialPolicy, bool discardDuplicates,
                   sb::DefaultRandomGenerator* generator) {
      nb::gil_scoped_release release;
      auto task = std::make_unique<Task>(reference, query ? std::move(*query) : sb::pp::RelativeQuery<T>{}, std::move(distributions),
        nPoints, nSamples, distributionAxis, replace, partialPolicy, discardDuplicates,
        generator ? *generator : sb::default_generator());
      task->start_async(sb::default_executor());
      return task;
    }, nb::arg("reference"), nb::arg("query"), nb::arg("distributions"),
       nb::arg("n_points"), nb::arg("n_samples"), nb::arg("distribution_axis"),
       nb::arg("replace"), nb::arg("allow_partial"), nb::arg("discard_duplicates"),
       nb::arg("generator").none());
  }
}

void sb_py::register_relative_subsample(nb::module_& m)
{
  bind_sampler<sb::PointCloud<sb::float32_t>>(m, "subsample_relative_pcloud32");
  bind_sampler<sb::PointCloud<sb::float64_t>>(m, "subsample_relative_pcloud64");
  bind_sampler<sb::DistanceMatrix<sb::float32_t>>(m, "subsample_relative_distmat32");
  bind_sampler<sb::DistanceMatrix<sb::float64_t>>(m, "subsample_relative_distmat64");
}
