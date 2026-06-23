/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "py_subsample.hpp"

#include <sbear/sampling/subsample.hpp>

#include <pybind11/stl.h>

#include <utility>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PySubsampleBindings
  {
  public:
    using PointCloudT = sb::PointCloud<T>;
    using TensorT = sb::Tensor<T>;

    // The runtime "which built-in distribution" choice lives here: each method
    // picks the matching functor and delegates to the core sampler, which
    // launches a stoppable draw task. Each returns a (task, samples) tuple;
    // samples is a (n_query, n_instances) tensor of indexed subsamples sharing
    // R, filled asynchronously by the task.

    // reference/query arrive as scalar Tensor<T> (a FloatTensor's data) and
    // convert implicitly to PointCloud<T>.
    static py::tuple sample_subsets_distance_gaussian(const TensorT& reference, const TensorT& query,
                                                      T mean, T sigma, size_t sampleSize,
                                                      size_t nInstances, bool replace,
                                                      const sb::DefaultRandomGenerator* gen)
    {
      sb::sampling::SubsampleHandle<T> handle = sb::sampling::sample_subsets(
          PointCloudT(reference), PointCloudT(query),
          sb::sampling::EuclideanDistance<T>{}, sb::sampling::Gaussian<T>{mean, sigma},
          sampleSize, nInstances, replace, pick(gen), sb::default_executor());
      return to_tuple(std::move(handle));
    }

    static py::tuple sample_subsets_distance_uniform(const TensorT& reference, const TensorT& query,
                                                     T inner, T outer, size_t sampleSize,
                                                     size_t nInstances, bool replace,
                                                     const sb::DefaultRandomGenerator* gen)
    {
      sb::sampling::SubsampleHandle<T> handle = sb::sampling::sample_subsets(
          PointCloudT(reference), PointCloudT(query),
          sb::sampling::EuclideanDistance<T>{}, sb::sampling::Uniform<T>{inner, outer},
          sampleSize, nInstances, replace, pick(gen), sb::default_executor());
      return to_tuple(std::move(handle));
    }

    static py::tuple sample_subsets_from_probabilities(const TensorT& reference,
                                                       const TensorT& probabilities,
                                                       size_t sampleSize, size_t nInstances,
                                                       bool replace,
                                                       const sb::DefaultRandomGenerator* gen)
    {
      sb::sampling::SubsampleHandle<T> handle = sb::sampling::sample_subsets_from_probabilities(
          PointCloudT(reference), probabilities,
          sampleSize, nInstances, replace, pick(gen), sb::default_executor());
      return to_tuple(std::move(handle));
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PySubsampleBindings> cls(m, ("Subsample" + suffix).c_str());

      cls
          .def_static("sample_subsets_distance_gaussian",
                      &PySubsampleBindings::sample_subsets_distance_gaussian,
                      py::arg("reference"), py::arg("query"),
                      py::arg("mean"), py::arg("sigma"), py::arg("sample_size"),
                      py::arg("n_instances"), py::arg("replace"),
                      py::arg("generator").none(true) = py::none())
          .def_static("sample_subsets_distance_uniform",
                      &PySubsampleBindings::sample_subsets_distance_uniform,
                      py::arg("reference"), py::arg("query"),
                      py::arg("inner"), py::arg("outer"), py::arg("sample_size"),
                      py::arg("n_instances"), py::arg("replace"),
                      py::arg("generator").none(true) = py::none())
          .def_static("sample_subsets_from_probabilities",
                      &PySubsampleBindings::sample_subsets_from_probabilities,
                      py::arg("reference"), py::arg("probabilities"),
                      py::arg("sample_size"), py::arg("n_instances"), py::arg("replace"),
                      py::arg("generator").none(true) = py::none())
          ;
    }

  private:
    // Unpack a launched run into the (task, samples) tuple Python expects.
    static py::tuple to_tuple(sb::sampling::SubsampleHandle<T> handle)
    {
      return py::make_tuple(std::move(handle.task), std::move(handle.samples));
    }

    static const sb::DefaultRandomGenerator& pick(const sb::DefaultRandomGenerator* gen)
    {
      return gen ? *gen : sb::default_generator();
    }
  };

}

void sb_py::register_sampling_subsample(py::module_& m)
{
  PySubsampleBindings<sb::float32_t>::register_bindings(m, "32");
  PySubsampleBindings<sb::float64_t>::register_bindings(m, "64");
}
