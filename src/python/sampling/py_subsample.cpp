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

namespace py = pybind11;

namespace
{

  template <typename T>
  class PySubsampleBindings
  {
  public:
    using PointCloudT = sb::PointCloud<T>;
    using TensorT = sb::Tensor<T>;
    using OutT = sb::Tensor<sb::PointCloud<T>>;

    // The runtime "which built-in distribution" choice lives here: each method
    // instantiates the templated core sampler with the matching functor. Each
    // returns a (n_query, n_instances) tensor of indexed subsamples sharing R.

    // reference/query arrive as scalar Tensor<T> (a FloatTensor's data) and
    // convert implicitly to PointCloud<T>.
    static OutT sample_subsets_distance_gaussian(const TensorT& reference, const TensorT& query,
                                                 T mean, T sigma, size_t sampleSize,
                                                 size_t nInstances, bool replace,
                                                 const sb::DefaultRandomGenerator* gen)
    {
      return sb::sampling::sample_subsets(PointCloudT(reference), PointCloudT(query),
                                            sb::sampling::EuclideanDistance<T>{},
                                            sb::sampling::Gaussian<T>{mean, sigma},
                                            sampleSize, nInstances, replace, pick(gen),
                                            sb::default_executor());
    }

    static OutT sample_subsets_distance_identity(const TensorT& reference, const TensorT& query,
                                                 size_t sampleSize, size_t nInstances,
                                                 bool replace,
                                                 const sb::DefaultRandomGenerator* gen)
    {
      return sb::sampling::sample_subsets(PointCloudT(reference), PointCloudT(query),
                                            sb::sampling::EuclideanDistance<T>{},
                                            sb::sampling::Identity<T>{},
                                            sampleSize, nInstances, replace, pick(gen),
                                            sb::default_executor());
    }

    static OutT sample_subsets_from_probabilities(const TensorT& reference, const TensorT& probabilities,
                                                  size_t sampleSize, size_t nInstances, bool replace,
                                                  const sb::DefaultRandomGenerator* gen)
    {
      return sb::sampling::sample_subsets_from_probabilities(PointCloudT(reference), probabilities,
                                                               sampleSize, nInstances, replace,
                                                               pick(gen), sb::default_executor());
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
          .def_static("sample_subsets_distance_identity",
                      &PySubsampleBindings::sample_subsets_distance_identity,
                      py::arg("reference"), py::arg("query"),
                      py::arg("sample_size"), py::arg("n_instances"), py::arg("replace"),
                      py::arg("generator").none(true) = py::none())
          .def_static("sample_subsets_from_probabilities",
                      &PySubsampleBindings::sample_subsets_from_probabilities,
                      py::arg("reference"), py::arg("probabilities"),
                      py::arg("sample_size"), py::arg("n_instances"), py::arg("replace"),
                      py::arg("generator").none(true) = py::none())
          ;
    }

  private:
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
