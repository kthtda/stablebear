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

#include "../py_async_support.hpp"

#include <sbear/sampling/subsample.hpp>

#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>

namespace py = pybind11;

namespace
{

  // Stoppable, progress-reporting draw task. The weight matrix is built up
  // front (cheap relative to the draw); this task runs the per-(query,instance)
  // draw asynchronously so it can be cancelled (request_stop) and report
  // progress, mirroring the distance/persistence tasks. The output tensor is
  // allocated once and shared (Tensor is shared_ptr-backed) with the handle
  // returned to Python, which reads it only after the task completes.
  template <typename T>
  class SubsampleTask : public sb::StoppableTask<void>
  {
  public:
    using PointCloudT = sb::PointCloud<T>;
    using OutT = sb::Tensor<PointCloudT>;

    SubsampleTask(PointCloudT R, sb::Tensor<T> weights, OutT out, size_t sampleSize,
                  bool replace, sb::DefaultRandomGenerator gen)
      : m_R(std::move(R)), m_weights(std::move(weights)), m_out(std::move(out)),
        m_sampleSize(sampleSize), m_replace(replace), m_gen(std::move(gen))
    { }

  private:
    tf::Future<void> run_async(sb::Executor& exec) override
    {
      const size_t nR = m_weights.shape(1);
      next_step(m_out.size(), "Drawing subsamples.", "subsample");

      // Walk the (n_query, n_instances) output grid; the per-element engine is
      // seeded from the element's flat index, so results are independent of
      // thread count. Each cell draws one subsample into a shared, indexed view.
      return sb::parallel_walk_async(m_out, m_gen,
          [this, nR](const std::vector<size_t>& idx, auto& engine) {
        if (stop_requested())
          return;
        sb::Tensor<uint64_t> row({m_sampleSize});
        sb::sampling::detail::draw_indices(
            m_weights, idx[0], nR, m_sampleSize, m_replace, engine,
            [&row](size_t s, size_t r) { row({s}) = static_cast<uint64_t>(r); });
        m_out(idx) = PointCloudT(m_R, std::move(row));
        add_progress(1);
      }, exec);
    }

    PointCloudT m_R;
    sb::Tensor<T> m_weights;
    OutT m_out;
    size_t m_sampleSize;
    bool m_replace;
    sb::DefaultRandomGenerator m_gen;  // owned: captured by reference in the async walk
  };

  template <typename T>
  class PySubsampleBindings
  {
  public:
    using PointCloudT = sb::PointCloud<T>;
    using TensorT = sb::Tensor<T>;
    using OutT = sb::Tensor<sb::PointCloud<T>>;

    // The runtime "which built-in distribution" choice lives here: each method
    // builds the weight matrix with the matching functor (or accepts a
    // precomputed one), then launches a stoppable draw task. Each returns a
    // (task, output) tuple; output is a (n_query, n_instances) tensor of indexed
    // subsamples sharing R, filled asynchronously by the task.

    // reference/query arrive as scalar Tensor<T> (a FloatTensor's data) and
    // convert implicitly to PointCloud<T>.
    static py::tuple sample_subsets_distance_gaussian(const TensorT& reference, const TensorT& query,
                                                      T mean, T sigma, size_t sampleSize,
                                                      size_t nInstances, bool replace,
                                                      const sb::DefaultRandomGenerator* gen)
    {
      return launch(PointCloudT(reference), PointCloudT(query),
                    sb::sampling::EuclideanDistance<T>{},
                    sb::sampling::Gaussian<T>{mean, sigma},
                    sampleSize, nInstances, replace, gen);
    }

    static py::tuple sample_subsets_distance_identity(const TensorT& reference, const TensorT& query,
                                                      size_t sampleSize, size_t nInstances,
                                                      bool replace,
                                                      const sb::DefaultRandomGenerator* gen)
    {
      return launch(PointCloudT(reference), PointCloudT(query),
                    sb::sampling::EuclideanDistance<T>{},
                    sb::sampling::Identity<T>{},
                    sampleSize, nInstances, replace, gen);
    }

    static py::tuple sample_subsets_from_probabilities(const TensorT& reference,
                                                       const TensorT& probabilities,
                                                       size_t sampleSize, size_t nInstances,
                                                       bool replace,
                                                       const sb::DefaultRandomGenerator* gen)
    {
      PointCloudT R(reference);
      sb::sampling::detail::validate_reference(R, sampleSize, replace);
      if (probabilities.rank() != 2)
        throw std::invalid_argument("probabilities must be a 2-D (n_query, n_reference) array");
      if (probabilities.shape(1) != R.n_points())
        throw std::invalid_argument("probabilities must have one column per reference point");

      size_t nQuery = probabilities.shape(0);
      OutT out({nQuery, nInstances});
      std::unique_ptr<sb::StoppableTask<void>> task =
          sb_py::execute_stoppable_task<SubsampleTask<T>>(
              std::move(R), probabilities, out, sampleSize, replace, pick(gen));
      return py::make_tuple(std::move(task), out);
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
    // Validate, build the weight matrix, then launch the stoppable draw task.
    template <typename FilterF, typename DistF>
    static py::tuple launch(PointCloudT R, PointCloudT X, FilterF filter, DistF distribution,
                            size_t sampleSize, size_t nInstances, bool replace,
                            const sb::DefaultRandomGenerator* gen)
    {
      sb::Executor& exec = sb::default_executor();
      sb::sampling::detail::validate_reference(R, sampleSize, replace);
      if (X.rank() != 2)
        throw std::invalid_argument("query must be a 2-D (n_points, dim) point cloud");
      if (X.dim() != R.dim())
        throw std::invalid_argument("reference and query must have the same dimension");

      TensorT weights = sb::sampling::detail::compute_weights(R, X, filter, distribution, exec);
      OutT out({X.n_points(), nInstances});

      std::unique_ptr<sb::StoppableTask<void>> task =
          sb_py::execute_stoppable_task<SubsampleTask<T>>(
              std::move(R), std::move(weights), out, sampleSize, replace, pick(gen));
      return py::make_tuple(std::move(task), out);
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
