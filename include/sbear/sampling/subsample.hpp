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

#ifndef STABLEBEAR_SAMPLING_SUBSAMPLE_H
#define STABLEBEAR_SAMPLING_SUBSAMPLE_H

#include "../executor.hpp"
#include "../point_cloud.hpp"
#include "../random_generator.hpp"
#include "../task.hpp"
#include "../tensor.hpp"
#include "../walk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace sb::sampling
{

  // ===========================================================================
  // Built-in functors
  //
  // A filter maps a (query point, reference point) pair to a scalar; a
  // distribution maps a filter value to a non-negative weight. The samplers
  // below take these as plain functors, so any callable with the matching
  // signature works just as well as the built-ins. A "point" is identified by
  // the cloud it belongs to and its row index, e.g. filter(X, i, R, r).
  // ===========================================================================

  /// Euclidean distance between query point @p i of @p X and reference point @p r of @p R.
  template <typename T>
  struct EuclideanDistance
  {
    T operator()(const PointCloud<T>& X, size_t i, const PointCloud<T>& R, size_t r) const
    {
      if (X.dim() != R.dim())
        throw std::invalid_argument("query and reference points must have the same dimension");
      T acc = T(0);
      for (size_t k = 0; k < X.dim(); ++k)
      {
        T d = X(i, k) - R(r, k);
        acc += d * d;
      }
      return std::sqrt(acc);
    }
  };

  /// Identity distribution: the filter value is used directly as the weight.
  template <typename T>
  struct Identity
  {
    T operator()(T v) const { return v; }
  };

  /// Unnormalized Gaussian of the filter value.
  template <typename T>
  struct Gaussian
  {
    T mean = T(0);
    T sigma = T(1);

    T operator()(T v) const
    {
      T d = (v - mean) / sigma;
      return std::exp(T(-0.5) * d * d);
    }
  };

  /// A launched subsampling run: the in-flight draw @p task plus the @p samples
  /// tensor it fills, of shape (n_query, n_instances). The two share storage —
  /// the task writes into @p samples — so read @p samples only once @p task
  /// reports complete.
  template <typename T>
  struct SubsampleHandle
  {
    std::unique_ptr<StoppableTask<void>> task;
    Tensor<PointCloud<T>> samples;
  };

  namespace detail
  {

    /// Build a cumulative distribution (prefix sums) from non-negative weights.
    template <typename T>
    void build_cdf(std::vector<T>& cdf, const std::vector<T>& weights)
    {
      cdf.resize(weights.size());
      T total = T(0);
      for (size_t i = 0; i < weights.size(); ++i)
      {
        T w = weights[i];
        if (w < T(0))
          throw std::invalid_argument("sampling weights must be non-negative");
        total += w;
        cdf[i] = total;
      }
      if (!(total > T(0)))
        throw std::invalid_argument("sampling weights must have a positive sum");
    }

    /// Draw a single reference index from a CDF via binary search.
    template <typename T, typename EngineT>
    size_t draw_one(const std::vector<T>& cdf, EngineT& engine)
    {
      std::uniform_real_distribution<T> u(T(0), cdf.back());
      auto it = std::upper_bound(cdf.begin(), cdf.end(), u(engine));
      size_t idx = static_cast<size_t>(it - cdf.begin());
      return std::min(idx, cdf.size() - 1);
    }

    /// Draw @p k reference indices for query @p qi (row @p qi of the
    /// (n_query, n_reference) weight matrix @p weights) and hand each drawn
    /// index to @p write(s, refIndex).
    template <typename T, typename EngineT, typename Writer>
    void draw_indices(const Tensor<T>& weights, size_t qi, size_t nR, size_t k,
                      bool replace, EngineT& engine, Writer write)
    {
      std::vector<T> w(nR);
      for (size_t r = 0; r < nR; ++r)
        w[r] = weights({qi, r});

      std::vector<T> cdf;

      if (replace)
      {
        build_cdf(cdf, w);
        for (size_t s = 0; s < k; ++s)
          write(s, draw_one(cdf, engine));
      }
      else
      {
        // Weighted sampling without replacement: zero a weight once chosen and
        // rebuild the CDF for the remaining points.
        for (size_t s = 0; s < k; ++s)
        {
          build_cdf(cdf, w);
          size_t r = draw_one(cdf, engine);
          write(s, r);
          w[r] = T(0);
        }
      }
    }

    template <typename T>
    void validate_reference(const PointCloud<T>& R, size_t sampleSize, bool replace)
    {
      if (R.rank() != 2)
        throw std::invalid_argument("reference must be a 2-D (n_points, dim) point cloud");
      if (sampleSize == 0)
        throw std::invalid_argument("sample_size must be positive");
      if (!replace && sampleSize > R.n_points())
        throw std::invalid_argument("sample_size must not exceed the number of reference "
                                    "points when sampling without replacement");
    }

    /// Stoppable, progress-reporting draw: fills a (n_query, n_instances) tensor
    /// of indexed subsamples, each drawn from a row of the weight matrix. Every
    /// subsample is an *indexed* PointCloud sharing the reference coordinates —
    /// only the chosen indices are stored, never the coordinates. The per-element
    /// engine (seeded from the flat index) keeps results independent of thread
    /// count. The output tensor is allocated by the caller and shared (Tensor is
    /// shared_ptr-backed); it is read only after the task completes.
    template <typename T>
    class SubsampleTask : public StoppableTask<void>
    {
    public:
      SubsampleTask(PointCloud<T> R, Tensor<T> weights, Tensor<PointCloud<T>> out,
                    size_t sampleSize, bool replace, DefaultRandomGenerator gen)
        : m_R(std::move(R)), m_weights(std::move(weights)), m_out(std::move(out)),
          m_sampleSize(sampleSize), m_replace(replace), m_gen(std::move(gen))
      { }

    private:
      tf::Future<void> run_async(Executor& exec) override
      {
        const size_t nR = m_weights.shape(1);
        next_step(m_out.size(), "Drawing subsamples.", "subsample");

        // Walk the (n_query, n_instances) output grid; the per-element engine is
        // seeded from the element's flat index, so results are independent of
        // thread count. Each cell draws one subsample into a shared, indexed view.
        return parallel_walk_async(m_out, m_gen,
            [this, nR](const std::vector<size_t>& idx, auto& engine) {
          if (this->stop_requested())
            return;
          Tensor<uint64_t> row({m_sampleSize});
          draw_indices(m_weights, idx[0], nR, m_sampleSize, m_replace, engine,
                       [&row](size_t s, size_t r) { row({s}) = static_cast<uint64_t>(r); });
          m_out(idx) = PointCloud<T>(m_R, std::move(row));
          this->add_progress(1);
        }, exec);
      }

      PointCloud<T> m_R;
      Tensor<T> m_weights;
      Tensor<PointCloud<T>> m_out;
      size_t m_sampleSize;
      bool m_replace;
      DefaultRandomGenerator m_gen;  // owned: captured by reference in the async walk
    };

    /// Shared draw step: allocate the (n_query, n_instances) output and launch a
    /// stoppable SubsampleTask that fills it from the (n_query, n_reference)
    /// weight matrix @p weights. Returns the running task together with its
    /// (not-yet-filled) output tensor.
    template <typename T>
    SubsampleHandle<T> draw_subsets_from_weights(PointCloud<T> R, Tensor<T> weights,
                                                 size_t sampleSize, size_t nInstances, bool replace,
                                                 DefaultRandomGenerator gen, Executor& exec)
    {
      Tensor<PointCloud<T>> samples({weights.shape(0), nInstances});
      auto task = std::make_unique<SubsampleTask<T>>(
          std::move(R), std::move(weights), samples, sampleSize, replace, std::move(gen));
      task->start_async(exec);
      return {std::move(task), std::move(samples)};
    }

    /// Evaluate @p distribution(@p filter(query, reference)) for every
    /// (query, reference) pair into an (n_query, n_reference) weight matrix.
    template <typename T, typename FilterF, typename DistF>
    Tensor<T> compute_weights(const PointCloud<T>& R, const PointCloud<T>& X,
                              FilterF filter, DistF distribution, Executor& exec)
    {
      Tensor<T> weights({X.n_points(), R.n_points()});
      parallel_walk(weights,
          [&weights, &R, &X, filter, distribution](const std::vector<size_t>& idx) {
        weights(idx) = distribution(filter(X, idx[0], R, idx[1]));
      }, exec);
      return weights;
    }

  } // namespace detail

  // ===========================================================================
  // Samplers
  // ===========================================================================

  /// Per-query-point subsampling of a reference point cloud, with sampling
  /// weights given by @p distribution applied to @p filter of each
  /// (query point, reference point) pair.
  ///
  /// Launches the draw asynchronously and returns a SubsampleHandle: a
  /// (n_query, n_instances) @p samples tensor whose element (i, j) is the j-th
  /// subsample for query point i — an indexed view sharing @p R's coordinates
  /// (each of shape (sample_size, dim)) — together with the task filling it.
  template <typename T, typename FilterF, typename DistF>
  SubsampleHandle<T> sample_subsets(const PointCloud<T>& R, const PointCloud<T>& X,
                                    FilterF filter, DistF distribution, size_t sampleSize,
                                    size_t nInstances, bool replace,
                                    DefaultRandomGenerator gen, Executor& exec)
  {
    detail::validate_reference(R, sampleSize, replace);
    if (X.rank() != 2)
      throw std::invalid_argument("query must be a 2-D (n_points, dim) point cloud");
    if (X.dim() != R.dim())
      throw std::invalid_argument("reference and query must have the same dimension");

    // Evaluate the filter/distribution once per (query, reference) pair into a
    // weight matrix, then share the draw step with the precomputed-weight path.
    Tensor<T> weights = detail::compute_weights(R, X, filter, distribution, exec);

    return detail::draw_subsets_from_weights(R, std::move(weights), sampleSize, nInstances,
                                             replace, std::move(gen), exec);
  }

  /// Per-query-point subsampling from precomputed sampling weights.
  ///
  /// @p probabilities must have shape (n_query, n_reference); row i gives the
  /// (unnormalized, non-negative) sampling weights over the reference for query
  /// point i. Launches the draw asynchronously and returns a SubsampleHandle
  /// whose @p samples tensor has shape (n_query, n_instances).
  template <typename T>
  SubsampleHandle<T> sample_subsets_from_probabilities(const PointCloud<T>& R,
                                                       Tensor<T> probabilities,
                                                       size_t sampleSize, size_t nInstances,
                                                       bool replace, DefaultRandomGenerator gen,
                                                       Executor& exec)
  {
    detail::validate_reference(R, sampleSize, replace);
    if (probabilities.rank() != 2)
      throw std::invalid_argument("probabilities must be a 2-D (n_query, n_reference) array");
    if (probabilities.shape(1) != R.n_points())
      throw std::invalid_argument("probabilities must have one column per reference point");

    return detail::draw_subsets_from_weights(R, std::move(probabilities), sampleSize, nInstances,
                                             replace, std::move(gen), exec);
  }

}

#endif
