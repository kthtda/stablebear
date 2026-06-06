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
#include "../tensor.hpp"
#include "../walk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
      T acc = T(0);
      for (size_t k = 0; k < X.dim(); ++k)
      {
        T d = X.coord(i, k) - R.coord(r, k);
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

    /// Shared draw step: build the (n_query, n_instances) tensor of subsamples,
    /// drawing each from row i of the (n_query, n_reference) weight matrix
    /// @p weights. Each subsample is an *indexed* PointCloud sharing @p R's
    /// coordinates — only the chosen indices are stored, never the coordinates.
    /// The per-entry engine from parallel_walk makes the result deterministic
    /// regardless of thread count.
    template <typename T, typename EngineT>
    Tensor<PointCloud<T>> draw_subsets_from_weights(const PointCloud<T>& R, const Tensor<T>& weights,
                                                    size_t sampleSize, size_t nInstances, bool replace,
                                                    const RandomGenerator<EngineT>& gen, Executor& exec)
    {
      size_t nQuery = weights.shape(0);
      size_t nR = weights.shape(1);

      Tensor<PointCloud<T>> out({nQuery, nInstances});

      // One parallel task per (query, instance); the grid only drives the walk.
      Tensor<uint8_t> grid({nQuery, nInstances});
      parallel_walk(grid, gen,
          [&out, &R, &weights, nR, sampleSize, replace](const std::vector<size_t>& idx,
                                                        EngineT& engine) {
        Tensor<uint64_t> row({sampleSize});
        draw_indices(weights, idx[0], nR, sampleSize, replace, engine,
                     [&row](size_t s, size_t r) { row({s}) = static_cast<uint64_t>(r); });
        out(idx) = PointCloud<T>(R, std::move(row));
      }, exec);

      return out;
    }

  } // namespace detail

  // ===========================================================================
  // Samplers
  // ===========================================================================

  /// Per-query-point subsampling of a reference point cloud, with sampling
  /// weights given by @p distribution applied to @p filter of each
  /// (query point, reference point) pair.
  ///
  /// Returns a (n_query, n_instances) tensor of subsamples; element (i, j) is the
  /// j-th subsample for query point i, as an indexed view sharing @p R's
  /// coordinates (each of shape (sample_size, dim)).
  template <typename T, typename FilterF, typename DistF, typename EngineT>
  Tensor<PointCloud<T>> sample_subsets(const PointCloud<T>& R, const PointCloud<T>& X,
                                       FilterF filter, DistF distribution, size_t sampleSize,
                                       size_t nInstances, bool replace,
                                       const RandomGenerator<EngineT>& gen, Executor& exec)
  {
    detail::validate_reference(R, sampleSize, replace);
    if (X.rank() != 2)
      throw std::invalid_argument("query must be a 2-D (n_points, dim) point cloud");
    if (X.dim() != R.dim())
      throw std::invalid_argument("reference and query must have the same dimension");

    // Evaluate the filter/distribution once per (query, reference) pair into a
    // weight matrix, then share the draw step with the precomputed-weight path.
    Tensor<T> weights({X.n_points(), R.n_points()});
    parallel_walk(weights,
        [&weights, &R, &X, filter, distribution](const std::vector<size_t>& idx) {
      weights(idx) = distribution(filter(X, idx[0], R, idx[1]));
    }, exec);

    return detail::draw_subsets_from_weights(R, weights, sampleSize, nInstances, replace, gen, exec);
  }

  /// Per-query-point subsampling from precomputed sampling weights.
  ///
  /// @p probabilities must have shape (n_query, n_reference); row i gives the
  /// (unnormalized, non-negative) sampling weights over the reference for query
  /// point i. Returns a (n_query, n_instances) tensor of indexed subsamples.
  template <typename T, typename EngineT>
  Tensor<PointCloud<T>> sample_subsets_from_probabilities(const PointCloud<T>& R,
                                                          const Tensor<T>& probabilities,
                                                          size_t sampleSize, size_t nInstances,
                                                          bool replace,
                                                          const RandomGenerator<EngineT>& gen,
                                                          Executor& exec)
  {
    detail::validate_reference(R, sampleSize, replace);
    if (probabilities.rank() != 2)
      throw std::invalid_argument("probabilities must be a 2-D (n_query, n_reference) array");
    if (probabilities.shape(1) != R.n_points())
      throw std::invalid_argument("probabilities must have one column per reference point");

    return detail::draw_subsets_from_weights(R, probabilities, sampleSize, nInstances, replace, gen, exec);
  }

}

#endif
