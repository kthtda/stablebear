#ifndef STABLEBEAR_SAMPLING_WEIGHTING_H
#define STABLEBEAR_SAMPLING_WEIGHTING_H

#include "weighted_draw.hpp"

#include "../distance_matrix.hpp"
#include "../executor.hpp"
#include "../point_cloud.hpp"
#include "../tensor.hpp"
#include "../walk.hpp"

#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace sb::sampling
{

  // ===========================================================================
  // Sampling weights: how each reference point earns its draw probability.
  //
  // A filter maps a (query point, reference point) pair to a scalar; a
  // distribution maps that value to a non-negative weight. Both are plain
  // functors, so custom callables work like the built-ins.
  //
  // A distribution may additionally provide `T log_weight(T v) const`
  // (= log(operator()(v)), see Gaussian): its rows are then evaluated in log
  // space and finished with a max-shift exponentiation, so weights that
  // mathematically never vanish cannot underflow to an all-zero row. Either
  // way, code downstream of compute_weights only ever sees plain weights.
  // ===========================================================================

  /// Euclidean distance between query point @p queryIdx and reference point
  /// @p refIdx. The samplers validate the shared dimension once up front, so
  /// this hot-path operator stays branch-free.
  template <typename T>
  struct EuclideanDistance
  {
    T operator()(const PointCloud<T>& query, size_t queryIdx,
                 const PointCloud<T>& reference, size_t refIdx) const
    {
      T acc = T(0);
      for (size_t k = 0; k < query.dim(); ++k)
      {
        T d = query(queryIdx, k) - reference(refIdx, k);
        acc += d * d;
      }
      return std::sqrt(acc);
    }
  };

  /// Weight 1 when the filter value lies in [low, high], 0 outside. With the
  /// distance filter that is a disk (low = 0), an annulus, or the whole cloud
  /// (the default, high = +inf). Member names match the Python spec.
  template <typename T>
  struct Uniform
  {
    T low = T(0);
    T high = std::numeric_limits<T>::infinity();

    T operator()(T v) const { return (v >= low && v <= high) ? T(1) : T(0); }
  };

  /// Unnormalized Gaussian exp(-((v - mean) / sigma)^2 / 2) of the filter
  /// value, offering the log-space channel: a raw exp would underflow to an
  /// artificial all-zero row once every reference point sits ~38 sigma
  /// (float64; ~14 float32) from the mean, while the max-shifted exp samples
  /// this exact Gaussian at any distance.
  template <typename T>
  struct Gaussian
  {
    T mean = T(0);
    T sigma = T(1);

    T operator()(T v) const { return std::exp(log_weight(v)); }

    T log_weight(T v) const
    {
      T d = (v - mean) / sigma;
      return T(-0.5) * d * d;
    }
  };

  namespace detail
  {

    /// Detects the optional log-space channel: a `log_weight(v)` member
    /// returning log(operator()(v)). Plain functors are evaluated directly.
    template <typename DistF, typename T>
    concept has_log_weight = requires(const DistF dist, T v) {
      { dist.log_weight(v) } -> std::convertible_to<T>;
    };

    /// In-place max-shift exponentiation of a log-weight row:
    /// w_i = exp(e_i - max_j e_j). The shift cancels on normalization, and
    /// the best point's weight is exactly 1, so the row can never underflow
    /// to all-zero.
    template <typename T>
    void log_row_to_weights(std::span<T> row)
    {
      T maxLog = -std::numeric_limits<T>::infinity();
      for (T e : row)
        maxLog = std::max(maxLog, e);  // drops NaNs; exp keeps them NaN below
      if (!(maxLog > -std::numeric_limits<T>::infinity()))
        maxLog = T(0);  // all -inf (empty region): avoid (-inf) - (-inf) = NaN
      for (T& e : row)
        e = std::exp(e - maxLog);
    }

    /// Max-shift every row of a freshly computed (hence contiguous)
    /// log-weight matrix into plain weights, in parallel.
    template <typename T>
    void exponentiate_log_rows(Tensor<T>& weights, Executor& exec)
    {
      Tensor<size_t> rowIndices({weights.shape(0)});
      parallel_walk(rowIndices, [&weights](const std::vector<size_t>& idx) {
        const std::span<T> row = weight_row(weights, idx[0]);
        log_row_to_weights(row);
      }, exec);
    }

    /// One weight-matrix entry: the log-weight when the distribution offers
    /// the channel (rows are then finished by exponentiate_log_rows), else
    /// the weight itself.
    template <typename T, typename DistF>
    T weight_entry(const DistF& distribution, T value)
    {
      if constexpr (has_log_weight<DistF, T>)
        return distribution.log_weight(value);
      else
        return distribution(value);
    }

    /// The (n_query, n_reference) weight matrix: @p distribution of
    /// @p filter(query, reference) per pair. Always returns plain
    /// non-negative weights.
    template <typename T, typename FilterF, typename DistF>
    Tensor<T> compute_weights(const PointCloud<T>& reference, const PointCloud<T>& query,
                              FilterF filter, DistF distribution, Executor& exec)
    {
      Tensor<T> weights({query.n_points(), reference.n_points()});
      parallel_walk(weights,
          [&weights, &reference, &query, filter, distribution](const std::vector<size_t>& idx) {
        const T filterValue = filter(query, idx[0], reference, idx[1]);
        weights(idx) = weight_entry(distribution, filterValue);
      }, exec);
      if constexpr (has_log_weight<DistF, T>)
        exponentiate_log_rows(weights, exec);
      return weights;
    }

    /// As compute_weights, with filter values read from the precomputed
    /// distance matrix @p source; @p query holds reference row indices.
    template <typename T, typename DistF>
    Tensor<T> compute_weights_distmat(const DistanceMatrix<T>& source,
                                      const Tensor<uint64_t>& query, DistF distribution,
                                      Executor& exec)
    {
      Tensor<T> weights({query.shape(0), source.size()});
      parallel_walk(weights, [&weights, &source, &query, distribution](const std::vector<size_t>& idx) {
        const size_t queryRow = static_cast<size_t>(query({idx[0]}));
        const T filterValue = source(queryRow, idx[1]);
        weights(idx) = weight_entry(distribution, filterValue);
      }, exec);
      if constexpr (has_log_weight<DistF, T>)
        exponentiate_log_rows(weights, exec);
      return weights;
    }

    /// Ready every row for drawing (see prepare_weight_row) and return the
    /// per-query eligible counts. Blocking, so the async draw task stays
    /// single-phase.
    template <typename T>
    Tensor<size_t> prepare_weight_matrix(Tensor<T>& weights, bool toCdf, Executor& exec)
    {
      // weight_row() needs contiguous rows; fresh matrices are, but normalize.
      if (!weights.is_contiguous())
        weights = weights.copy();

      Tensor<size_t> nEligible({weights.shape(0)});
      // get(), not wait(): a validation error thrown on a worker (negative or
      // NaN weight) must resurface here, at the caller.
      parallel_walk_async(nEligible, [&weights, &nEligible, toCdf](const std::vector<size_t>& idx) {
        const std::span<T> row = weight_row(weights, idx[0]);
        nEligible(idx) = prepare_weight_row(row, toCdf);
      }, exec).get();
      return nEligible;
    }

  } // namespace detail

}

#endif
