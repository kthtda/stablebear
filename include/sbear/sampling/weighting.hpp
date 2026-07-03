#ifndef STABLEBEAR_SAMPLING_WEIGHTING_H
#define STABLEBEAR_SAMPLING_WEIGHTING_H

#include "weighted_draw.hpp"

#include "../distance_matrix.hpp"
#include "../executor.hpp"
#include "../point_cloud.hpp"
#include "../tensor.hpp"
#include "../walk.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sb::sampling
{

  // ===========================================================================
  // Sampling weights: how each reference point earns its draw probability.
  //
  // A filter maps a (query point, reference point) pair to a scalar; a
  // distribution maps a filter value to a non-negative weight. The samplers
  // take these as plain functors, so any callable with the matching signature
  // works just as well as the built-ins. A "point" is identified by the cloud
  // it belongs to and its row index, e.g. filter(query, i, reference, r).
  // ===========================================================================

  /// Euclidean distance between point @p queryIdx of @p query and point
  /// @p refIdx of @p reference. Callers are responsible for validating that
  /// @p query and @p reference share a dimension (the samplers do, once,
  /// before the per-pair walk) so this hot-path operator stays branch-free.
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

  /// Uniform distribution over a distance band [low, high]: weight 1 when the
  /// filter value lies in the band, 0 otherwise. With the "distance" filter this
  /// samples uniformly from a region around the query point — a disk
  /// (low = 0), a circle/annulus (0 < low < high), or the whole cloud
  /// (low = 0, high = +inf, the default). Member names match the Python
  /// Uniform(low, high) spec.
  template <typename T>
  struct Uniform
  {
    T low = T(0);
    T high = std::numeric_limits<T>::infinity();

    T operator()(T v) const { return (v >= low && v <= high) ? T(1) : T(0); }
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

    /// Evaluate @p distribution(@p filter(query, reference)) for every
    /// (query, reference) pair into an (n_query, n_reference) weight matrix.
    template <typename T, typename FilterF, typename DistF>
    Tensor<T> compute_weights(const PointCloud<T>& reference, const PointCloud<T>& query,
                              FilterF filter, DistF distribution, Executor& exec)
    {
      Tensor<T> weights({query.n_points(), reference.n_points()});
      parallel_walk(weights,
          [&weights, &reference, &query, filter, distribution](const std::vector<size_t>& idx) {
        weights(idx) = distribution(filter(query, idx[0], reference, idx[1]));
      }, exec);
      return weights;
    }

    /// Evaluate @p distribution of the precomputed distance source(query[qi], j)
    /// for every (query point, reference point) pair into a weight matrix.
    template <typename T, typename DistF>
    Tensor<T> compute_weights_distmat(const DistanceMatrix<T>& source,
                                      const Tensor<uint64_t>& query, DistF distribution,
                                      Executor& exec)
    {
      Tensor<T> weights({query.shape(0), source.size()});
      parallel_walk(weights, [&weights, &source, &query, distribution](const std::vector<size_t>& idx) {
        const size_t queryRow = static_cast<size_t>(query({idx[0]}));
        weights(idx) = distribution(source(queryRow, idx[1]));
      }, exec);
      return weights;
    }

    /// Prepare every row of a freshly computed weight matrix for drawing (see
    /// prepare_weight_row): validation and, when sampling with replacement
    /// (@p toCdf), in-place conversion to row CDFs. Returns the per-query
    /// eligible counts. Blocking and parallel over the query points — this
    /// runs on the caller's thread next to compute_weights, so the async draw
    /// task stays single-phase.
    template <typename T>
    Tensor<size_t> prepare_weight_matrix(Tensor<T>& weights, bool toCdf, Executor& exec)
    {
      // weight_row() addresses rows as contiguous spans; a freshly computed
      // weight matrix always is contiguous, but normalize to be safe.
      if (!weights.is_contiguous())
        weights = weights.copy();

      Tensor<size_t> nEligible({weights.shape(0)});
      // get(), not wait(): a validation error thrown on a worker (negative or
      // NaN weight) must resurface here, at the caller.
      parallel_walk_async(nEligible, [&weights, &nEligible, toCdf](const std::vector<size_t>& idx) {
        nEligible(idx) = prepare_weight_row(weight_row(weights, idx[0]), toCdf);
      }, exec).get();
      return nEligible;
    }

  } // namespace detail

}

#endif
