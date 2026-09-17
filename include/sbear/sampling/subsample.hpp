#ifndef STABLEBEAR_SAMPLING_SUBSAMPLE_H
#define STABLEBEAR_SAMPLING_SUBSAMPLE_H

// Per-query subsampling of a reference point cloud or distance matrix — the
// public entry points of the relative-subsampling pipeline:
//
//   1. compute_weights          (weighting.hpp, blocking) evaluate the
//                               filter/distribution functors into an
//                               (n_query, n_reference) weight matrix
//   2. prepare_weight_matrix    (weighting.hpp, blocking) validate each row
//                               once and ready it for drawing
//   3. draw_subsets_from_weights (subsample_task.hpp, async) launch the
//                               stoppable draw task filling the samples tensor
//
// The row-level draw primitives shared by stages 2 and 3 live in
// weighted_draw.hpp. This header adds input validation and calls the stages
// in order; only the draws — the long phase — run asynchronously with
// progress reporting and cancellation.

#include "subsample_task.hpp"
#include "weighting.hpp"

#include "../distance_matrix.hpp"
#include "../executor.hpp"
#include "../point_cloud.hpp"
#include "../random_generator.hpp"
#include "../tensor.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sb::sampling
{

  namespace detail
  {

    // sample_size is a soft maximum (draws shrink to the eligible count without
    // replacement), so no upper bound is validated in either replace mode.
    template <typename T>
    void validate_reference(const PointCloud<T>& reference, size_t sampleSize)
    {
      if (reference.rank() != 2)
        throw std::invalid_argument("reference must be a 2-D (n_points, dim) point cloud");
      if (sampleSize == 0)
        throw std::invalid_argument("sample_size must be positive");
    }

    template <typename T>
    void validate_distmat(const DistanceMatrix<T>& source, size_t sampleSize)
    {
      if (source.size() == 0)
        throw std::invalid_argument("reference distance matrix must be non-empty");
      if (sampleSize == 0)
        throw std::invalid_argument("sample_size must be positive");
    }

  } // namespace detail

  /// Subsample @p reference relative to each point of @p query, weighted by
  /// @p distribution of @p filter per point pair. Returns a SubsampleHandle:
  /// the async draw task plus its (n_query, n_instances) samples tensor of
  /// indexed views into @p reference. @p sampleSize is a maximum — see
  /// detail::draw_indices for the length contract. @p gen advances past the
  /// draw's reserved seed block, so consecutive calls draw fresh samples.
  template <typename T, typename FilterF, typename DistF>
  SubsampleHandle<PointCloud<T>> sample_subsets(const PointCloud<T>& reference,
                                                const PointCloud<T>& query, FilterF filter,
                                                DistF distribution, size_t sampleSize,
                                                size_t nInstances, bool replace,
                                                DefaultRandomGenerator& gen, Executor& exec)
  {
    detail::validate_reference(reference, sampleSize);
    if (query.rank() != 2)
      throw std::invalid_argument("query must be a 2-D (n_points, dim) point cloud");
    if (query.dim() != reference.dim())
      throw std::invalid_argument("reference and query must have the same dimension");

    Tensor<T> weights = detail::compute_weights(reference, query, filter, distribution, exec);
    Tensor<size_t> nEligible = detail::prepare_weight_matrix(weights, replace, exec);

    return detail::draw_subsets_from_weights<PointCloud<T>>(reference, std::move(weights),
        std::move(nEligible), sampleSize, nInstances, replace, gen, exec);
  }

  /// As sample_subsets, but distances come from the precomputed @p source
  /// matrix and @p query holds reference row indices. Each subsample is the
  /// principal submatrix over the drawn points (an indexed DistanceMatrix
  /// view).
  template <typename T, typename DistF>
  SubsampleHandle<DistanceMatrix<T>> sample_subsets_distmat(const DistanceMatrix<T>& source,
                                                            const Tensor<uint64_t>& query,
                                                            DistF distribution, size_t sampleSize,
                                                            size_t nInstances, bool replace,
                                                            DefaultRandomGenerator& gen, Executor& exec)
  {
    detail::validate_distmat(source, sampleSize);
    Tensor<T> weights = detail::compute_weights_distmat(source, query, distribution, exec);
    Tensor<size_t> nEligible = detail::prepare_weight_matrix(weights, replace, exec);
    return detail::draw_subsets_from_weights<DistanceMatrix<T>>(source, std::move(weights),
        std::move(nEligible), sampleSize, nInstances, replace, gen, exec);
  }

}

#endif
