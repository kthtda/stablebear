#ifndef STABLEBEAR_SAMPLING_WEIGHTED_DRAW_H
#define STABLEBEAR_SAMPLING_WEIGHTED_DRAW_H

#include "../tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace sb::sampling::detail
{

  // ===========================================================================
  // Row-level weighted-draw primitives.
  //
  // Pure functions of a weight/CDF row and a random engine: a row is prepared
  // once per query point (prepare_weight_row) and then drawn from once per
  // subsample. Nothing here knows about tasks, threads or point clouds — the
  // parallel orchestration lives in subsample_task.hpp.
  // ===========================================================================

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

  /// Map a @p target in [0, cdf.back()] to the index whose CDF interval
  /// contains it. uniform_real_distribution may round up to exactly
  /// cdf.back() (LWG 2524); such a target is pulled back inside the range,
  /// so the draw lands on the last positive-weight interval — never past it
  /// onto a trailing zero-weight (ineligible) point.
  template <typename T>
  size_t index_for_target(std::span<const T> cdf, T target)
  {
    if (target >= cdf.back())
      target = std::nextafter(cdf.back(), T(0));
    const auto it = std::upper_bound(cdf.begin(), cdf.end(), target);
    return static_cast<size_t>(it - cdf.begin());
  }

  /// Draw a single reference index from a CDF via binary search.
  template <typename T, typename EngineT>
  size_t draw_one(std::span<const T> cdf, EngineT& engine)
  {
    std::uniform_real_distribution<T> uniform(T(0), cdf.back());
    return index_for_target(cdf, uniform(engine));
  }

  /// Validate a weight row and return its eligible (strictly positive)
  /// count; with @p toCdf, also replace it in place by its prefix sums.
  /// Called once per query point, reused by all of the query's subsamples.
  template <typename T>
  size_t prepare_weight_row(std::span<T> row, bool toCdf)
  {
    size_t nEligible = 0;
    T total = T(0);
    for (T& w : row)
    {
      // Reject negatives rather than counting them ineligible: an invalid
      // row must not become a silent empty draw.
      if (w < T(0))
        throw std::invalid_argument("sampling weights must be non-negative");
      if (w > T(0))
        ++nEligible;
      total += w;
      if (toCdf)
        w = total;  // with replacement the CDF never changes: build it once
    }
    // A row with eligible weights but no positive total (NaN weights) can
    // not be drawn from; an all-zero row is a valid empty region.
    if (nEligible > 0 && !(total > T(0)))
      throw std::invalid_argument("sampling weights must have a positive sum");
    return nEligible;
  }

  /// Draw @p sampleSize reference indices with replacement from a prepared
  /// CDF row (repeats fill the sample). The shared row is not modified.
  template <typename T, typename EngineT>
  Tensor<uint64_t> draw_with_replacement(std::span<const T> cdf, size_t sampleSize,
                                         EngineT& engine)
  {
    Tensor<uint64_t> drawn({sampleSize});
    for (size_t drawIdx = 0; drawIdx < sampleSize; ++drawIdx)
      drawn({drawIdx}) = static_cast<uint64_t>(draw_one(cdf, engine));
    return drawn;
  }

  /// Draw @p nDraws *distinct* reference indices from a raw weight row.
  /// Requires nDraws <= the row's eligible count, which keeps the CDF total
  /// positive throughout.
  template <typename T, typename EngineT>
  Tensor<uint64_t> draw_without_replacement(std::span<const T> weights, size_t nDraws,
                                            EngineT& engine)
  {
    std::vector<T> remaining(weights.begin(), weights.end());
    std::vector<T> cdf;
    Tensor<uint64_t> drawn({nDraws});
    for (size_t drawIdx = 0; drawIdx < nDraws; ++drawIdx)
    {
      build_cdf(cdf, remaining);
      const size_t refIdx = draw_one(std::span<const T>(cdf), engine);
      drawn({drawIdx}) = static_cast<uint64_t>(refIdx);
      remaining[refIdx] = T(0);  // a drawn point cannot be drawn again
    }
    return drawn;
  }

  /// The reference indices of one subsample, drawn from a prepared row.
  /// @p sampleSize is a maximum — the result has length
  ///   - 0 when @p nEligible is 0 (the query's region holds no points),
  ///   - min(sampleSize, nEligible) without replacement,
  ///   - sampleSize with replacement (repeats fill the sample).
  template <typename T, typename EngineT>
  Tensor<uint64_t> draw_indices(std::span<const T> row, size_t nEligible, size_t sampleSize,
                                bool replace, EngineT& engine)
  {
    if (nEligible == 0)
      return Tensor<uint64_t>({0});  // empty region -> length-0 subsample
    return replace
        ? draw_with_replacement(row, sampleSize, engine)
        : draw_without_replacement(row, std::min(sampleSize, nEligible), engine);
  }

  /// The contiguous row of one query point in a (n_query, n_reference)
  /// weight matrix (prepare_weight_matrix normalizes contiguity once).
  template <typename T>
  std::span<T> weight_row(const Tensor<T>& weights, size_t queryIdx)
  {
    const size_t nReference = weights.shape(1);
    return {weights.data() + queryIdx * nReference, nReference};
  }

}

#endif
