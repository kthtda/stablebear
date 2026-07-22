#ifndef STABLEBEAR_SAMPLING_WEIGHTED_DRAW_H
#define STABLEBEAR_SAMPLING_WEIGHTED_DRAW_H

#include "../tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sb::sampling::detail
{

  // ===========================================================================
  // Row-level weighted-draw primitives.
  //
  // Pure functions of a weight/CDF row and a random engine: a row is prepared
  // once per query point (prepare_weight_row) and then drawn from once per
  // subsample — inverse-CDF draws with replacement, reservoir sampling by
  // exponential keys without. Nothing here knows about tasks, threads or
  // point clouds — the parallel orchestration lives in subsample.hpp.
  // ===========================================================================

  /// Map a @p target in [0, cdf.back()] to the index whose CDF interval
  /// contains it. uniform_real_distribution may round up to exactly
  /// cdf.back() (LWG 2524); such a target is pulled back inside the range,
  /// so the draw lands on the last positive-weight interval — never past it
  /// onto a trailing zero-weight (ineligible) point.
  template <typename T>
  size_t index_for_target(std::span<const T> cdf, T target)
  {
    if (target >= cdf.back())
    {
      target = std::nextafter(cdf.back(), T(0));
    }
    const auto it = std::upper_bound(cdf.begin(), cdf.end(), target);
    return static_cast<size_t>(it - cdf.begin());
  }

  /// Draw a single reference index from a CDF via binary search.
  template <typename T, typename EngineT>
  size_t draw_one(std::span<const T> cdf, EngineT &engine)
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
    for (T &w : row)
    {
      // Reject negatives rather than counting them ineligible: an invalid
      // row must not become a silent empty draw.
      if (w < T(0))
      {
        throw std::invalid_argument("sampling weights must be non-negative");
      }
      if (w > T(0))
      {
        ++nEligible;
      }
      total += w;
      if (toCdf)
      {
        w = total; // with replacement the CDF never changes: build it once
      }
    }
    // A row with eligible weights but no positive total (NaN weights) can
    // not be drawn from; an all-zero row is a valid empty region.
    if (nEligible > 0 && !(total > T(0)))
    {
      throw std::invalid_argument("sampling weights must have a positive sum");
    }
    return nEligible;
  }

  /// Draw @p sampleSize reference indices with replacement from a prepared
  /// CDF row (repeats fill the sample). The shared row is not modified.
  template <typename T, typename EngineT>
  Tensor<uint64_t> draw_with_replacement(std::span<const T> cdf, size_t sampleSize, EngineT &engine)
  {
    Tensor<uint64_t> drawn({sampleSize});
    for (size_t drawIdx = 0; drawIdx < sampleSize; ++drawIdx)
    {
      drawn(drawIdx) = static_cast<uint64_t>(draw_one(cdf, engine));
    }
    return drawn;
  }

  /// Draw @p nDraws *distinct* reference indices from a raw weight row by
  /// weighted reservoir sampling (Efraimidis-Spirakis) in exponential form:
  /// each eligible point gets an independent key Exp(1) / weight, and the
  /// nDraws smallest keys are the sample. Competing exponential clocks make
  /// the smallest key fall on point i with probability w_i / sum(w), and by
  /// memorylessness the same holds among the remaining points, so ascending
  /// key order has exactly the distribution of drawing sequentially without
  /// replacement — at one pass and one random number per eligible point
  /// instead of a CDF rebuild per draw. Requires nDraws <= the row's
  /// eligible count.
  template <typename T, typename EngineT>
  Tensor<uint64_t> draw_without_replacement(std::span<const T> weights, size_t nDraws, EngineT &engine)
  {
    std::exponential_distribution<T> exponential(T(1));
    std::vector<std::pair<T, uint64_t>> keyed;
    keyed.reserve(weights.size());
    for (size_t refIdx = 0; refIdx < weights.size(); ++refIdx)
    {
      if (weights[refIdx] > T(0))
      {
        keyed.emplace_back(exponential(engine) / weights[refIdx], static_cast<uint64_t>(refIdx));
      }
    }
    std::partial_sort(keyed.begin(), keyed.begin() + static_cast<std::ptrdiff_t>(nDraws), keyed.end());

    Tensor<uint64_t> drawn({nDraws});
    for (size_t drawIdx = 0; drawIdx < nDraws; ++drawIdx)
    {
      drawn(drawIdx) = keyed[drawIdx].second;
    }
    return drawn;
  }

  /// The reference indices of one subsample, drawn from a prepared row.
  /// @p sampleSize is a maximum — the result has length
  ///   - 0 when @p nEligible is 0 (the query's region holds no points),
  ///   - min(sampleSize, nEligible) without replacement,
  ///   - sampleSize with replacement (repeats fill the sample).
  template <typename T, typename EngineT>
  Tensor<uint64_t> draw_indices(
      std::span<const T> row, size_t nEligible, size_t sampleSize, bool replace, EngineT &engine)
  {
    if (nEligible == 0)
    {
      return Tensor<uint64_t>({0}); // empty region -> length-0 subsample
    }
    return replace ? draw_with_replacement(row, sampleSize, engine)
                   : draw_without_replacement(row, std::min(sampleSize, nEligible), engine);
  }

} // namespace sb::sampling::detail

#endif
