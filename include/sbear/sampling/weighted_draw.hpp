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
  // once per query/distribution pair (log_probabilities_to_cdf) and then drawn
  // from once per subsample — inverse-CDF draws with replacement, reservoir
  // sampling by exponential keys without. Nothing here knows about tasks,
  // threads or point clouds — the parallel orchestration lives in
  // subsample_relative.hpp.
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

  /// Replace normalized log probabilities (finite or -inf, maximum 0) with
  /// the prefix sums of their probabilities. An all -inf row is a valid empty
  /// region and must not be drawn from.
  template <typename T>
  void log_probabilities_to_cdf(std::span<T> row)
  {
    T total = T(0);
    for (T &w : row)
    {
      total += std::exp(w);
      w = total;
    }
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

  /// Draw @p nDraws *distinct* reference indices from a log-weight row by
  /// weighted reservoir sampling (Efraimidis-Spirakis) in exponential form:
  /// each eligible point gets an independent key Exp(1) / weight, and the
  /// nDraws smallest keys are the sample. Competing exponential clocks make
  /// the smallest key fall on point i with probability w_i / sum(w), and by
  /// memorylessness the same holds among the remaining points, so ascending
  /// key order has exactly the distribution of drawing sequentially without
  /// replacement — at one pass and one random number per eligible point
  /// instead of a CDF rebuild per draw. Keys are compared as logarithms so
  /// weights that would underflow keep positive support. The row holds only
  /// eligible (finite) log weights; @p sourceIndices maps each entry back to
  /// its reference index. Requires nDraws <= logWeights.size().
  template <typename T, typename EngineT>
  Tensor<uint64_t> draw_without_replacement(std::span<const T> logWeights,
                                          std::span<const uint64_t> sourceIndices,
                                          size_t nDraws, EngineT &engine,
                                          std::vector<std::pair<T, uint64_t>>& keyed)
  {
    std::exponential_distribution<T> exponential(T(1));
    keyed.clear();
    keyed.reserve(logWeights.size());
    for (size_t i = 0; i < logWeights.size(); ++i)
    {
      T clock;
      do
      {
        clock = exponential(engine);
      } while (clock == T(0) || !std::isfinite(clock));
      keyed.emplace_back(std::log(clock), static_cast<uint64_t>(i));
    }
    // Compare differences rather than adding noise to an enormous log weight:
    // even equal far-tail weights must retain their independent random clocks.
    const auto earlier = [&](const auto& a, const auto& b) {
      const T weightDifference = logWeights[a.second] - logWeights[b.second];
      const T clockDifference = a.first - b.first;
      if (weightDifference == clockDifference)
        return a.second < b.second;
      return weightDifference > clockDifference;
    };
    std::partial_sort(keyed.begin(), keyed.begin() + static_cast<std::ptrdiff_t>(nDraws), keyed.end(), earlier);

    Tensor<uint64_t> drawn({nDraws});
    for (size_t drawIdx = 0; drawIdx < nDraws; ++drawIdx)
    {
      drawn(drawIdx) = sourceIndices[keyed[drawIdx].second];
    }
    return drawn;
  }

} // namespace sb::sampling::detail

#endif
