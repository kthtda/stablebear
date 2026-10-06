#ifndef STABLEBEAR_SAMPLING_PARTIAL_POLICY_HPP
#define STABLEBEAR_SAMPLING_PARTIAL_POLICY_HPP

#include <algorithm>
#include <cstddef>

namespace sb::sampling
{
  enum class PartialPolicy
  {
    Disallow,
    Keep,
    Drop
  };

  /// True when PartialPolicy::Disallow forbids sampling @p nPoints draws from
  /// @p available points: with replacement, one point suffices.
  inline bool is_insufficient(PartialPolicy policy, bool replace, size_t nPoints, size_t available)
  {
    return policy == PartialPolicy::Disallow && (available == 0 || (!replace && available < nPoints));
  }

  /// Number of draws for one sample before duplicate removal; zero marks an
  /// empty sample.
  inline size_t sample_count(PartialPolicy policy, bool replace, size_t nPoints, size_t available)
  {
    if (available == 0 || (policy == PartialPolicy::Drop && available < nPoints))
      return 0;
    return replace ? nPoints : std::min(nPoints, available);
  }
}

#endif
