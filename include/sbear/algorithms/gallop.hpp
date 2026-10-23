#ifndef SB_ALGORITHMS_GALLOP_H
#define SB_ALGORITHMS_GALLOP_H

#include <algorithm>
#include <iterator>

namespace sb
{
  /// std::upper_bound(first, last, value) for an ascending range, found by
  /// galloping from @p hint, any position in [first, last]: step 1, 2, 4, ...
  /// elements toward the answer until stepping past it, then bisect the last
  /// step. An answer d elements from the hint costs O(log d) comparisons,
  /// which beats a binary search over the whole range when the hint is close,
  /// as when queries arrive in order or near a known position.
  template <std::random_access_iterator It, typename T>
  [[nodiscard]] It gallop_upper_bound(It first, It last, It hint, const T& value)
  {
    typename std::iterator_traits<It>::difference_type step = 1;
    if (hint == last || value < *hint)
    {
      // The answer is at or before hint.
      auto hi = hint;
      while (hi != first)
      {
        const auto probe = hi - first > step ? hi - step : first;
        if (!(value < *probe))
        {
          return std::upper_bound(probe + 1, hi, value);
        }
        hi = probe;
        step *= 2;
      }
      return first;
    }
    // The answer is after hint.
    auto lo = hint;
    while (true)
    {
      const auto probe = last - lo > step ? lo + step : last;
      if (probe == last || value < *probe)
      {
        return std::upper_bound(lo + 1, probe, value);
      }
      lo = probe;
      step *= 2;
    }
  }
}

#endif // SB_ALGORITHMS_GALLOP_H
