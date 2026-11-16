#ifndef SB_ALGORITHM_ARGSORT_SEGMENTS_H
#define SB_ALGORITHM_ARGSORT_SEGMENTS_H

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>

namespace sb
{
  /**
   * Write to @p order the positions start, ..., start + values.size() - 1
   * arranged so that the values they refer to increase, where values[i] is
   * the value at position start + i. Equal values keep their order.
   */
  template <typename T, typename IndexT>
  void argsort_segment(std::span<const T> values, IndexT start, std::span<IndexT> order)
  {
    std::iota(order.begin(), order.end(), start);
    std::stable_sort(order.begin(), order.end(), [&values, start](IndexT a, IndexT b) {
      return values[static_cast<size_t>(a - start)] < values[static_cast<size_t>(b - start)];
    });
  }
}

#endif // SB_ALGORITHM_ARGSORT_SEGMENTS_H
