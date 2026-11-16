#ifndef STABLEBEAR_BARS_WITH_EDGES_H
#define STABLEBEAR_BARS_WITH_EDGES_H

#include "barcode.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace sb::ph
{
  namespace detail
  {
    /// The points {a, b, c, d} of the critical edges of a bar: its birth is
    /// the distance between points a and b, and its death that between c
    /// and d. -1 where an endpoint has no edge: a birth at 0 in dimension
    /// 0, and an infinite death.
    using CriticalEdges = std::array<std::ptrdiff_t, 4>;
  }

  /// A barcode with the critical edges of each of its bars. What the ufunc
  /// operations that report critical edges give per element.
  template <typename T>
  struct BarsWithEdges
  {
    Barcode<T> barcode;
    std::vector<detail::CriticalEdges> edges;
  };
}

#endif // STABLEBEAR_BARS_WITH_EDGES_H
