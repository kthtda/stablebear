#ifndef STABLEBEAR_KERNEL_CRITICAL_EDGES_H
#define STABLEBEAR_KERNEL_CRITICAL_EDGES_H

#include "../ufunc.hpp"

#include "barcode.hpp"
#include "bars_with_edges.hpp"
#include "compute_homological_kernel.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace sb::ph
{
  /// homological_kernel_with_critical_edges as a ufunc operation on a pair
  /// of point clouds or of distance matrices: the first of the pair gives
  /// the deaths, and the second the births.
  template <typename ElemT, typename T = typename ElemT::value_type>
  struct KernelCriticalEdgesOperation
  {
    using input_type = ElementPair<ElemT, ElemT>;
    using output_type = BarsWithEdges<T>;

    [[nodiscard]] std::vector<size_t> extra_dims() const { return {}; }

    void operator()(const input_type& pair, std::span<BarsWithEdges<T>> out) const
    {
      Barcode<T> barcode;
      std::vector<std::array<size_t, 4>> edges;
      homological_kernel_with_critical_edges(pair.first, pair.second, barcode, edges);
      std::vector<detail::CriticalEdges> signedEdges;
      signedEdges.reserve(edges.size());
      for (const auto& e : edges)
      {
        signedEdges.push_back({static_cast<std::ptrdiff_t>(e[0]), static_cast<std::ptrdiff_t>(e[1]),
                               static_cast<std::ptrdiff_t>(e[2]), static_cast<std::ptrdiff_t>(e[3])});
      }
      out[0] = BarsWithEdges<T>{std::move(barcode), std::move(signedEdges)};
    }
  };
}

#endif // STABLEBEAR_KERNEL_CRITICAL_EDGES_H
