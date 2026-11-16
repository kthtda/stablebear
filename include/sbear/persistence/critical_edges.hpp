#ifndef STABLEBEAR_CRITICAL_EDGES_H
#define STABLEBEAR_CRITICAL_EDGES_H

// Includes Ripser, whose header can be part of one translation unit only.

#include "../ufunc.hpp"

#include "barcode.hpp"
#include "bars_with_edges.hpp"
#include "compute_persistence.hpp"

#include <span>
#include <utility>
#include <vector>

namespace sb::ph
{
  /// persistence_with_critical_edges as a ufunc operation on a point cloud
  /// or a distance matrix, with extra dimensions (maxDim + 1,): out[i] is
  /// the barcode of dimension i with its critical edges.
  template <typename ElemT, typename T = typename ElemT::value_type>
  struct PersistenceCriticalEdgesOperation
  {
    using input_type = ElemT;
    using output_type = BarsWithEdges<T>;

    size_t maxDim = 1;
    bool reducedHomology = false;

    [[nodiscard]] std::vector<size_t> extra_dims() const { return {maxDim + 1}; }

    void operator()(const ElemT& element, std::span<BarsWithEdges<T>> out) const
    {
      std::vector<Barcode<T>> barcodes;
      std::vector<std::vector<detail::CriticalEdges>> edges;
      persistence_with_critical_edges(element, maxDim, reducedHomology, barcodes, edges);
      for (size_t i = 0; i < out.size(); ++i)
      {
        out[i] = BarsWithEdges<T>{std::move(barcodes[i]), std::move(edges[i])};
      }
    }
  };
}

#endif // STABLEBEAR_CRITICAL_EDGES_H
