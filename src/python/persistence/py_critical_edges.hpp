#ifndef STABLEBEAR_PY_CRITICAL_EDGES_H
#define STABLEBEAR_PY_CRITICAL_EDGES_H

#include "../nb.hpp"
#include "../py_np_support.hpp"

#include <sbear/persistence/bars_with_edges.hpp>
#include <sbear/tensor.hpp>

#include <cstdint>
#include <vector>

namespace sb_py
{
  /// A tensor of barcodes with critical edges as arrays: the bars of all
  /// elements as an (m, 2) array, in C order of the elements; the
  /// [start, stop) rows of each element as an array of shape
  /// tensor.shape + (2,); and the critical edges of each bar as an (m, 4)
  /// array. The tensor must be contiguous, as the result of a ufunc is.
  template <typename T>
  nb::tuple bars_with_edges_to_arrays(const sb::Tensor<sb::ph::BarsWithEdges<T>>& tensor)
  {
    const size_t count = tensor.size();
    size_t total = 0;
    for (size_t k = 0; k < count; ++k)
    {
      total += tensor.flat(k).barcode.bars().size();
    }
    std::vector<size_t> enumerationShape(tensor.shape().begin(), tensor.shape().end());
    enumerationShape.push_back(2);
    auto bars = new_array<T>({total, 2});
    auto enumeration = new_array<int64_t>(enumerationShape);
    auto critical = new_array<int64_t>({total, 4});
    size_t row = 0;
    for (size_t k = 0; k < count; ++k)
    {
      const auto& element = tensor.flat(k);
      const auto& elementBars = element.barcode.bars();
      enumeration.data()[2 * k] = static_cast<int64_t>(row);
      for (size_t b = 0; b < elementBars.size(); ++b, ++row)
      {
        bars.data()[2 * row] = elementBars[b].birth;
        bars.data()[2 * row + 1] = elementBars[b].death;
        for (size_t j = 0; j < 4; ++j)
        {
          critical.data()[4 * row + j] = static_cast<int64_t>(element.edges[b][j]);
        }
      }
      enumeration.data()[2 * k + 1] = static_cast<int64_t>(row);
    }
    return nb::make_tuple(bars, enumeration, critical);
  }
}

#endif // STABLEBEAR_PY_CRITICAL_EDGES_H
