#ifndef STABLEBEAR_PERSISTENCE_LINKAGE_HPP
#define STABLEBEAR_PERSISTENCE_LINKAGE_HPP

#include "barcode.hpp"
#include <sbear/concepts.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sb::ph
{
  // Convert a SciPy-format linkage tensor. Empty input produces an empty barcode.
  template <IsTensor TensorT>
  Barcode<typename TensorT::value_type> linkage_to_barcode(const TensorT& matrix, bool reduced = false)
  {
    using T = typename TensorT::value_type;
    if (matrix.rank() != 2 || matrix.shape(1) != 4)
      throw std::invalid_argument("Z must have shape (n - 1, 4)");
    const auto rows = matrix.shape(0);
    if (rows == 0)
      return Barcode<T>{};
    if (rows > (std::numeric_limits<std::size_t>::max() - 1) / 2)
      throw std::invalid_argument("Linkage matrix is too large");

    const auto n = rows + 1;
    std::vector<std::size_t> sizes(2 * rows + 1, 1);
    std::vector<PersistencePair<T>> bars;
    bars.reserve(rows + !reduced);
    T previousHeight = 0;

    for (std::size_t i = 0; i < rows; ++i)
    {
      const T left = matrix(i, 0);
      const T right = matrix(i, 1);
      const T height = matrix(i, 2);
      const T count = matrix(i, 3);
      if (!std::isfinite(left) || !std::isfinite(right) ||
          !std::isfinite(height) || !std::isfinite(count))
        throw std::invalid_argument("Z must contain only finite values");
      if (height < previousHeight)
        throw std::invalid_argument("Merge heights must be nonnegative and nondecreasing");
      previousHeight = height;

      const auto limit = n + i;
      const auto validIndex = [limit](T index) {
        return index >= 0 && std::trunc(index) == index &&
               static_cast<long double>(index) < static_cast<long double>(limit);
      };
      const auto prefix = [i] { return "Row " + std::to_string(i) + ": "; };
      if (!validIndex(left) || !validIndex(right))
        throw std::invalid_argument(prefix() + "cluster indices must reference existing clusters");
      const auto a = static_cast<std::size_t>(left);
      const auto b = static_cast<std::size_t>(right);
      if (a == b || sizes[a] == 0 || sizes[b] == 0)
        throw std::invalid_argument(prefix() + "clusters must be distinct and active");
      const auto size = sizes[a] + sizes[b];
      if (static_cast<long double>(count) != static_cast<long double>(size))
        throw std::invalid_argument(prefix() + "cluster count must equal " + std::to_string(size));
      sizes[limit] = size;
      sizes[a] = sizes[b] = 0;
      if (height > 0)
        bars.emplace_back(T{0}, height);
    }
    if (!reduced)
      bars.emplace_back(T{0}, std::numeric_limits<T>::infinity());
    return Barcode<T>(std::move(bars));
  }
}

#endif
