#ifndef STABLEBEAR_BARCODE_TENSOR_H
#define STABLEBEAR_BARCODE_TENSOR_H

#include "barcode.hpp"
#include "../tensor.hpp"
#include "../walk.hpp"

#include <stdexcept>

namespace sb::ph
{
  /**
   * Test whether two aligned tensors of barcodes are elementwise isomorphic.
   *
   * The outer tensor shapes must be identical. Each corresponding pair of
   * barcodes is compared as an order-independent multiset of bars using
   * `Barcode::is_isomorphic_to` and the supplied endpoint tolerances.
   *
   * @throws std::invalid_argument if the outer tensor shapes differ.
   */
  template <typename T, TensorProperties LhsProperties,
    TensorProperties RhsProperties>
  [[nodiscard]] bool are_isomorphic(
    const Tensor<Barcode<T>, LhsProperties>& lhs,
    const Tensor<Barcode<T>, RhsProperties>& rhs,
    double atol = 1e-8, double rtol = 1e-5)
  {
    if (lhs.shape() != rhs.shape())
    {
      throw std::invalid_argument(
        "Barcode tensor shapes must match, got "
        + shape_to_string(lhs.shape()) + " and "
        + shape_to_string(rhs.shape()));
    }

    bool result = true;
    walk(lhs, [&](const std::vector<size_t>& index) {
      result = lhs(index).is_isomorphic_to(rhs(index), atol, rtol);
      return result;
    });
    return result;
  }
}

#endif // STABLEBEAR_BARCODE_TENSOR_H
