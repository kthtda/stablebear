#ifndef STABLEBEAR_SPECIAL_TENSORS_H
#define STABLEBEAR_SPECIAL_TENSORS_H

#include "config.hpp"
#include "tensor.hpp"

#include <type_traits>
#include <algorithm>

namespace sb
{
  /**
   * Creates a tensor whose \f$ (i_1,i_2,\ldots,i_d) \f$-th element is given by the formula
   * \f$ a_{i_1,i_2,\ldots,i_d} = \sum_{j=1}^d i_j \times 10^{d-j} \f$, where \f$ d \f$ is the dimension of the tensor.
   * For example, if the requested shape is \f$ \begin{pmatrix} 2, 3 \end{pmatrix} \f$, we get the tensor
   * \f[
   *  A = \begin{pmatrix}
   *    11 & 12 & 13 \\
   *    21 & 22 & 23
   *  \end{pmatrix}.
   * \f]
   *
   * @tparam T Element type (must be an arithmetic type, e.g., `int` or `float32_t`, etc.)
   * @param shape Shape of the returned tensor
   * @return `Tensor<T>` of shape `shape` with values as described above.
   */
  template <typename T> requires Arithmetic<T>
  Tensor<T> mapping_tensor(const std::vector<size_t>& shape)
  {
    Tensor<T> ret{shape};
    if (shape.empty())
    {
      return ret;
    }
    std::vector<T> multiples(shape.size(), static_cast<T>(10));
    multiples.back() = static_cast<T>(1);
    std::partial_sum(multiples.rbegin() + 1, multiples.rend(), multiples.rbegin() + 1, std::multiplies<>());

    sb::walk(ret, [&ret, &multiples](const std::vector<size_t>& idx) {
      auto val = std::inner_product(idx.begin(), idx.end(), multiples.begin(), 0_uz);
      ret(idx) = val;
    });

    return ret;
  }

}

#endif //STABLEBEAR_SPECIAL_TENSORS_H
