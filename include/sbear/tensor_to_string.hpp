#ifndef STABLEBEAR_PRINT_TENSOR_H
#define STABLEBEAR_PRINT_TENSOR_H

#include "tensor.hpp"

#include <string>
#include <sstream>
#include <vector>

namespace sb
{

  namespace detail
  {
    template <typename T>
    void print_tensor_recursive(const T& tensor, std::vector<size_t>& indices, std::ostream& os)
    {
      auto const & shape = tensor.shape();

      size_t current_dim = indices.size();
      size_t dim_size = shape[current_dim];

      os << "[";
      for (size_t i = 0; i < dim_size; ++i) {
        indices.push_back(i); // Step "into" the next dimension

        if (indices.size() == shape.size()) {
          // Base Case: We have a full set of indices, get the value
          os << tensor(indices);
        } else {
          // Recursive Case: Move to the next nested level
          print_tensor_recursive(tensor, indices, os);
        }

        indices.pop_back(); // Step "out" to try the next index at this level

        if (i < dim_size - 1) {
          os << ", ";
          // Pretty-printing: Add newlines for higher-order dimensions
          if (shape.size() - indices.size() > 1) {
            os << "\n" << std::string(indices.size() + 1, ' ');
          }
        }
      }
      os << "]";
    }
  }

  /**
   *
   * @tparam T
   * @param tensor
   * @param os
   */
  template <typename T> requires IsTensor<T>
  void print_tensor(const T& tensor, std::ostream& os)
  {
    std::vector<size_t> indices;
    indices.reserve(tensor.shape().size());
    detail::print_tensor_recursive(tensor, indices, os);
  }

  template <typename T> requires IsTensor<T>
  std::string tensor_to_string(const T& tensor)
  {
    std::stringstream ss;
    print_tensor(tensor, ss);
    return ss.str();
  }

}

#endif //STABLEBEAR_PRINT_TENSOR_H
