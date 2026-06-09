/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef STABLEBEAR_SYMMETRIC_MATRIX_H
#define STABLEBEAR_SYMMETRIC_MATRIX_H

#include "config.hpp"
#include "concepts.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <sstream>

namespace sb
{
  namespace io::detail
  {
    template <typename MatT>
    MatT read_compressed_matrix(std::istream&);
  }

  /// Lower-triangular compressed symmetric matrix.
  ///
  /// Stores n*(n+1)/2 elements for an n×n symmetric matrix.
  /// Element (i, j) maps to storage index max(i,j)*(max(i,j)+1)/2 + min(i,j).
  template <ArithmeticType T>
  class SymmetricMatrix
  {
  public:
    using value_type = T;

    explicit SymmetricMatrix(size_t n, const T& init = {})
      : m_data(std::make_shared<T[]>(storage_size(n)))
      , m_size(n)
    {
      std::fill(m_data.get(), m_data.get() + storage_size(n), init);
    }

    SymmetricMatrix() : SymmetricMatrix(0) { }

    [[nodiscard]] size_t size() const { return m_size; }
    [[nodiscard]] size_t storage_count() const { return storage_size(m_size); }

    [[nodiscard]] T& operator()(size_t i, size_t j)
    {
      return m_data[compressed_index(i, j)];
    }

    [[nodiscard]] const T& operator()(size_t i, size_t j) const
    {
      return m_data[compressed_index(i, j)];
    }

    [[nodiscard]] bool operator==(const SymmetricMatrix& rhs) const
    {
      if (m_size != rhs.m_size)
        return false;
      return std::equal(m_data.get(), m_data.get() + storage_size(m_size), rhs.m_data.get());
    }

    [[nodiscard]] bool operator!=(const SymmetricMatrix& rhs) const
    {
      if (m_size != rhs.m_size)
        return true;
      return std::mismatch(m_data.get(), m_data.get() + storage_size(m_size), rhs.m_data.get()).first
        != m_data.get() + storage_size(m_size);
    }

    [[nodiscard]] const T* data() const { return m_data.get(); }

    [[nodiscard]] static size_t storage_size(size_t n)
    {
      return n * (n + 1) / 2;
    }

  private:
    [[nodiscard]] T* mutable_data() { return m_data.get(); }

    template <typename MatT>
    friend MatT io::detail::read_compressed_matrix(std::istream&);

    [[nodiscard]] size_t compressed_index(size_t i, size_t j) const
    {
      if (i >= m_size || j >= m_size)
      {
        std::ostringstream oss;
        oss << "SymmetricMatrix index (" << i << ", " << j << ") out of range for matrix of size " << m_size;
        throw std::out_of_range(oss.str());
      }

      auto row = std::max(i, j);
      auto col = std::min(i, j);
      return row * (row + 1) / 2 + col;
    }

    std::shared_ptr<T[]> m_data;
    size_t m_size;
  };

  template <typename T>
  struct is_compressed_matrix<SymmetricMatrix<T>> : std::true_type {};

}

#endif // STABLEBEAR_SYMMETRIC_MATRIX_H
