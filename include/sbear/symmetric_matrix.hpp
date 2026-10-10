#ifndef STABLEBEAR_SYMMETRIC_MATRIX_H
#define STABLEBEAR_SYMMETRIC_MATRIX_H

#include "config.hpp"
#include "concepts.hpp"
#include "fixed_rank_tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <sstream>

namespace sb
{
  class SymmetricMatrixLayout
  {
  public:
    using shape_type = std::array<size_t, 2>;

    SymmetricMatrixLayout() = default;
    explicit SymmetricMatrixLayout(size_t size) : m_size(size) { }

    [[nodiscard]] shape_type shape() const noexcept { return {m_size, m_size}; }
    [[nodiscard]] size_t storage_size() const noexcept
    {
      return m_size * (m_size + 1) / 2;
    }
    [[nodiscard]] size_t offset(const shape_type& indices) const
    {
      if (indices[0] >= m_size || indices[1] >= m_size)
        throw std::out_of_range("SymmetricMatrix index out of range");
      const auto row = std::max(indices[0], indices[1]);
      const auto col = std::min(indices[0], indices[1]);
      return row * (row + 1) / 2 + col;
    }
    [[nodiscard]] size_t size() const noexcept { return m_size; }

  private:
    size_t m_size = 0;
  };

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
      : m_storage(SymmetricMatrixLayout(n), init)
    { }

    SymmetricMatrix() : SymmetricMatrix(0) { }

    /// Return an independent deep copy. The implicit copy shares the
    /// std::shared_ptr buffer (view-like), which is relied on internally; this
    /// is used when a matrix must not alias its source (e.g. a tensor cell).
    [[nodiscard]] SymmetricMatrix copy() const
    {
      SymmetricMatrix result;
      result.m_storage = m_storage.copy();
      return result;
    }

    [[nodiscard]] size_t size() const { return m_storage.layout().size(); }
    [[nodiscard]] size_t storage_count() const { return storage_size(size()); }

    [[nodiscard]] T& operator()(size_t i, size_t j)
    {
      return m_storage(i, j);
    }

    [[nodiscard]] const T& operator()(size_t i, size_t j) const
    {
      return m_storage(i, j);
    }

    [[nodiscard]] bool operator==(const SymmetricMatrix& rhs) const
    {
      return size() == rhs.size()
        && std::equal(data(), data() + storage_count(), rhs.data());
    }

    [[nodiscard]] bool operator!=(const SymmetricMatrix& rhs) const
    {
      return !(*this == rhs);
    }

    [[nodiscard]] const T* data() const { return m_storage.storage_data(); }

    [[nodiscard]] static size_t storage_size(size_t n)
    {
      return n * (n + 1) / 2;
    }

  private:
    FixedRankTensor<T, 2, SymmetricMatrixLayout> m_storage;
  };

  template <typename T>
  struct is_compressed_matrix<SymmetricMatrix<T>> : std::true_type {};

}

#endif // STABLEBEAR_SYMMETRIC_MATRIX_H
