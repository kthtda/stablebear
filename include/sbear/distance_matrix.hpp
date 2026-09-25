#ifndef STABLEBEAR_DISTANCE_MATRIX_H
#define STABLEBEAR_DISTANCE_MATRIX_H

#include "config.hpp"
#include "fixed_rank_tensor.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <sstream>

namespace sb
{
  class DistanceMatrixLayout
  {
  public:
    using shape_type = std::array<size_t, 2>;
    using index_type = Tensor<uint64_t>;

    DistanceMatrixLayout() = default;
    explicit DistanceMatrixLayout(size_t sourceSize) : m_sourceSize(sourceSize) { }
    DistanceMatrixLayout(size_t sourceSize, index_type indices)
      : m_sourceSize(sourceSize), m_indices(std::move(indices))
    { }

    [[nodiscard]] shape_type shape() const noexcept
    {
      const size_t logicalSize = size();
      return {logicalSize, logicalSize};
    }
    [[nodiscard]] size_t storage_size() const noexcept
    {
      return m_sourceSize * (m_sourceSize - 1) / 2;
    }
    [[nodiscard]] size_t offset(const shape_type& indices) const
    {
      if (indices[0] >= size() || indices[1] >= size())
        throw std::out_of_range("DistanceMatrix index out of range");
      const size_t i = source_index(indices[0]);
      const size_t j = source_index(indices[1]);
      if (i == j)
        throw std::logic_error("DistanceMatrix diagonal has no physical storage");
      const auto row = std::max(i, j);
      const auto col = std::min(i, j);
      return row * (row - 1) / 2 + col;
    }
    [[nodiscard]] size_t size() const noexcept
    {
      return is_indexed() ? m_indices.shape(0) : m_sourceSize;
    }
    [[nodiscard]] size_t source_size() const noexcept { return m_sourceSize; }
    [[nodiscard]] size_t source_index(size_t logicalIndex) const
    {
      return is_indexed()
        ? static_cast<size_t>(m_indices(logicalIndex))
        : logicalIndex;
    }
    [[nodiscard]] bool is_indexed() const noexcept
    {
      return m_indices.rank() == 1;
    }
    [[nodiscard]] const index_type& indices() const noexcept { return m_indices; }

  private:
    size_t m_sourceSize = 0;
    index_type m_indices;
  };

  /// Lower-triangular compressed distance matrix (zero diagonal, nonnegative entries).
  ///
  /// Stores n*(n-1)/2 elements for an n×n symmetric matrix with
  /// implicit zeros on the diagonal.
  /// For i != j, element (i, j) maps to storage index
  /// max(i,j)*(max(i,j)-1)/2 + min(i,j).
  template <ArithmeticType T>
  class DistanceMatrix
  {
  public:
    using value_type = T;
    using index_type = Tensor<uint64_t>;

    class EntryProxy
    {
    public:
      explicit EntryProxy(T* ptr) : m_ptr(ptr) { }

      operator T() const
      {
        if (!m_ptr)
          return T{};
        return *m_ptr;
      }

      EntryProxy& operator=(const T& value)
      {
        if (value < T{})
          throw std::invalid_argument("Distance matrix entries must be nonnegative");
        if (!m_ptr)
        {
          if (value != T{})
            throw std::invalid_argument("Diagonal entries of a distance matrix must be zero");
          return *this;
        }
        *m_ptr = value;
        return *this;
      }

    private:
      T* m_ptr;
    };

    explicit DistanceMatrix(size_t n, const T& init = {})
      : m_storage(DistanceMatrixLayout(n), init)
    {
      if (init < T{})
        throw std::invalid_argument("Distance matrix entries must be nonnegative");
    }

    DistanceMatrix() : DistanceMatrix(0) { }

    /// Return an independent deep copy. The implicit copy shares the
    /// std::shared_ptr buffer (view-like), which is relied on internally; this
    /// is used when a matrix must not alias its source (e.g. a tensor cell).
    [[nodiscard]] DistanceMatrix copy() const
    {
      DistanceMatrix result(size());
      for (size_t i = 1; i < size(); ++i)
      {
        for (size_t j = 0; j < i; ++j)
          result(i, j) = (*this)(i, j);
      }
      return result;
    }

    [[nodiscard]] size_t size() const
    {
      return m_storage.layout().size();
    }
    [[nodiscard]] size_t storage_count() const { return storage_size(size()); }

    [[nodiscard]] bool is_indexed() const
    {
      return m_storage.layout().is_indexed();
    }
    [[nodiscard]] const index_type& indices() const
    {
      return m_storage.layout().indices();
    }
    /// Physical compressed storage, before any logical vertex selection.
    [[nodiscard]] const T* storage_data() const noexcept
    {
      return m_storage.storage_data();
    }

    [[nodiscard]] EntryProxy operator()(size_t i, size_t j)
    {
      bounds_check(i, j);
      if (is_indexed())
      {
        throw std::logic_error(
          "Cannot mutate an indexed DistanceMatrix directly; materialize its shared tensor backing first");
      }
      if (i == j)
        return EntryProxy(nullptr);
      return EntryProxy(&m_storage(i, j));
    }

    [[nodiscard]] T operator()(size_t i, size_t j) const
    {
      bounds_check(i, j);
      if (i == j)
        return T{};
      if (is_indexed())
      {
        if (m_storage.layout().source_index(i)
            == m_storage.layout().source_index(j))
          return T{};
      }
      return m_storage(i, j);
    }

    [[nodiscard]] bool operator==(const DistanceMatrix& rhs) const
    {
      if (size() != rhs.size())
        return false;
      for (size_t i = 1; i < size(); ++i)
      {
        for (size_t j = 0; j < i; ++j)
        {
          if ((*this)(i, j) != rhs(i, j))
            return false;
        }
      }
      return true;
    }

    [[nodiscard]] bool operator!=(const DistanceMatrix& rhs) const
    {
      return !(*this == rhs);
    }

    [[nodiscard]] const T* data() const
    {
      if (is_indexed())
      {
        throw std::logic_error(
          "Indexed DistanceMatrix storage is not logically contiguous; copy it before accessing data()");
      }
      return m_storage.storage_data();
    }

    void validate_index(const index_type& selection) const
    {
      if (selection.rank() != 1)
      {
        throw std::invalid_argument(
          "Distance-matrix selections must have 1 dimension, got "
          + std::to_string(selection.rank()));
      }
      for (size_t i = 0; i < selection.shape(0); ++i)
      {
        if (selection(i) >= size())
        {
          throw std::out_of_range(
            "Distance-matrix index " + std::to_string(selection(i))
            + " is out of bounds for a matrix of size " + std::to_string(size()));
        }
      }
    }

    [[nodiscard]] DistanceMatrix index_into(const index_type& selection) const
    {
      validate_index(selection);
      Tensor<uint64_t> resolved({selection.shape(0)});
      for (size_t i = 0; i < selection.shape(0); ++i)
      {
        const auto logical = static_cast<size_t>(selection(i));
        resolved(i) = m_storage.layout().source_index(logical);
      }
      return DistanceMatrix(*this, std::move(resolved));
    }

    [[nodiscard]] DistanceMatrix materialized_copy() const { return copy(); }

    /// Return a shallow view of the complete backing matrix, before any
    /// logical row/column selection is applied.
    [[nodiscard]] DistanceMatrix source_view() const
    {
      DistanceMatrix result;
      result.m_storage = m_storage.with_layout(
        DistanceMatrixLayout(m_storage.layout().source_size()));
      return result;
    }

    [[nodiscard]] static size_t storage_size(size_t n)
    {
      return n * (n - 1) / 2;
    }

  private:
    DistanceMatrix(const DistanceMatrix& source, index_type indices)
      : m_storage(source.m_storage.with_layout(
          DistanceMatrixLayout(source.m_storage.layout().source_size(),
            std::move(indices))))
    { }

    void bounds_check(size_t i, size_t j) const
    {
      if (i >= size() || j >= size())
      {
        std::ostringstream oss;
        oss << "DistanceMatrix index (" << i << ", " << j << ") out of range for matrix of size " << size();
        throw std::out_of_range(oss.str());
      }
    }

    FixedRankTensor<T, 2, DistanceMatrixLayout> m_storage;
  };

  template <typename T>
  struct is_compressed_matrix<DistanceMatrix<T>> : std::true_type {};

}

#endif // STABLEBEAR_DISTANCE_MATRIX_H
