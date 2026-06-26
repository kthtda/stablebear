#ifndef STABLEBEAR_DISTANCE_MATRIX_H
#define STABLEBEAR_DISTANCE_MATRIX_H

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
        // Reject negatives AND NaN (NaN < 0 is false, so a plain `< 0` check
        // would let NaN through); +inf stays allowed.
        if (!(value >= T{}))
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
      : m_data(std::make_shared<T[]>(storage_size(n)))
      , m_size(n)
    {
      if (!(init >= T{}))
        throw std::invalid_argument("Distance matrix entries must be nonnegative");
      std::fill(m_data.get(), m_data.get() + storage_size(n), init);
    }

    DistanceMatrix() : DistanceMatrix(0) { }

    /// Return an independent deep copy. The implicit copy shares the
    /// std::shared_ptr buffer (view-like), which is relied on internally; this
    /// is used when a matrix must not alias its source (e.g. a tensor cell).
    [[nodiscard]] DistanceMatrix copy() const
    {
      DistanceMatrix result(m_size);
      std::copy(m_data.get(), m_data.get() + storage_size(m_size), result.m_data.get());
      return result;
    }

    [[nodiscard]] size_t size() const { return m_size; }
    [[nodiscard]] size_t storage_count() const { return storage_size(m_size); }

    [[nodiscard]] EntryProxy operator()(size_t i, size_t j)
    {
      bounds_check(i, j);
      if (i == j)
        return EntryProxy(nullptr);
      return EntryProxy(&m_data[compressed_index(i, j)]);
    }

    [[nodiscard]] T operator()(size_t i, size_t j) const
    {
      bounds_check(i, j);
      if (i == j)
        return T{};
      return m_data[compressed_index(i, j)];
    }

    [[nodiscard]] bool operator==(const DistanceMatrix& rhs) const
    {
      if (m_size != rhs.m_size)
        return false;
      return std::equal(m_data.get(), m_data.get() + storage_size(m_size), rhs.m_data.get());
    }

    [[nodiscard]] bool operator!=(const DistanceMatrix& rhs) const
    {
      if (m_size != rhs.m_size)
        return true;
      return std::mismatch(m_data.get(), m_data.get() + storage_size(m_size), rhs.m_data.get()).first
        != m_data.get() + storage_size(m_size);
    }

    [[nodiscard]] const T* data() const { return m_data.get(); }

    [[nodiscard]] static size_t storage_size(size_t n)
    {
      return n * (n - 1) / 2;
    }

  private:
    [[nodiscard]] T* mutable_data() { return m_data.get(); }

    template <typename MatT>
    friend MatT io::detail::read_compressed_matrix(std::istream&);

    void bounds_check(size_t i, size_t j) const
    {
      if (i >= m_size || j >= m_size)
      {
        std::ostringstream oss;
        oss << "DistanceMatrix index (" << i << ", " << j << ") out of range for matrix of size " << m_size;
        throw std::out_of_range(oss.str());
      }
    }

    [[nodiscard]] static size_t compressed_index(size_t i, size_t j)
    {
      auto row = std::max(i, j);
      auto col = std::min(i, j);
      return row * (row - 1) / 2 + col;
    }

    std::shared_ptr<T[]> m_data;
    size_t m_size;
  };

  template <typename T>
  struct is_compressed_matrix<DistanceMatrix<T>> : std::true_type {};

}

#endif // STABLEBEAR_DISTANCE_MATRIX_H
