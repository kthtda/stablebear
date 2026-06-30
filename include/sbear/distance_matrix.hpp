#ifndef STABLEBEAR_DISTANCE_MATRIX_H
#define STABLEBEAR_DISTANCE_MATRIX_H

#include "config.hpp"
#include "concepts.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
  ///
  /// A matrix may also be an *indexed view*: it shares another matrix's buffer
  /// and selects a subset of points through an attached index set, so size() and
  /// operator()(i, j) report the principal submatrix over those indices. This lets
  /// a tensor of subsampled distance matrices store one shared source plus small
  /// index arrays instead of re-storing every sub-matrix. Access is transparent,
  /// so Ripser (which uses only size()/operator()) needs no special case. An
  /// indexed view is read-only and not serializable; materialize() it first.
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
      : m_data(std::make_shared<T[]>(storage_size(n)))
      , m_size(n)
    {
      if (init < T{})
        throw std::invalid_argument("Distance matrix entries must be nonnegative");
      std::fill(m_data.get(), m_data.get() + storage_size(n), init);
    }

    DistanceMatrix() : DistanceMatrix(0) { }

    /// Indexed view: shares @p source's compressed buffer and selects the
    /// principal submatrix over @p indices (rows/cols of @p source).
    DistanceMatrix(const DistanceMatrix& source, Tensor<uint64_t> indices)
      : m_data(source.m_data), m_size(source.m_size), m_indices(std::move(indices)) { }

    /// Whether this is an indexed view rather than owning its full buffer.
    [[nodiscard]] bool is_indexed() const { return m_indices.rank() == 1; }

    /// The selected source indices (rank-1 when indexed, empty otherwise).
    [[nodiscard]] const Tensor<uint64_t>& indices() const { return m_indices; }

    /// Return an independent value. An owning matrix is deep-copied; an indexed
    /// view keeps sharing the (immutable) source buffer but copies its index
    /// array, so the result never aliases another cell's indices. Tensor cells
    /// route stores through store_copy, which prefers copy().
    [[nodiscard]] DistanceMatrix copy() const
    {
      if (is_indexed())
        return DistanceMatrix(*this, m_indices.copy());
      DistanceMatrix result(m_size);
      std::copy(m_data.get(), m_data.get() + storage_size(m_size), result.m_data.get());
      return result;
    }

    /// Materialize the (selected) entries into an owning DistanceMatrix; returns
    /// the matrix as-is when not indexed.
    [[nodiscard]] DistanceMatrix materialize() const
    {
      if (!is_indexed())
        return *this;
      const size_t n = size();
      DistanceMatrix out(n);
      for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
          out(i, j) = (*this)(i, j);
      return out;
    }

    /// Number of points: selected indices when indexed, otherwise the full size.
    [[nodiscard]] size_t size() const { return is_indexed() ? m_indices.shape(0) : m_size; }

    /// Number of stored compressed entries (owning matrices only).
    [[nodiscard]] size_t storage_count() const { return storage_size(m_size); }

    [[nodiscard]] EntryProxy operator()(size_t i, size_t j)
    {
      if (is_indexed())
        throw std::logic_error("cannot write to an indexed distance-matrix view");
      bounds_check(i, j);
      if (i == j)
        return EntryProxy(nullptr);
      return EntryProxy(&m_data[compressed_index(i, j)]);
    }

    [[nodiscard]] T operator()(size_t i, size_t j) const
    {
      bounds_check(i, j);
      // Map view indices to source indices; equal source indices (the diagonal,
      // or a point drawn twice with replacement) are distance zero.
      const size_t a = source_index(i);
      const size_t b = source_index(j);
      if (a == b)
        return T{};
      return m_data[compressed_index(a, b)];
    }

    [[nodiscard]] bool operator==(const DistanceMatrix& rhs) const
    {
      if (size() != rhs.size())
        return false;
      if (!is_indexed() && !rhs.is_indexed())
        return std::equal(m_data.get(), m_data.get() + storage_size(m_size), rhs.m_data.get());
      const size_t n = size();
      for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
          if ((*this)(i, j) != rhs(i, j))
            return false;
      return true;
    }

    [[nodiscard]] bool operator!=(const DistanceMatrix& rhs) const { return !(*this == rhs); }

    /// Raw compressed buffer (owning matrices only; an indexed view shares the
    /// full source buffer, which is not its principal submatrix).
    [[nodiscard]] const T* data() const { return m_data.get(); }

    [[nodiscard]] static size_t storage_size(size_t n)
    {
      return n * (n - 1) / 2;
    }

  private:
    [[nodiscard]] T* mutable_data() { return m_data.get(); }

    template <typename MatT>
    friend MatT io::detail::read_compressed_matrix(std::istream&);

    [[nodiscard]] size_t source_index(size_t k) const
    {
      return is_indexed() ? static_cast<size_t>(m_indices({k})) : k;
    }

    void bounds_check(size_t i, size_t j) const
    {
      const size_t n = size();
      if (i >= n || j >= n)
      {
        std::ostringstream oss;
        oss << "DistanceMatrix index (" << i << ", " << j << ") out of range for matrix of size " << n;
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
    size_t m_size;                 // size of the (shared) source buffer
    Tensor<uint64_t> m_indices;    // rank-1 when an indexed view, empty otherwise
  };

  template <typename T>
  struct is_compressed_matrix<DistanceMatrix<T>> : std::true_type {};

}

#endif // STABLEBEAR_DISTANCE_MATRIX_H
