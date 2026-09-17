//
// Created by bwehlin on 2/24/26.
//

#ifndef STABLEBEAR_POINT_CLOUD_H
#define STABLEBEAR_POINT_CLOUD_H

#include "tensor.hpp"

#include <type_traits>
#include <vector>

namespace sb
{

  /// A point cloud of shape (n_points, dim).
  ///
  /// A PointCloud either owns its coordinates or is an indexed view: it shares
  /// another cloud's coordinate buffer and selects rows through an attached
  /// index set. Access via n_points()/dim()/operator()(i, j) is transparent to
  /// which mode it is in, so consumers need no special case. This
  /// lets a tensor of subsamples store one shared source plus small index arrays
  /// instead of re-storing every (possibly high-dimensional) point.
  ///
  /// Deliberately not a Tensor<T>: the raw coordinate storage and the selected
  /// points disagree for indexed views, so tensor-level access has no single
  /// meaning here. Use the cloud-level members for the selected points, or
  /// coords() to reach the underlying storage explicitly.
  template <ArithmeticType T>
  class PointCloud
  {
  public:
    using value_type = T;

    PointCloud() = default;
    PointCloud(const Tensor<T>& coords) : m_coords(coords) { }
    PointCloud(Tensor<T>&& coords) : m_coords(std::move(coords)) { }

    /// Indexed view: shares @p source's coordinates and selects rows via @p indices.
    PointCloud(const Tensor<T>& source, Tensor<uint64_t> indices)
      : m_coords(source), m_indices(std::move(indices)) { }

    /// Indexed view over another cloud's coordinates. @p indices refer to rows
    /// of @p source's coordinate storage (not to the rows @p source selects).
    PointCloud(const PointCloud& source, Tensor<uint64_t> indices)
      : m_coords(source.m_coords), m_indices(std::move(indices)) { }

    /// Whether this is an indexed view (rather than owning its coordinates).
    [[nodiscard]] bool is_indexed() const { return m_indices.rank() == 1; }

    /// Number of points: selected rows when indexed, otherwise stored rows.
    [[nodiscard]] size_t n_points() const { return is_indexed() ? m_indices.shape(0) : m_coords.shape(0); }

    /// Point dimension.
    [[nodiscard]] size_t dim() const { return m_coords.shape(1); }

    /// The attached indices (rank-1 when indexed, empty otherwise).
    [[nodiscard]] const Tensor<uint64_t>& indices() const { return m_indices; }

    /// The underlying coordinate storage: the shared source when indexed. Use
    /// the cloud-level members for the selected points.
    [[nodiscard]] const Tensor<T>& coords() const { return m_coords; }

    /// Coordinate @p j of point @p i, transparent to indexing. (Read-only;
    /// writes go through materialize().)
    const T& operator()(size_t i, size_t j) const
    {
      const size_t row = is_indexed() ? static_cast<size_t>(m_indices(i)) : i;
      return m_coords({row, j});
    }

    /// View-transparent equality: two clouds are equal when they present the
    /// same points, regardless of whether either is an indexed view.
    [[nodiscard]] bool operator==(const PointCloud& rhs) const
    {
      if (m_coords.rank() != 2 || rhs.m_coords.rank() != 2)
      {
        // Degenerate (e.g. default-constructed) cells: compare storage directly.
        return m_coords == rhs.m_coords;
      }
      if (n_points() != rhs.n_points() || dim() != rhs.dim())
      {
        return false;
      }
      const size_t n = n_points();
      const size_t d = dim();
      for (size_t i = 0; i < n; ++i)
      {
        for (size_t j = 0; j < d; ++j)
        {
          if ((*this)(i, j) != rhs(i, j))
          {
            return false;
          }
        }
      }
      return true;
    }

    /// Deep copy. Tensor cells route stores through detail::store_copy, which
    /// prefers copy(). An owning cloud copies its coordinates. An indexed view
    /// copies its index array (so cells don't alias) and, by default
    /// (@p keepSource), keeps sharing the source coordinates — immutable by
    /// convention, the point of indexed views; with @p keepSource false it also
    /// deep-copies the source, yielding a view that aliases nothing.
    /// @p keepSource is moot for an owning cloud, which never shares.
    [[nodiscard]] PointCloud copy(bool keepSource = true) const
    {
      if (is_indexed())
      {
        if (keepSource)
        {
          return PointCloud(m_coords, m_indices.copy());
        }
        return PointCloud(m_coords.copy(), m_indices.copy());
      }
      return PointCloud(m_coords.copy());
    }

    /// Materialize the selected points into a contiguous coordinate tensor.
    /// Returns the coordinates as-is when not indexed.
    [[nodiscard]] Tensor<T> materialize() const
    {
      if (!is_indexed())
      {
        return m_coords;
      }

      const size_t n = n_points();
      const size_t d = dim();
      Tensor<T> out({n, d});
      for (size_t i = 0; i < n; ++i)
      {
        const auto row = static_cast<size_t>(m_indices(i));
        for (size_t j = 0; j < d; ++j)
        {
          out({i, j}) = m_coords({row, j});
        }
      }
      return out;
    }

  private:
    Tensor<T> m_coords;         // (n_source_points, dim), possibly shared
    Tensor<uint64_t> m_indices; // rank-1 when an indexed view, empty otherwise
  };

  /// Identifies PointCloud<T> instantiations (exposing scalar_type = T), for
  /// the io and Python binding layers.
  template <typename T>
  struct is_point_cloud : std::false_type {};

  template <typename T>
  struct is_point_cloud<PointCloud<T>> : std::true_type { using scalar_type = T; };

  template <typename T>
  inline constexpr bool is_point_cloud_v = is_point_cloud<T>::value;

  /**
   * Cast a tensor of point clouds (Tensor<PointCloud<U>>) to a different precision
   * (Tensor<PointCloud<T>>), converting each point cloud's coordinates.
   */
  template <typename T, typename U>
  requires std::is_constructible_v<T, U>
  [[nodiscard]] Tensor<PointCloud<T>> pcloud_cast(const Tensor<PointCloud<U>>& src)
  {
    Tensor<PointCloud<T>> result(src.shape());
    walk(src, [&](const std::vector<size_t>& idx) {
      result(idx) = PointCloud<T>(tensor_cast<T>(src(idx).materialize()));
    });
    return result;
  }

} // namespace sb

#endif // STABLEBEAR_POINT_CLOUD_H
