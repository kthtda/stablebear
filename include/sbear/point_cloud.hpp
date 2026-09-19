//
// Created by bwehlin on 2/24/26.
//

#ifndef STABLEBEAR_POINT_CLOUD_H
#define STABLEBEAR_POINT_CLOUD_H

#include "tensor.hpp"
#include "nested_tensor.hpp"

#include <map>
#include <stdexcept>
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
    using index_type = Tensor<uint64_t>;

    PointCloud() = default;
    explicit PointCloud(const std::vector<size_t>& shape) : m_coords(shape) { }
    PointCloud(const Tensor<T>& coords) : m_coords(coords) { }
    PointCloud(Tensor<T>&& coords) : m_coords(std::move(coords)) { }

    // Compatibility with the former PointCloud = Tensor<T> alias. For an
    // indexed cloud, shape() describes the selected logical coordinates.
    [[nodiscard]] size_t rank() const noexcept { return m_coords.rank(); }

    [[nodiscard]] std::vector<size_t> shape() const
    {
      if (is_indexed())
      {
        return {n_points(), dim()};
      }
      return m_coords.shape();
    }

    [[nodiscard]] size_t shape(size_t axis) const
    {
      if (is_indexed() && axis == 0)
      {
        return n_points();
      }
      return m_coords.shape(axis);
    }

    [[nodiscard]] size_t size() const
    {
      return is_indexed() ? n_points() * dim() : m_coords.size();
    }

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
    [[nodiscard]] size_t n_points() const
    {
      if (m_coords.rank() == 0)
      {
        return 0;
      }
      return is_indexed() ? m_indices.shape(0) : m_coords.shape(0);
    }

    /// Point dimension. A default-constructed empty cloud has dimension zero.
    [[nodiscard]] size_t dim() const
    {
      return m_coords.rank() < 2 ? 0 : m_coords.shape(1);
    }

    /// The attached indices (rank-1 when indexed, empty otherwise).
    [[nodiscard]] const Tensor<uint64_t>& indices() const { return m_indices; }

    /// The underlying coordinate storage: the shared source when indexed. Use
    /// the cloud-level members for the selected points.
    [[nodiscard]] const Tensor<T>& coords() const { return m_coords; }

    /// Coordinate @p j of point @p i, transparent to indexing. Mutable access
    /// is only available for dense clouds; an indexed tensor must transition
    /// its complete shared backing before exposing mutable coordinates.
    const T& operator()(size_t i, size_t j) const
    {
      const size_t row = is_indexed() ? static_cast<size_t>(m_indices(i)) : i;
      return m_coords({row, j});
    }

    T& operator()(size_t i, size_t j)
    {
      return mutable_coords()({i, j});
    }

    const T& operator()(const std::vector<size_t>& index) const
    {
      if (!is_indexed())
      {
        return m_coords(index);
      }
      if (index.size() != 2)
      {
        throw std::invalid_argument(
          "Point-cloud coordinate index must have 2 dimensions");
      }
      return (*this)(index[0], index[1]);
    }

    T& operator()(const std::vector<size_t>& index)
    {
      return mutable_coords()(index);
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

    /// Apply a row selection without copying coordinates. The returned value
    /// is a lightweight logical element produced on demand by an indexed
    /// tensor; indexed state is not stored in the tensor's source elements.
    [[nodiscard]] PointCloud index_into(const index_type& selection) const
    {
      if (m_coords.rank() != 2)
      {
        throw std::invalid_argument("Point-cloud coordinates must have 2 dimensions");
      }

      const Tensor<uint64_t>& requested = selection;
      if (requested.rank() != 1)
      {
        throw std::invalid_argument("Point-cloud selections must have 1 dimension");
      }

      for (size_t i = 0; i < requested.shape(0); ++i)
      {
        if (requested(i) >= n_points())
        {
          throw std::out_of_range("Point-cloud index out of bounds");
        }
      }

      if (!is_indexed())
      {
        return PointCloud(m_coords, requested);
      }

      Tensor<uint64_t> resolved({requested.shape(0)});
      for (size_t i = 0; i < requested.shape(0); ++i)
      {
        const uint64_t logicalIndex = requested(i);
        resolved(i) = m_indices(static_cast<size_t>(logicalIndex));
      }
      return PointCloud(m_coords, std::move(resolved));
    }

    /// Return an owning value containing exactly the logical coordinates.
    [[nodiscard]] PointCloud materialized_copy() const
    {
      return is_indexed() ? PointCloud(materialize()) : PointCloud(m_coords.copy());
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

    [[nodiscard]] Tensor<T>& mutable_coords()
    {
      if (is_indexed())
      {
        throw std::logic_error(
          "Cannot mutate an indexed PointCloud directly; materialize its shared tensor backing first");
      }
      return m_coords;
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
   *
   * Sharing carries over: each distinct coordinate buffer is cast once
   * (deduplicated by storage identity, as in the io layer) and indexed views are
   * rebuilt on top of the cast source with their own index arrays. Casting cell
   * by cell through materialize() would instead expand every view into a full
   * private copy of its source -- exactly the storage blow-up that indexed
   * subsample tensors exist to avoid.
   */
  template <typename T, typename U>
  requires std::is_constructible_v<T, U>
  [[nodiscard]] Tensor<PointCloud<T>> pcloud_cast(const Tensor<PointCloud<U>>& src)
  {
    Tensor<PointCloud<T>> result(src.shape());

    // Cast each distinct source buffer once...
    std::map<std::shared_ptr<const void>, Tensor<T>, std::owner_less<std::shared_ptr<const void>>> castSources;
    walk(src, [&](const std::vector<size_t>& idx) {
      const Tensor<U>& coords = src(idx).coords();
      if (!castSources.contains(coords.storage_owner()))
      {
        castSources.emplace(coords.storage_owner(), tensor_cast<T>(coords));
      }
    });

    // ...then rebuild every cell on its shared cast source.
    walk(src, [&](const std::vector<size_t>& idx) {
      const PointCloud<U>& cloud = src(idx);
      const Tensor<T>& source = castSources.at(cloud.coords().storage_owner());
      if (cloud.is_indexed())
      {
        result(idx) = PointCloud<T>(source, cloud.indices().copy());
      }
      else
      {
        result(idx) = PointCloud<T>(source);
      }
    });
    return result;
  }

} // namespace sb

#endif // STABLEBEAR_POINT_CLOUD_H
