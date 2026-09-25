//
// Created by bwehlin on 2/24/26.
//

#ifndef STABLEBEAR_POINT_CLOUD_H
#define STABLEBEAR_POINT_CLOUD_H

#include "fixed_rank_tensor.hpp"
#include "nested_tensor.hpp"
#include "tensor.hpp"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace sb
{
  class PointCloudLayout
  {
  public:
    using shape_type = std::array<size_t, 2>;
    using index_type = Tensor<uint64_t>;

    PointCloudLayout() = default;
    explicit PointCloudLayout(shape_type sourceShape)
      : m_sourceShape(std::move(sourceShape))
    { }
    PointCloudLayout(shape_type sourceShape, index_type indices)
      : m_sourceShape(std::move(sourceShape)), m_indices(std::move(indices))
    { }

    [[nodiscard]] shape_type shape() const noexcept
    {
      return {is_indexed() ? m_indices.shape(0) : m_sourceShape[0],
              m_sourceShape[1]};
    }

    [[nodiscard]] size_t storage_size() const noexcept
    {
      return m_sourceShape[0] * m_sourceShape[1];
    }

    [[nodiscard]] size_t offset(const shape_type& indices) const
    {
      const auto logicalShape = shape();
      if (indices[0] >= logicalShape[0] || indices[1] >= logicalShape[1])
        throw std::out_of_range("Point-cloud coordinate index out of range");
      const size_t row = is_indexed()
        ? static_cast<size_t>(m_indices(indices[0]))
        : indices[0];
      return row * m_sourceShape[1] + indices[1];
    }

    [[nodiscard]] bool is_indexed() const noexcept
    {
      return m_indices.rank() == 1;
    }
    [[nodiscard]] const shape_type& source_shape() const noexcept
    {
      return m_sourceShape;
    }
    [[nodiscard]] const index_type& indices() const noexcept { return m_indices; }

  private:
    shape_type m_sourceShape{};
    index_type m_indices;
  };

  /// A point cloud of shape (n_points, dim).
  ///
  /// A PointCloud either owns its coordinates or is an indexed view: it shares
  /// another cloud's coordinate buffer and selects points through an attached
  /// index set. Access via n_points()/dim()/operator()(i, j) is transparent to
  /// which mode it is in, so consumers need no special case. This
  /// lets a tensor of subsamples store one shared source plus small index arrays
  /// instead of re-storing every (possibly high-dimensional) point.
  ///
  /// Deliberately not a Tensor<T>: FixedRankTensor owns the shared coordinate
  /// storage while PointCloudLayout carries the logical point selection. This
  /// keeps the storage/view model generic without teaching FixedRankTensor
  /// about point clouds.
  template <ArithmeticType T>
  class PointCloud
  {
  public:
    using value_type = T;
    using index_type = Tensor<uint64_t>;

    using coordinate_storage_type = FixedRankTensor<T, 2, PointCloudLayout>;

    PointCloud() = default;
    explicit PointCloud(const std::vector<size_t>& shape)
      : m_coords(PointCloudLayout(coordinate_shape(shape)))
    { }
    PointCloud(const Tensor<T>& coords) : m_coords(copy_coordinates(coords)) { }

    // Compatibility with the former PointCloud = Tensor<T> alias. For an
    // indexed cloud, shape() describes the selected logical coordinates.
    [[nodiscard]] static constexpr size_t rank() noexcept { return 2; }

    [[nodiscard]] std::vector<size_t> shape() const
    {
      return {n_points(), dim()};
    }

    [[nodiscard]] size_t shape(size_t axis) const
    {
      if (axis == 0)
        return n_points();
      return m_coords.shape(axis);
    }

    [[nodiscard]] size_t size() const
    {
      return n_points() * dim();
    }

    /// Indexed view: shares @p source's coordinates and selects points via @p indices.
    PointCloud(const Tensor<T>& source, Tensor<uint64_t> indices)
      : m_coords(copy_coordinates(source))
    {
      validate_index(indices);
      m_coords = m_coords.with_layout(
        PointCloudLayout(m_coords.layout().source_shape(), std::move(indices)));
    }

    /// Indexed view over another cloud's coordinates. @p indices refer to points
    /// in @p source's coordinate storage (not to the points @p source selects).
    PointCloud(const PointCloud& source, Tensor<uint64_t> indices)
      : m_coords(source.m_coords.with_layout(
          PointCloudLayout(source.m_coords.layout().source_shape(),
            std::move(indices))))
    { }

    /// Whether this is an indexed view (rather than owning its coordinates).
    [[nodiscard]] bool is_indexed() const { return m_coords.layout().is_indexed(); }

    /// Number of points: selected points when indexed, otherwise stored points.
    [[nodiscard]] size_t n_points() const
    {
      return m_coords.shape(0);
    }

    /// Point dimension. A default-constructed empty cloud has dimension zero.
    [[nodiscard]] size_t dim() const
    {
      return m_coords.shape(1);
    }

    /// The attached indices (rank-1 when indexed, empty otherwise).
    [[nodiscard]] const Tensor<uint64_t>& indices() const
    {
      return m_coords.layout().indices();
    }

    /// Fixed-rank coordinate storage, including the logical selection layout.
    [[nodiscard]] const coordinate_storage_type& coords() const { return m_coords; }

    /// Physical coordinate storage, before any logical point selection.
    [[nodiscard]] const T* storage_data() const noexcept
    {
      return m_coords.storage_data();
    }

    /// Coordinate @p j of point @p i, transparent to indexing. Mutable access
    /// is only available for dense clouds; an indexed tensor must transition
    /// its complete shared backing before exposing mutable coordinates.
    const T& operator()(size_t i, size_t j) const
    {
      return m_coords(i, j);
    }

    T& operator()(size_t i, size_t j)
    {
      return mutable_coords()(i, j);
    }

    const T& operator()(const std::vector<size_t>& index) const
    {
      if (index.size() != 2)
      {
        throw std::invalid_argument(
          "Point-cloud coordinate index must have 2 dimensions");
      }
      return (*this)(index[0], index[1]);
    }

    T& operator()(const std::vector<size_t>& index)
    {
      if (index.size() != 2)
        throw std::invalid_argument(
          "Point-cloud coordinate index must have 2 dimensions");
      return mutable_coords()(index[0], index[1]);
    }

    /// View-transparent equality: two clouds are equal when they present the
    /// same points, regardless of whether either is an indexed view.
    [[nodiscard]] bool operator==(const PointCloud& rhs) const
    {
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

    /// Return a self-contained copy of the logical coordinates. Indexed point
    /// clouds are transient read-only views produced by indexed tensors; a
    /// copy crosses into ordinary value storage and is therefore materialized.
    [[nodiscard]] PointCloud copy() const
    {
      PointCloud result(std::vector<size_t>{n_points(), dim()});
      for (size_t i = 0; i < n_points(); ++i)
      {
        for (size_t j = 0; j < dim(); ++j)
          result(i, j) = (*this)(i, j);
      }
      return result;
    }

    /// Validate a point selection without constructing the selected cloud.
    void validate_index(const index_type& selection) const
    {
      const Tensor<uint64_t>& requested = selection;
      if (requested.rank() != 1)
      {
        throw std::invalid_argument(
          "Point-cloud selections must have 1 dimension, got "
          + std::to_string(requested.rank()));
      }

      for (size_t i = 0; i < requested.shape(0); ++i)
      {
        if (requested(i) >= n_points())
        {
          throw std::out_of_range(
            "Point-cloud index " + std::to_string(requested(i))
            + " is out of bounds for a cloud with "
            + std::to_string(n_points()) + " points");
        }
      }
    }

    /// Validate and apply a point selection. Indexed tensors also validate at
    /// construction for immediate errors, but retain this check because their
    /// shared source elements may subsequently be replaced.
    [[nodiscard]] PointCloud index_into(const index_type& selection) const
    {
      validate_index(selection);
      const Tensor<uint64_t>& requested = selection;

      if (!is_indexed())
      {
        return PointCloud(*this, requested);
      }

      Tensor<uint64_t> resolved({requested.shape(0)});
      for (size_t i = 0; i < requested.shape(0); ++i)
      {
        const uint64_t logicalIndex = requested(i);
        resolved(i) = indices()(static_cast<size_t>(logicalIndex));
      }
      return PointCloud(*this, std::move(resolved));
    }

    /// Return an owning value containing exactly the logical coordinates.
    [[nodiscard]] PointCloud materialized_copy() const
    {
      return copy();
    }

    /// Return a shallow view of the complete backing cloud before any logical
    /// point selection is applied.
    [[nodiscard]] PointCloud source_view() const
    {
      PointCloud result;
      result.m_coords = m_coords.with_layout(
        PointCloudLayout(m_coords.layout().source_shape()));
      return result;
    }

    /// Return an independent coordinate tensor containing the logical points.
    /// This is an interoperability operation; point-cloud materialization is
    /// represented by copy(), which preserves the PointCloud type.
    [[nodiscard]] Tensor<T> coordinates_copy() const
    {
      const size_t n = n_points();
      const size_t d = dim();
      Tensor<T> out({n, d});
      for (size_t i = 0; i < n; ++i)
      {
        for (size_t j = 0; j < d; ++j)
          out({i, j}) = (*this)(i, j);
      }
      return out;
    }

    /// Return an independent tensor containing the complete backing storage,
    /// before any logical point selection is applied.
    [[nodiscard]] Tensor<T> source_coordinates_copy() const
    {
      const auto& sourceShape = m_coords.layout().source_shape();
      const auto source = m_coords.with_layout(PointCloudLayout(sourceShape));
      Tensor<T> out({sourceShape[0], sourceShape[1]});
      for (size_t i = 0; i < sourceShape[0]; ++i)
      {
        for (size_t j = 0; j < sourceShape[1]; ++j)
          out({i, j}) = source(i, j);
      }
      return out;
    }

    [[nodiscard]] coordinate_storage_type& mutable_coords()
    {
      if (is_indexed())
      {
        throw std::logic_error(
          "Cannot mutate an indexed PointCloud directly; materialize its shared tensor backing first");
      }
      return m_coords;
    }

  private:
    [[nodiscard]] static typename coordinate_storage_type::shape_type
    coordinate_shape(const std::vector<size_t>& shape)
    {
      if (shape.size() != 2)
        throw std::invalid_argument(
          "Point-cloud coordinates must have 2 dimensions, got "
          + std::to_string(shape.size()));
      return {shape[0], shape[1]};
    }

    [[nodiscard]] static coordinate_storage_type
    copy_coordinates(const Tensor<T>& coords)
    {
      if (coords.rank() != 2)
        throw std::invalid_argument(
          "Point-cloud coordinates must have 2 dimensions, got "
          + std::to_string(coords.rank()));
      coordinate_storage_type result(
        PointCloudLayout({coords.shape(0), coords.shape(1)}));
      for (size_t i = 0; i < coords.shape(0); ++i)
      {
        for (size_t j = 0; j < coords.shape(1); ++j)
          result(i, j) = coords({i, j});
      }
      return result;
    }

    coordinate_storage_type m_coords;
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
   * Dense coordinate sharing carries over: each distinct coordinate buffer is
   * cast once. A transient indexed view is materialized because an ordinary
   * tensor must contain self-contained point-cloud values.
   */
  template <typename T, typename U>
  requires std::is_constructible_v<T, U>
  [[nodiscard]] Tensor<PointCloud<T>> pcloud_cast(const Tensor<PointCloud<U>>& src)
  {
    Tensor<PointCloud<T>> result(src.shape());

    // Cast each distinct source buffer once...
    std::map<const void*, PointCloud<T>> castSources;
    walk(src, [&](const std::vector<size_t>& idx) {
      const PointCloud<U>& cloud = src(idx);
      if (cloud.is_indexed())
        return;
      const auto& coords = cloud.coords();
      const void* source = static_cast<const void*>(coords.storage_data());
      if (!castSources.contains(source))
      {
        PointCloud<T> casted(
          std::vector<size_t>{coords.shape(0), coords.shape(1)});
        for (size_t i = 0; i < coords.shape(0); ++i)
        {
          for (size_t j = 0; j < coords.shape(1); ++j)
            casted(i, j) = static_cast<T>(coords(i, j));
        }
        castSources.emplace(source, std::move(casted));
      }
    });

    // ...then rebuild every cell on its shared cast source.
    walk(src, [&](const std::vector<size_t>& idx) {
      const PointCloud<U>& cloud = src(idx);
      if (cloud.is_indexed())
      {
        result(idx) = PointCloud<T>(tensor_cast<T>(cloud.coordinates_copy()));
      }
      else
      {
        result(idx) = castSources.at(
          static_cast<const void*>(cloud.coords().storage_data()));
      }
    });
    return result;
  }

} // namespace sb

#endif // STABLEBEAR_POINT_CLOUD_H
