// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// Created by bwehlin on 2/24/26.
//

#ifndef STABLEBEAR_POINT_CLOUD_H
#define STABLEBEAR_POINT_CLOUD_H

#include "tensor.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

namespace sb
{

  /// A point cloud of shape (n_points, dim).
  ///
  /// A PointCloud either owns its coordinates (like any Tensor<T>) or is an
  /// indexed view: it shares another cloud's coordinate buffer and selects rows
  /// through an attached index set. Access via n_points()/dim()/operator()(i, j) is
  /// transparent to which mode it is in, so consumers (e.g. Ripser) need no
  /// special case. This lets a tensor of subsamples store one shared source plus
  /// small index arrays instead of re-storing every (possibly high-dimensional)
  /// point.
  template <ArithmeticType T>
  class PointCloud : public Tensor<T>
  {
  public:
    using Tensor<T>::Tensor;
    using Tensor<T>::operator=;
    using Tensor<T>::operator();  // keep base subscripts visible alongside the (i, j) overload below

    PointCloud() = default;
    PointCloud(const Tensor<T>& coords) : Tensor<T>(coords) { }
    PointCloud(Tensor<T>&& coords) : Tensor<T>(std::move(coords)) { }

    /// Indexed view: shares @p source's coordinates and selects rows via @p indices.
    PointCloud(const Tensor<T>& source, Tensor<uint64_t> indices)
      : Tensor<T>(source), m_indices(std::move(indices)) { }

    /// Whether this is an indexed view (rather than owning its coordinates).
    bool is_indexed() const { return m_indices.rank() == 1; }

    /// Number of points: selected rows when indexed, otherwise stored rows.
    size_t n_points() const { return is_indexed() ? m_indices.shape(0) : this->shape(0); }

    /// Point dimension.
    size_t dim() const { return this->shape(1); }

    /// Coordinate @p j of point @p i, transparent to indexing.
    const T& operator()(size_t i, size_t j) const
    {
      size_t row = is_indexed() ? static_cast<size_t>(m_indices({i})) : i;
      return (*this)({row, j});
    }

    /// The attached indices (rank-1 when indexed, empty otherwise).
    const Tensor<uint64_t>& indices() const { return m_indices; }

    /// Materialize the selected points into a contiguous coordinate tensor.
    /// Returns the coordinates as-is when not indexed.
    Tensor<T> materialize() const
    {
      if (!is_indexed())
      {
        return *this;
      }

      const size_t n = n_points();
      const size_t d = dim();
      Tensor<T> out({n, d});
      for (size_t i = 0; i < n; ++i)
      {
        const size_t row = static_cast<size_t>(m_indices({i}));
        for (size_t j = 0; j < d; ++j)
        {
          out({i, j}) = (*this)({row, j});
        }
      }
      return out;
    }

  private:
    Tensor<uint64_t> m_indices;  // rank-1 when an indexed view, empty otherwise
  };

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

}

#endif //STABLEBEAR_POINT_CLOUD_H
