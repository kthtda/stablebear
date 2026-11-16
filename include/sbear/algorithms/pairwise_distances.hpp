#ifndef STABLEBEAR_PAIRWISE_DISTANCES_H
#define STABLEBEAR_PAIRWISE_DISTANCES_H

#include "../distance_matrix.hpp"
#include "../executor.hpp"
#include "../point_cloud.hpp"
#include "../tensor.hpp"
#include "../walk.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sb
{
  namespace detail
  {
    template <typename T>
    void require_lp_exponent(T p)
    {
      // Written so that NaN fails as well.
      if (!(p >= T{1}))
      {
        throw std::invalid_argument("p must be >= 1.");
      }
    }
  }

  /// The distance of the Lp norm between points i and j of a cloud:
  /// (sum_k |x_ik - x_jk|^p)^(1/p) for p >= 1, and the largest
  /// |x_ik - x_jk| for p = infinity.
  template <typename T>
  T lp_distance(const PointCloud<T>& cloud, size_t i, size_t j, T p)
  {
    const size_t dim = cloud.dim();
    T total = 0;
    if (std::isinf(p))
    {
      for (size_t k = 0; k < dim; ++k)
      {
        total = std::max(total, std::abs(cloud(i, k) - cloud(j, k)));
      }
      return total;
    }
    if (p == T{1})
    {
      for (size_t k = 0; k < dim; ++k)
      {
        total += std::abs(cloud(i, k) - cloud(j, k));
      }
      return total;
    }
    if (p == T{2})
    {
      for (size_t k = 0; k < dim; ++k)
      {
        const T difference = cloud(i, k) - cloud(j, k);
        total += difference * difference;
      }
      return std::sqrt(total);
    }
    for (size_t k = 0; k < dim; ++k)
    {
      total += std::pow(std::abs(cloud(i, k) - cloud(j, k)), p);
    }
    return std::pow(total, T{1} / p);
  }

  /// The Lp distances between all pairs of points of a cloud, for p >= 1
  /// or infinity.
  template <typename T>
  DistanceMatrix<T> pairwise_lp_distances(const PointCloud<T>& cloud, T p)
  {
    detail::require_lp_exponent(p);
    const size_t n = cloud.n_points();
    DistanceMatrix<T> distances(n);
    for (size_t i = 1; i < n; ++i)
    {
      for (size_t j = 0; j < i; ++j)
      {
        distances(i, j) = lp_distance(cloud, i, j, p);
      }
    }
    return distances;
  }

  /// pairwise_lp_distances of every cloud of a tensor, in parallel over
  /// the clouds.
  template <typename T, typename ExecutorT>
  Tensor<DistanceMatrix<T>> pairwise_lp_distances(const Tensor<PointCloud<T>>& clouds, T p, ExecutorT& executor)
  {
    detail::require_lp_exponent(p);
    Tensor<DistanceMatrix<T>> distances(clouds.shape());
    parallel_walk(clouds, [&](const std::vector<size_t>& idx) {
      distances(idx) = pairwise_lp_distances(clouds(idx), p);
    }, executor);
    return distances;
  }
}

#endif // STABLEBEAR_PAIRWISE_DISTANCES_H
