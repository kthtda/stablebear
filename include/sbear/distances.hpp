#ifndef STABLEBEAR_DISTANCES_H
#define STABLEBEAR_DISTANCES_H

#include "concepts.hpp"
#include "point_cloud.hpp"
#include "tensor.hpp"

#include <cstddef>

namespace sb
{
  // Distance oracles: lightweight functors that answer d(i, j) on demand over
  // some underlying data, satisfying the DistanceOracle concept (see
  // concepts.hpp). Together with DistanceMatrix<T> they let algorithms be
  // templated over "anything that measures distance between indexed points" --
  // wrapping data such as a point cloud in an oracle is the intended way to
  // add new metrics without materializing matrices.

  /// Squared-Euclidean distance functor over a point cloud: answers d(i, j)^2
  /// on demand without materializing a distance matrix. Satisfies
  /// DistanceOracle just like DistanceMatrix<T>, so algorithms such as MST
  /// construction can be templated over either.
  /// Precondition: the cloud must be rank 2 with shape (n, dim); callers
  /// validate that before constructing the oracle. Works
  /// in squared distances because comparisons are unchanged (x -> x^2 is
  /// monotone on [0, inf)); the caller applies sqrt to the few distances it
  /// keeps (e.g. the n-1 merge distances of an MST) instead of one sqrt per
  /// O(n^2) query.
  template <typename T>
  class SquaredEuclideanDistance
  {
  public:
    explicit SquaredEuclideanDistance(const PointCloud<T> &points)
        : m_points(points)
    {
    }

    [[nodiscard]] T operator()(size_t i, size_t j) const
    {
      T sumSq{0};
      for (size_t k = 0; k < m_points.dim(); ++k)
      {
        auto diff = m_points(i, k) - m_points(j, k);
        sumSq += diff * diff;
      }
      return sumSq;
    }

    [[nodiscard]] size_t size() const noexcept
    {
      return m_points.n_points();
    }

  private:
    PointCloud<T> m_points;
  };
} // namespace sb

#endif
