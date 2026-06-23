#ifndef SB_POINT_PROCESS_POISSON_H
#define SB_POINT_PROCESS_POISSON_H

#include "../tensor.hpp"
#include "../walk.hpp"

#include <random>
#include <stdexcept>
#include <vector>

namespace sb::pp
{

  /// Generate a tensor of point clouds from a homogeneous spatial Poisson process.
  ///
  /// Each element of @p out is filled with a point cloud of shape (N, dim) where
  /// N ~ Poisson(rate * volume) and points are drawn uniformly in [lo, hi].
  template <typename T, typename EngineT>
  void sample_poisson(
      Tensor<PointCloud<T>>& out,
      size_t dim,
      T rate,
      const std::vector<T>& lo,
      const std::vector<T>& hi,
      RandomGenerator<EngineT>& gen,
      Executor& exec)
  {
    if (lo.size() != dim || hi.size() != dim)
    {
      throw std::invalid_argument("lo and hi must have length equal to dim");
    }

    for (size_t i = 0; i < dim; ++i)
    {
      if (lo[i] > hi[i])
      {
        throw std::invalid_argument("lo must be <= hi in every dimension");
      }
    }

    T volume = static_cast<T>(1);
    for (size_t i = 0; i < dim; ++i)
    {
      volume *= (hi[i] - lo[i]);
    }

    T lambda = rate * volume;

    sb::parallel_walk(out, gen, [dim, lambda, &lo, &hi, &out](const std::vector<size_t>& idx, auto& engine) {

      std::poisson_distribution<size_t> countDist(static_cast<double>(lambda));
      auto nPoints = countDist(engine);

      PointCloud<T> pc({nPoints, dim});

      for (size_t i = 0; i < nPoints; ++i)
      {
        for (size_t j = 0; j < dim; ++j)
        {
          std::uniform_real_distribution<T> coordDist(lo[j], hi[j]);
          pc({i, j}) = coordDist(engine);
        }
      }

      out(idx) = std::move(pc);
    }, exec);
  }

}

#endif
