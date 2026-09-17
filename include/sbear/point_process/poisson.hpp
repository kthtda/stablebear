#ifndef SB_POINT_PROCESS_POISSON_H
#define SB_POINT_PROCESS_POISSON_H

#include "../tensor.hpp"
#include "../point_cloud.hpp"
#include "../walk.hpp"

#include <cmath>
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

    if (!std::isfinite(static_cast<double>(rate)) || rate < static_cast<T>(0))
    {
      throw std::invalid_argument("rate must be finite and non-negative");
    }

    for (size_t i = 0; i < dim; ++i)
    {
      if (!std::isfinite(static_cast<double>(lo[i])) ||
          !std::isfinite(static_cast<double>(hi[i])))
      {
        throw std::invalid_argument("lo and hi must be finite in every dimension");
      }
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

    // A finite, non-negative rate and a finite box keep lambda well-defined,
    // but a finite-yet-enormous box can still overflow volume to +inf. Guard
    // the mean before it reaches std::poisson_distribution, whose behavior is
    // undefined for a non-finite mean.
    if (!std::isfinite(static_cast<double>(lambda)))
    {
      throw std::invalid_argument(
          "rate * volume is not finite (the sampling region is too large)");
    }

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
