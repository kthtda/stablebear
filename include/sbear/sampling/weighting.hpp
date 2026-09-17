#ifndef STABLEBEAR_SAMPLING_WEIGHTING_H
#define STABLEBEAR_SAMPLING_WEIGHTING_H

#include "../point_cloud.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <span>

namespace sb::sampling
{

  // ===========================================================================
  // Sampling weights: how each reference point earns its draw probability.
  //
  // A filter maps a (query point, reference point) pair to a scalar; a
  // distribution maps that value to a non-negative weight. Both are plain
  // functors, so the built-ins compile into a fully fused per-query draw path
  // (see subsample.hpp, which computes one query's weight row at a time).
  //
  // A distribution may additionally provide `T log_weight(T v) const`
  // (= log(operator()(v)), see Gaussian): a query's row is then evaluated in
  // log space and finished with a max-shift exponentiation (log_row_to_weights),
  // so weights that mathematically never vanish cannot underflow to an all-zero
  // row. Either way the draw only ever sees plain weights.
  // ===========================================================================

  /// Euclidean distance between point @p i of cloud @p x and point @p j of
  /// cloud @p y. Named generically (x/y rather than query/reference) so it
  /// reads naturally anywhere a distance between two clouds is wanted. The
  /// samplers validate the shared dimension once up front, so this hot-path
  /// operator stays branch-free.
  template <typename T>
  struct EuclideanDistance
  {
    [[nodiscard]] T operator()(const PointCloud<T> &x, size_t i, const PointCloud<T> &y, size_t j) const noexcept
    {
      T acc = T(0);
      for (size_t k = 0; k < x.dim(); ++k)
      {
        T d = x(i, k) - y(j, k);
        acc += d * d;
      }
      return std::sqrt(acc);
    }
  };

  /// Weight 1 when the filter value lies in [low, high], 0 outside. With the
  /// distance filter that is a disk (low = 0), an annulus, or the whole cloud
  /// (the default, high = +inf). Member names match the Python spec.
  template <typename T>
  struct Uniform
  {
    T low = T(0);
    T high = std::numeric_limits<T>::infinity();

    [[nodiscard]] T operator()(T v) const noexcept
    {
      return (v >= low && v <= high) ? T(1) : T(0);
    }
  };

  /// Unnormalized Gaussian of the filter value. operator() is the ordinary
  /// exp(-((v - mean) / sigma)^2 / 2). The extra log_weight member is the
  /// log-space channel: the sampler evaluates a query's row through log_weight
  /// and finishes it with log_row_to_weights, so a row of points many sigma
  /// from the mean cannot underflow to all-zero (a raw exp does so past ~38
  /// sigma in float64, ~14 in float32). log_weight, not operator(), is what
  /// makes far-tail sampling well behaved.
  template <typename T>
  struct Gaussian
  {
    T mean = T(0);
    T sigma = T(1);

    [[nodiscard]] T operator()(T v) const noexcept
    {
      return std::exp(log_weight(v));
    }

    [[nodiscard]] T log_weight(T v) const noexcept
    {
      T d = (v - mean) / sigma;
      return T(-0.5) * d * d;
    }
  };

  namespace detail
  {

    /// Detects the optional log-space channel: a `log_weight(v)` member
    /// returning log(operator()(v)). Plain functors are evaluated directly.
    template <typename DistF, typename T>
    concept has_log_weight = requires(const DistF dist, T v) {
      { dist.log_weight(v) } -> std::convertible_to<T>;
    };

    /// In-place max-shift exponentiation of a log-weight row:
    /// w_i = exp(e_i - max_j e_j). The shift cancels on normalization, and
    /// the best point's weight is exactly 1, so the row can never underflow
    /// to all-zero.
    template <typename T>
    void log_row_to_weights(std::span<T> row)
    {
      T maxLog = -std::numeric_limits<T>::infinity();
      for (T e : row)
      {
        maxLog = std::max(maxLog, e); // drops NaNs; exp keeps them NaN below
      }
      if (!(maxLog > -std::numeric_limits<T>::infinity()))
      {
        maxLog = T(0); // all -inf (empty region): avoid (-inf) - (-inf) = NaN
      }
      for (T &e : row)
      {
        e = std::exp(e - maxLog);
      }
    }

    /// One weight-row entry: the log-weight when the distribution offers the
    /// channel (the row is then finished by log_row_to_weights), else the
    /// weight itself.
    template <typename T, typename DistF>
    T weight_entry(const DistF &distribution, T value)
    {
      if constexpr (has_log_weight<DistF, T>)
      {
        return distribution.log_weight(value);
      }
      else
      {
        return distribution(value);
      }
    }

  } // namespace detail

} // namespace sb::sampling

#endif
