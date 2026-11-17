#ifndef SB_OPERATIONS_CUH
#define SB_OPERATIONS_CUH

#include <cmath>
#include <limits>

#ifdef BUILD_WITH_CUDA
  #include <cuda_runtime.h>
#else
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
#endif

namespace sb
{
  // An operation integrates a function of two PCFs, f and g, over the
  // rectangles on which both are constant. It provides
  //
  //   accumulator_type      state while integrating one pair
  //   init()                the state before the first rectangle
  //   add(acc, len, f, g)   adds a rectangle of width len with values f and g
  //   result(acc)           the value for the pair
  //
  // All members run on both host and device, so NaN and infinity are tested
  // with comparisons rather than std::isnan and std::isinf.

  template <typename Tt, typename Tv>
  struct OperationL1Dist
  {
    using accumulator_type = Tv;

    __host__ __device__ accumulator_type init() const
    {
      return Tv(0);
    }

    __host__ __device__ void add(accumulator_type& acc, Tt len, Tv f, Tv g) const
    {
      acc += len * std::abs(f - g);
    }

    __host__ __device__ Tv result(const accumulator_type& acc) const
    {
      return acc;
    }
  };

  /// Accumulates the integral of |f - g|^p as scale^p * sum, where scale is
  /// the largest |f - g| so far. Each term is then (|f - g| / scale)^p <= 1,
  /// so it cannot overflow as |f - g|^p does for large values or large p.
  /// The sum is rescaled whenever a larger |f - g| appears, as in the BLAS
  /// nrm2 routine.
  template <typename Tv>
  struct LpAccumulator
  {
    Tv scale = Tv(0);
    Tv sum = Tv(0);
  };

  template <typename Tt, typename Tv>
  struct OperationLpDist
  {
    using accumulator_type = LpAccumulator<Tv>;

    Tv p = 2;
    OperationLpDist() = default;
    explicit OperationLpDist(Tv pv)
      : p(pv)
    { }

    __host__ __device__ accumulator_type init() const
    {
      return accumulator_type{};
    }

    __host__ __device__ void add(accumulator_type& acc, Tt len, Tv f, Tv g) const
    {
      if (!(len > Tt(0)))
      {
        return;
      }
      Tv d = std::abs(f - g);
      if (d != d) // NaN
      {
        acc.scale = d;
        acc.sum = d;
        return;
      }
      if (d == Tv(0) || !(acc.scale <= max_value()))
      {
        // Nothing to add, or the result is already infinite or NaN
        return;
      }
      if (d > acc.scale)
      {
        acc.sum = acc.sum * std::pow(acc.scale / d, p) + Tv(len);
        acc.scale = d;
      }
      else
      {
        acc.sum += Tv(len) * std::pow(d / acc.scale, p);
      }
    }

    __host__ __device__ Tv result(const accumulator_type& acc) const
    {
      // The L-infinity distance is the scale itself
      if (acc.scale == Tv(0) || !(acc.scale <= max_value()) || p > max_value())
      {
        return acc.scale;
      }
      return acc.scale * std::pow(acc.sum, Tv(1) / p);
    }

  private:
    __host__ __device__ static constexpr Tv max_value()
    {
      return std::numeric_limits<Tv>::max();
    }
  };

  template <typename Tt, typename Tv>
  struct OperationL2InnerProduct
  {
    using accumulator_type = Tv;

    __host__ __device__ accumulator_type init() const
    {
      return Tv(0);
    }

    __host__ __device__ void add(accumulator_type& acc, Tt len, Tv f, Tv g) const
    {
      acc += len * (f * g);
    }

    __host__ __device__ Tv result(const accumulator_type& acc) const
    {
      return acc;
    }
  };
}

#endif
