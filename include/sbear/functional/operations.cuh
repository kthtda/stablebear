#ifndef SB_OPERATIONS_CUH
#define SB_OPERATIONS_CUH

#include <cmath>

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
  template <typename Tt, typename Tv>
  struct OperationL1Dist
  {
    __host__ __device__ Tv operator()(Tv t, Tv b) const
    {
      return std::abs(t - b);
    }
    
    __host__ __device__ Tv operator()(Tv x) const
    {
      return x;
    }
  };
  
  template <typename Tt, typename Tv>
  struct OperationLpDist
  {
    Tv p = 2;
    OperationLpDist() = default;
    explicit OperationLpDist(Tv pv)
      : p(pv)
    { }
    
    __host__ __device__ Tv operator()(Tv t, Tv b) const
    {
      return std::pow(std::abs(t - b), p);
    }

    __host__ __device__ Tv operator()(Tv x) const
    {
      return std::pow(x, Tv(1) / p);
    }
  };
  
  template <typename Tt, typename Tv>
  struct OperationL2InnerProduct
  {
    __host__ __device__ Tv operator()(Tv t, Tv b) const
    {
      return t * b;
    }
    
    __host__ __device__ Tv operator()(Tv x) const
    {
      return x;
    }
  };
}

#endif
