#ifndef SB_CUDA_MATRIX_INTEGRATE_STRUCTS
#define SB_CUDA_MATRIX_INTEGRATE_STRUCTS

#include <cstddef>

namespace sb::internal
{
  // POD point type for piecewise constant functions on the GPU.
  template <typename Tt, typename Tv>
  struct SimplePoint
  {
    Tt t;
    Tv v;
  };
}

#endif
