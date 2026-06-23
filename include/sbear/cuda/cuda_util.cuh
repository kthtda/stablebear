#ifndef SB_CUDA_UTIL_H
#define SB_CUDA_UTIL_H

#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

#define CHK_CUDA(x) { auto rv = x; if (rv != cudaSuccess) { \
  throw std::runtime_error(__FILE__ + std::string(":") + std::to_string(__LINE__) + ": " +  cudaGetErrorString(rv)); \
  } }

#endif
