#include <sbear/executor.hpp>
#include <taskflow/taskflow.hpp>

#include <stdexcept>

#ifdef BUILD_WITH_CUDA
#pragma message("Building stablebear C++ extension with CUDA")
#else
#pragma message("Building stablebear C++ extension without CUDA")
#endif

#ifdef BUILD_WITH_CUDA
#include <cuda_runtime.h>
#include <sbear/cuda/cuda_util.cuh>
#endif

size_t sb::get_num_cuda_devices()
{
  int nGpus = 0;
#ifdef BUILD_WITH_CUDA
  if (cudaGetDeviceCount(&nGpus) != cudaSuccess)
  {
    return 0;
  }
  if (nGpus < 0)
  {
    // Just in case...
    throw std::runtime_error("Negative number (" + std::to_string(nGpus) + ") of GPUs reported!");
  }
#endif
  return static_cast<size_t>(nGpus);
}

sb::Executor& sb::default_executor()
{
  static Executor exec = Executor(std::thread::hardware_concurrency(), get_num_cuda_devices());
  return exec;
}
