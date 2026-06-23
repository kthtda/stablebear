#ifndef SB_CUDA_FUNCTIONAL_SUPPORT_H
#define SB_CUDA_FUNCTIONAL_SUPPORT_H

#include "cuda_util.cuh"

#include <cuda_runtime.h>
#include <iostream>

namespace sb::detail
{
  // https://stackoverflow.com/a/65393822 (user: jakob) (CC-BY-SA 4.0)
  // https://stackoverflow.com/a/9001502  (user: brano) (CC-BY-SA 3.0)
  template <typename TDeviceFuncPtr>
  struct CudaCallableFunctionPointer
  {
  public:
    explicit CudaCallableFunctionPointer(TDeviceFuncPtr* pf)
    {
      TDeviceFuncPtr hostPtr = nullptr;
      CHK_CUDA(cudaMalloc((void**)&m_ptr, sizeof(TDeviceFuncPtr)));
      CHK_CUDA(cudaMemcpyFromSymbol(&hostPtr, *pf, sizeof(TDeviceFuncPtr)));
      CHK_CUDA(cudaMemcpy(m_ptr, &hostPtr, sizeof(TDeviceFuncPtr), cudaMemcpyHostToDevice));
    }

    ~CudaCallableFunctionPointer()
    {
      auto success = cudaFree(m_ptr); // no-op if m_ptr == nullptr
      if (success != cudaSuccess)
      {
        std::cerr << "Warning! Unable to free " << sizeof(TDeviceFuncPtr) << " bytes of GPU memory\n";
      }
    }

    CudaCallableFunctionPointer(const CudaCallableFunctionPointer&) = delete;
    CudaCallableFunctionPointer& operator=(const CudaCallableFunctionPointer&) = delete;

    [[nodiscard]] TDeviceFuncPtr* get() const { return m_ptr; }

  private:
    TDeviceFuncPtr* m_ptr = nullptr;
  };

}

#endif