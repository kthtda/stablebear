// Generic GPU data manager for variable-length objects.
//
// Host-side logic (init, queries, offset re-indexing) is in
// OffsetDataManager — pure C++, no CUDA dependency.
//
// CudaOffsetDataManager extends it with upload_subset() for GPU transfers.

#ifndef SB_CUDA_OFFSET_DATA_MANAGER_CUH
#define SB_CUDA_OFFSET_DATA_MANAGER_CUH

#include "cuda_device_array.cuh"
#include "cuda_util.cuh"
#include "offset_data_manager.hpp"

namespace sb
{
  /// Extends OffsetDataManager with CUDA device upload capability.
  template <typename ElementT>
  class CudaOffsetDataManager : public OffsetDataManager<ElementT>
  {
  public:
    /// Upload a contiguous subset of objects [start, start+count) to device arrays.
    /// Offsets are re-indexed to 0-based for the subset.
    void upload_subset(
        int gpuId,
        size_t start, size_t count,
        CudaDeviceArray<size_t>& deviceOffsets,
        CudaDeviceArray<ElementT>& deviceElements) const
    {
      CHK_CUDA(cudaSetDevice(gpuId));

      auto const& hostData = this->host_data();
      size_t baseOffset = hostData.offsets[start];
      size_t nElements = hostData.offsets[start + count] - baseOffset;

      // Local scratch: one manager instance is shared by all GPU worker
      // threads, so this must not be shared mutable state.
      std::vector<size_t> localOffsets(count + 1);
      for (size_t i = 0; i <= count; ++i)
      {
        localOffsets[i] = hostData.offsets[start + i] - baseOffset;
      }

      deviceOffsets.toDevice(localOffsets.data(), count + 1);
      deviceElements.toDevice(hostData.elements.data() + baseOffset, nElements);
    }
  };
}

#endif
