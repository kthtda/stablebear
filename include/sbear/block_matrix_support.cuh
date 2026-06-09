/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef SB_BLOCK_MATRIX_SUPPORT_CUH
#define SB_BLOCK_MATRIX_SUPPORT_CUH

#include <cuda_runtime.h> // dim3
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include <sbear/algorithms/subdivide.hpp>

namespace sb::internal
{
  inline size_t get_row_size(size_t maxAllocationN, size_t nSplits, size_t nPcfs)
  {
    size_t maxRowHeight = maxAllocationN / nPcfs;
    
    maxRowHeight = std::min(maxRowHeight, nPcfs);
    maxRowHeight /= nSplits;
    maxRowHeight = std::max<size_t>(maxRowHeight, 1ul);
    
    return maxRowHeight;
  }
  
  inline dim3 get_grid_dims(dim3 blockSz, size_t rowHeight, size_t nPcfs)
  {
    size_t iover = rowHeight % blockSz.x == 0 ? 0 : 1;
    size_t jover = nPcfs % blockSz.y == 0 ? 0 : 1;
    
    auto x = rowHeight / static_cast<size_t>(blockSz.x) + iover;
    auto y = nPcfs / static_cast<size_t>(blockSz.y) + jover;

    return dim3(static_cast<unsigned int>(x), static_cast<unsigned int>(y), 1);
  }
  
  inline size_t get_row_height_from_boundaries(std::pair<size_t, size_t> boundaries)
  {
    return boundaries.second - boundaries.first + 1;
  }

  /// Query the minimum SM count across nGpus devices and derive a minimum
  /// block side length that ensures reasonable GPU occupancy.
  /// Targets ~50% max occupancy (smCount * 1024 threads).
  inline size_t get_min_block_side(size_t nGpus)
  {
    int minSmCount = std::numeric_limits<int>::max();

    for (size_t i = 0; i < nGpus; ++i)
    {
      auto setErr = cudaSetDevice(static_cast<int>(i));
      if (setErr != cudaSuccess)
        throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(setErr));

      int smCount = 0;
      auto attrErr = cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, static_cast<int>(i));
      if (attrErr != cudaSuccess)
        throw std::runtime_error(std::string("cudaDeviceGetAttribute failed: ") + cudaGetErrorString(attrErr));

      minSmCount = std::min(minSmCount, smCount);
    }

    size_t targetThreads = static_cast<size_t>(minSmCount) * 1024;
    return static_cast<size_t>(std::sqrt(static_cast<double>(targetThreads)));
  }

  /// Query GPU memory across nGpus devices and return the maximum number of
  /// elements of size elementSize that can be used for block output buffers.
  inline size_t get_max_output_elements(size_t nGpus, size_t elementSize)
  {
    constexpr float allocationPct = 0.8f;
    size_t retVal = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < nGpus; ++i)
    {
      auto setErr = cudaSetDevice(static_cast<int>(i));
      if (setErr != cudaSuccess)
        throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(setErr));

      size_t freeMem, totalMem;
      auto memErr = cudaMemGetInfo(&freeMem, &totalMem);
      if (memErr != cudaSuccess)
        throw std::runtime_error(std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(memErr));

      size_t maxAllocSz = static_cast<size_t>(static_cast<float>(freeMem) * allocationPct);
      size_t maxN = (maxAllocSz / 2) / elementSize;
      maxN = (maxN / 1024) * 1024;

      retVal = std::min(retVal, maxN);
    }

    return retVal;
  }

}

inline bool operator==(const dim3& a, const dim3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}



#endif
