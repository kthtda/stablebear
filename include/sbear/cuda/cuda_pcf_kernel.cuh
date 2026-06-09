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

// CUDA kernel for piecewise constant function integration.
// Contains the rectangle iteration device function, the block
// integration kernel, and its host-side launch wrapper.

#ifndef SB_CUDA_PCF_KERNEL_CUH
#define SB_CUDA_PCF_KERNEL_CUH

#include <cuda_runtime.h>

#include "cuda_matrix_integrate_structs.cuh"
#include "cuda_util.cuh"
#include "triangle_skip_mode.hpp"

namespace sb
{

  namespace internal
  {
    template <typename Tt, typename Tv>
    struct PcfBlockKernelParams
    {
      Tv* matrix;
      size_t* rowTimePointOffsets;
      size_t* colTimePointOffsets;
      SimplePoint<Tt, Tv>* rowPoints;
      SimplePoint<Tt, Tv>* colPoints;
      size_t nRows;
      size_t nCols;
      size_t globalRowStart;
      size_t globalColStart;
      TriangleSkipMode skipMode;
    };

    /// Walk two PCFs simultaneously through their breakpoints,
    /// calling cb(left, right, fValue, gValue) for each rectangle.
    template<typename Tt, typename Tv, typename RectangleCallback>
    __device__ void cuda_pcf_iterate_rectangles(
        const SimplePoint<Tt, Tv>* rowPoints, size_t fOffset, size_t fsz,
        const SimplePoint<Tt, Tv>* colPoints, size_t gOffset, size_t gsz,
        Tt a, Tt b, RectangleCallback cb)
    {
      Tt t = a;
      Tt tprev = t;

      Tv fv;
      Tv gv;

      size_t fi = 0;
      size_t gi = 0;

      const SimplePoint<Tt, Tv>* fpts = rowPoints + fOffset;
      const SimplePoint<Tt, Tv>* gpts = colPoints + gOffset;

      while (t < b)
      {
        tprev = t;
        fv = fpts[fi].v;
        gv = gpts[gi].v;

        if (fi + 1 < fsz && gi + 1 < gsz)
        {
          auto delta = fpts[fi + 1].t - gpts[gi + 1].t;
          if (delta <= 0)
          {
            ++fi;
          }
          if (delta >= 0)
          {
            ++gi;
          }
        }
        else
        {
          if (fi + 1 < fsz)
          {
            ++fi;
          }
          else if (gi + 1 < gsz)
          {
            ++gi;
          }
          else
          {
            cb(tprev, b, fv, gv);
            return;
          }
        }

        t = max(fpts[fi].t, gpts[gi].t);
        cb(tprev, t, fv, gv);
      }
    }

    /// CUDA kernel: integrate one PCF pair per thread across a 2D block.
    template <typename Tt, typename Tv, typename ComboOp>
    __global__
    void cuda_pcf_block_integrate(
        PcfBlockKernelParams<Tt, Tv> params,
        Tt a, Tt b, ComboOp op)
    {
      size_t iLocal = blockDim.x * blockIdx.x + threadIdx.x;
      size_t jLocal = blockDim.y * blockIdx.y + threadIdx.y;

      if (iLocal >= params.nRows || jLocal >= params.nCols)
      {
        return;
      }

      size_t iGlobal = iLocal + params.globalRowStart;
      size_t jGlobal = jLocal + params.globalColStart;

      if (params.skipMode == TriangleSkipMode::LowerTriangleSkipDiag && iGlobal <= jGlobal)
      {
        return;
      }
      if (params.skipMode == TriangleSkipMode::LowerTriangle && iGlobal < jGlobal)
      {
        return;
      }

      size_t fOffset = params.rowTimePointOffsets[iLocal];
      size_t fsz = params.rowTimePointOffsets[iLocal + 1] - fOffset;

      size_t gOffset = params.colTimePointOffsets[jLocal];
      size_t gsz = params.colTimePointOffsets[jLocal + 1] - gOffset;

      Tv ret = 0;
      cuda_pcf_iterate_rectangles<Tt, Tv>(
          params.rowPoints, fOffset, fsz,
          params.colPoints, gOffset, gsz,
          a, b, [&ret, op](Tt l, Tt r, Tv f, Tv g) {
            ret += (r - l) * op(f, g);
          });

      params.matrix[iLocal * params.nCols + jLocal] = op(ret);
    }

    /// Host-side launch wrapper.
    template <typename Tt, typename Tv, typename ComboOp>
    void launch_pcf_block_integrate(dim3 gridDim, dim3 blockDim, const PcfBlockKernelParams<Tt, Tv>& params, Tt a, Tt b, ComboOp op)
    {
      cuda_pcf_block_integrate<Tt, Tv, ComboOp><<<gridDim, blockDim>>>(params, a, b, op);
      CHK_CUDA(cudaGetLastError());
    }

  } // namespace internal

} // namespace sb

#endif
