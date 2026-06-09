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

#ifndef SB_CUDA_PCF_DATA_MANAGER_CUH
#define SB_CUDA_PCF_DATA_MANAGER_CUH

#include "cuda_offset_data_manager.cuh"
#include "cuda_matrix_integrate_structs.cuh"

namespace sb
{
  template <typename Tt, typename Tv>
  using CudaPcfDataManager = CudaOffsetDataManager<internal::SimplePoint<Tt, Tv>>;

  /// Initialize a CudaPcfDataManager from a range of PCFs.
  template <typename Tt, typename Tv, typename PcfFwdIt>
  void init_pcf_data(CudaPcfDataManager<Tt, Tv>& manager, PcfFwdIt begin, PcfFwdIt end)
  {
    using point_type = internal::SimplePoint<Tt, Tv>;

    manager.init(begin, end,
        [](const auto& f) { return f.points().size(); },
        [](const auto& f, size_t i) -> point_type {
          return { f.points()[i].t, f.points()[i].v };
        });
  }
}

#endif
