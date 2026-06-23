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
