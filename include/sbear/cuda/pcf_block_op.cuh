// PCF-specific block operation for the CUDA block executor.
// Contains the piecewise-constant rectangle iteration kernel
// and the PcfBlockOp policy class.

#ifndef SB_CUDA_PCF_BLOCK_OP_CUH
#define SB_CUDA_PCF_BLOCK_OP_CUH

#include <cuda_runtime.h>

#include "../task.hpp"
#include "../executor.hpp"
#include "../settings.hpp"
#include "../functional/operations.cuh"
#include "../block_matrix_support.cuh"
#include "cuda_block_executor.cuh"
#include "cuda_block_scheduler.hpp"
#include "cuda_pcf_kernel.cuh"
#include "cuda_pcf_data_manager.cuh"
#include "cuda_device_array.cuh"
#include "cuda_util.cuh"

#include <limits>
#include <string>
#include <vector>

namespace sb
{

  /// BlockOp policy for piecewise constant functions (PCFs).
  /// Handles PCF data upload and kernel launch for the generic CudaBlockPipeline.
  /// Supports separate row/col data sources (for cdist) or the same source (for pdist).
  template <typename Tt, typename Tv, typename ComboOp>
  class PcfBlockOp
  {
  public:
    using point_type = internal::SimplePoint<Tt, Tv>;

    struct GpuStorage
    {
      CudaDeviceArray<size_t> rowOffsets;
      CudaDeviceArray<point_type> rowPoints;
      CudaDeviceArray<size_t> colOffsets;
      CudaDeviceArray<point_type> colPoints;

      GpuStorage() = default;
      GpuStorage(GpuStorage&&) = default;
      GpuStorage& operator=(GpuStorage&&) = default;
    };

    PcfBlockOp(CudaPcfDataManager<Tt, Tv>& rowDataManager,
               CudaPcfDataManager<Tt, Tv>& colDataManager,
               ComboOp op, Tt a, Tt b, TriangleSkipMode skipMode)
      : m_rowDataManager(rowDataManager)
      , m_colDataManager(colDataManager)
      , m_op(op)
      , m_a(a)
      , m_b(b)
      , m_skipMode(skipMode)
    { }

    GpuStorage init_gpu_storage(size_t gpuId, const CudaBlockScheduler& scheduler)
    {
      size_t maxRowHeight = scheduler.max_row_height();
      size_t maxColWidth = scheduler.max_col_width();

      size_t maxRowPoints = 0;
      size_t maxColPoints = 0;
      for (auto const& block : scheduler.blocks())
      {
        maxRowPoints = std::max(maxRowPoints, m_rowDataManager.total_elements_for_range(block.rowStart, block.rowHeight));
        maxColPoints = std::max(maxColPoints, m_colDataManager.total_elements_for_range(block.colStart, block.colWidth));
      }

      CHK_CUDA(cudaSetDevice(static_cast<int>(gpuId)));

      GpuStorage storage;
      storage.rowOffsets = CudaDeviceArray<size_t>(maxRowHeight + 1);
      storage.rowPoints = CudaDeviceArray<point_type>(maxRowPoints);
      storage.colOffsets = CudaDeviceArray<size_t>(maxColWidth + 1);
      storage.colPoints = CudaDeviceArray<point_type>(maxColPoints);
      return storage;
    }

    void exec_block(GpuStorage& storage, const BlockInfo& block,
                    CudaDeviceArray<Tv>& gpuOutputMatrix, dim3 blockDim)
    {
      int gpuId;
      CHK_CUDA(cudaGetDevice(&gpuId));

      m_rowDataManager.upload_subset(
          gpuId, block.rowStart, block.rowHeight,
          storage.rowOffsets, storage.rowPoints);

      m_colDataManager.upload_subset(
          gpuId, block.colStart, block.colWidth,
          storage.colOffsets, storage.colPoints);

      internal::PcfBlockKernelParams<Tt, Tv> params;
      params.matrix = gpuOutputMatrix.get();
      params.rowTimePointOffsets = storage.rowOffsets.get();
      params.colTimePointOffsets = storage.colOffsets.get();
      params.rowPoints = storage.rowPoints.get();
      params.colPoints = storage.colPoints.get();
      params.nRows = block.rowHeight;
      params.nCols = block.colWidth;
      params.globalRowStart = block.rowStart;
      params.globalColStart = block.colStart;
      params.skipMode = m_skipMode;

      dim3 gridDim = internal::get_grid_dims(blockDim, block.rowHeight, block.colWidth);

      internal::launch_pcf_block_integrate<Tt, Tv>(gridDim, blockDim, params, m_a, m_b, m_op);
      CHK_CUDA(cudaPeekAtLastError());
    }

  private:
    CudaPcfDataManager<Tt, Tv>& m_rowDataManager;
    CudaPcfDataManager<Tt, Tv>& m_colDataManager;
    ComboOp m_op;
    Tt m_a;
    Tt m_b;
    TriangleSkipMode m_skipMode;
  };

  /// CUDA pairwise integration task (pdist / l2_kernel).
  /// Computes lower triangle from a single set of functions.
  template <typename PcfFwdIt, typename ComboOp, typename ResultWriter>
  class CudaPairwiseIntegrationTask : public StoppableTask<void>
  {
  public:
    using pcf_type = typename PcfFwdIt::value_type;
    using time_type = typename pcf_type::time_type;
    using value_type = typename pcf_type::value_type;

    CudaPairwiseIntegrationTask(
        tf::Executor& cudaThreads,
        ResultWriter writer,
        PcfFwdIt beginPcfs, PcfFwdIt endPcfs,
        ComboOp op, time_type a, time_type b,
        TriangleSkipMode skipMode,
        std::string stepDescription = "Computing lower triangle",
        std::string stepUnit = "integral")
      : m_fs(beginPcfs, endPcfs)
      , m_op(op)
      , m_a(a)
      , m_b(b)
      , m_writer(std::move(writer))
      , m_cudaThreads(cudaThreads)
      , m_skipMode(skipMode)
      , m_stepDescription(std::move(stepDescription))
      , m_stepUnit(std::move(stepUnit))
    { }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      init_pcf_data(m_dataManager, m_fs.begin(), m_fs.end());

      auto& s = sb::settings();
      auto n = m_fs.size();
      auto nGpus = m_cudaThreads.num_workers();
      size_t maxOutputElements = internal::get_max_output_elements(nGpus, sizeof(value_type));
      size_t minSide = s.minBlockSide > 0 ? s.minBlockSide : internal::get_min_block_side(nGpus);

      CudaBlockScheduler scheduler({
        .nRows = n, .nCols = n,
        .maxOutputElements = maxOutputElements,
        .nSplitsHint = nGpus * 32,
        .triangleMode = BlockTriangleMode::LowerTriangle,
        .minBlockSide = minSide
      });

      next_step(n * (n - 1) / 2, m_stepDescription, m_stepUnit);

      using block_op_t = PcfBlockOp<time_type, value_type, ComboOp>;

      dim3 blockDim(s.blockDimX, s.blockDimY, 1);
      tf::Taskflow flow;
      flow.emplace([this, scheduler = std::move(scheduler), blockDim] {
        block_op_t blockOp(m_dataManager, m_dataManager, m_op, m_a, m_b, m_skipMode);

        CudaBlockPipeline<value_type, block_op_t, ResultWriter> pipeline(
            m_cudaThreads, blockOp, scheduler, m_writer,
            [this](size_t n) { add_progress(n); });

        if (stop_requested()) return;
        pipeline.execute(blockDim);
      });

      return exec.cpu()->run(std::move(flow));
    }

    std::vector<pcf_type> m_fs;
    ComboOp m_op;
    time_type m_a;
    time_type m_b;
    ResultWriter m_writer;
    tf::Executor& m_cudaThreads;
    TriangleSkipMode m_skipMode;
    std::string m_stepDescription;
    std::string m_stepUnit;
    CudaPcfDataManager<time_type, value_type> m_dataManager;
  };

  /// CUDA cross-distance integration task (cdist).
  /// Computes all pairs between two separate function sets.
  template <typename PcfFwdIt, typename ComboOp, typename ResultWriter>
  class CudaCrossIntegrationTask : public StoppableTask<void>
  {
  public:
    using pcf_type = typename PcfFwdIt::value_type;
    using time_type = typename pcf_type::time_type;
    using value_type = typename pcf_type::value_type;

    CudaCrossIntegrationTask(
        tf::Executor& cudaThreads,
        ResultWriter writer,
        PcfFwdIt beginRows, PcfFwdIt endRows,
        PcfFwdIt beginCols, PcfFwdIt endCols,
        ComboOp op, time_type a, time_type b,
        std::string stepDescription = "Computing cross-distances",
        std::string stepUnit = "integral")
      : m_rowFs(beginRows, endRows)
      , m_colFs(beginCols, endCols)
      , m_op(op)
      , m_a(a)
      , m_b(b)
      , m_writer(std::move(writer))
      , m_cudaThreads(cudaThreads)
      , m_stepDescription(std::move(stepDescription))
      , m_stepUnit(std::move(stepUnit))
    { }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      init_pcf_data(m_rowDataManager, m_rowFs.begin(), m_rowFs.end());
      init_pcf_data(m_colDataManager, m_colFs.begin(), m_colFs.end());

      auto& s = sb::settings();
      auto nRows = m_rowFs.size();
      auto nCols = m_colFs.size();
      auto nGpus = m_cudaThreads.num_workers();
      size_t maxOutputElements = internal::get_max_output_elements(nGpus, sizeof(value_type));
      size_t minSide = s.minBlockSide > 0 ? s.minBlockSide : internal::get_min_block_side(nGpus);

      CudaBlockScheduler scheduler({
        .nRows = nRows, .nCols = nCols,
        .maxOutputElements = maxOutputElements,
        .nSplitsHint = nGpus * 32,
        .triangleMode = BlockTriangleMode::Full,
        .minBlockSide = minSide
      });

      next_step(nRows * nCols, m_stepDescription, m_stepUnit);

      using block_op_t = PcfBlockOp<time_type, value_type, ComboOp>;

      dim3 blockDim(s.blockDimX, s.blockDimY, 1);
      tf::Taskflow flow;
      flow.emplace([this, scheduler = std::move(scheduler), blockDim] {
        block_op_t blockOp(m_rowDataManager, m_colDataManager, m_op, m_a, m_b, TriangleSkipMode::None);

        CudaBlockPipeline<value_type, block_op_t, ResultWriter> pipeline(
            m_cudaThreads, blockOp, scheduler, m_writer,
            [this](size_t n) { add_progress(n); });

        if (stop_requested()) return;
        pipeline.execute(blockDim);
      });

      return exec.cpu()->run(std::move(flow));
    }

    std::vector<pcf_type> m_rowFs;
    std::vector<pcf_type> m_colFs;
    ComboOp m_op;
    time_type m_a;
    time_type m_b;
    ResultWriter m_writer;
    tf::Executor& m_cudaThreads;
    std::string m_stepDescription;
    std::string m_stepUnit;
    CudaPcfDataManager<time_type, value_type> m_rowDataManager;
    CudaPcfDataManager<time_type, value_type> m_colDataManager;
  };

} // namespace sb

#endif
