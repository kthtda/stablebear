// Generic CUDA block executor — function-type agnostic.
// Orchestrates block iteration, GPU output buffer management,
// result download, and scatter into the output matrix.
//
// Uses double buffering to overlap GPU computation with D2H
// transfer and CPU-side scatter of the previous block's results.
//
// Function-type-specific work (data upload, kernel launch) is
// delegated to a BlockOp policy class. See pcf_block_op.cuh for
// the PCF implementation.
//
// BlockOp concept (duck-typed):
//   typename BlockOp::GpuStorage
//     Per-GPU storage for function-specific device arrays. Move-constructible.
//
//   GpuStorage init_gpu_storage(size_t gpuId, const CudaBlockScheduler& scheduler)
//     Allocate function-specific GPU arrays for one GPU.
//
//   void exec_block(GpuStorage& storage, const BlockInfo& block,
//                   CudaDeviceArray<Tv>& gpuOutputMatrix, dim3 blockDim)
//     Upload data and launch kernel. Must NOT call cudaDeviceSynchronize —
//     the pipeline handles synchronization to enable overlap with CPU work.
//     Output matrix is pre-cleared.

#ifndef SB_CUDA_BLOCK_EXECUTOR_CUH
#define SB_CUDA_BLOCK_EXECUTOR_CUH

#include <cuda_runtime.h>

#include "cuda_block_scheduler.hpp"
#include "cuda_device_array.cuh"
#include "cuda_util.cuh"

#include <taskflow/taskflow.hpp>

#include <atomic>
#include <functional>
#include <iostream>
#include <vector>

namespace sb
{
  static constexpr size_t NUM_OUTPUT_BUFFERS = 2;

  /// Page-locked (pinned) host memory buffer for async D2H transfers.
  template <typename T>
  class PinnedHostBuffer
  {
  public:
    PinnedHostBuffer() = default;

    explicit PinnedHostBuffer(size_t count)
      : m_count(count)
    {
      if (count > 0)
      {
        CHK_CUDA(cudaMallocHost(&m_ptr, count * sizeof(T)));
      }
    }

    ~PinnedHostBuffer()
    {
      if (!m_ptr) return;
      auto rv = cudaFreeHost(m_ptr);
      if (rv != cudaSuccess)
      {
        std::cerr << "Warning: cudaFreeHost failed for " << m_count * sizeof(T) << " bytes" << std::endl;
      }
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
      : m_ptr(other.m_ptr), m_count(other.m_count)
    {
      other.m_ptr = nullptr;
      other.m_count = 0;
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& rhs) noexcept
    {
      if (this != &rhs)
      {
        if (m_ptr)
        {
          auto rv = cudaFreeHost(m_ptr);
          if (rv != cudaSuccess)
            std::cerr << "Warning: cudaFreeHost failed" << std::endl;
        }
        m_ptr = rhs.m_ptr;
        m_count = rhs.m_count;
        rhs.m_ptr = nullptr;
        rhs.m_count = 0;
      }
      return *this;
    }

    T* data() { return m_ptr; }
    const T* data() const { return m_ptr; }
    size_t size() const { return m_count; }

  private:
    T* m_ptr = nullptr;
    size_t m_count = 0;
  };

  template <typename Tv, typename BlockOp, typename ResultWriter>
  class CudaBlockPipeline
  {
  public:
    CudaBlockPipeline(
        tf::Executor& gpuThreads,
        BlockOp& blockOp,
        const CudaBlockScheduler& scheduler,
        ResultWriter& writer,
        std::function<void(size_t)> progressCb = [](size_t) {},
        std::function<bool()> stopRequested = [] { return false; })
      : m_gpuThreads(gpuThreads)
      , m_blockOp(blockOp)
      , m_scheduler(scheduler)
      , m_writer(writer)
      , m_progressCb(progressCb)
      , m_stopRequested(stopRequested)
      , m_nGpus(gpuThreads.num_workers())
    {
      m_canceled.store(false);
    }

    void execute(dim3 blockDim)
    {
      init_gpu_storage();

      auto const& blocks = m_scheduler.blocks();

      tf::Taskflow flow;
      flow.for_each_index<size_t, size_t, size_t>(0ul, blocks.size(), 1ul, [this, &blocks, blockDim](size_t i) {
        // Checked per block so a stop request (e.g. Ctrl-C via
        // StoppableTask::request_stop) interrupts a running computation at
        // block granularity instead of after the whole matrix.
        if (m_canceled.load() || m_stopRequested())
        {
          return;
        }
        exec_block(blocks[i], blockDim);
      });

      // wait_for_all() discards task exceptions; get() rethrows the first
      // CHK_CUDA failure (OOM, bad launch, ...) so it reaches the caller
      // instead of silently leaving the remaining blocks zeroed.
      m_gpuThreads.run(std::move(flow)).get();

      // Flush any pending scatter on each GPU
      for (size_t iGpu = 0; iGpu < m_nGpus; ++iGpu)
      {
        flush_pending_scatter(m_gpuStorages[iGpu]);
      }
    }

    void cancel()
    {
      m_canceled.store(true);
    }

  private:
    struct GpuBlockStorage
    {
      CudaDeviceArray<Tv> matrix[NUM_OUTPUT_BUFFERS];
      PinnedHostBuffer<Tv> hostScratch[NUM_OUTPUT_BUFFERS];
      typename BlockOp::GpuStorage opStorage;

      cudaStream_t downloadStream = nullptr;
      size_t currentBuffer = 0;
      bool hasPendingScatter = false;
      BlockInfo pendingBlock{};
      size_t pendingBuffer = 0;

      GpuBlockStorage() = default;
      GpuBlockStorage(const GpuBlockStorage&) = delete;
      GpuBlockStorage& operator=(const GpuBlockStorage&) = delete;

      GpuBlockStorage(GpuBlockStorage&& other)
        : opStorage(std::move(other.opStorage))
        , downloadStream(other.downloadStream)
        , currentBuffer(other.currentBuffer)
        , hasPendingScatter(other.hasPendingScatter)
        , pendingBlock(other.pendingBlock)
        , pendingBuffer(other.pendingBuffer)
      {
        for (size_t i = 0; i < NUM_OUTPUT_BUFFERS; ++i)
        {
          matrix[i] = std::move(other.matrix[i]);
          hostScratch[i] = std::move(other.hostScratch[i]);
        }
        other.downloadStream = nullptr;
      }

      GpuBlockStorage& operator=(GpuBlockStorage&& rhs)
      {
        if (this == &rhs) return *this;
        for (size_t i = 0; i < NUM_OUTPUT_BUFFERS; ++i)
        {
          matrix[i] = std::move(rhs.matrix[i]);
          hostScratch[i] = std::move(rhs.hostScratch[i]);
        }
        opStorage = std::move(rhs.opStorage);
        downloadStream = rhs.downloadStream;
        currentBuffer = rhs.currentBuffer;
        hasPendingScatter = rhs.hasPendingScatter;
        pendingBlock = rhs.pendingBlock;
        pendingBuffer = rhs.pendingBuffer;
        rhs.downloadStream = nullptr;
        return *this;
      }

      ~GpuBlockStorage()
      {
        if (!downloadStream) return;
        auto rv = cudaStreamDestroy(downloadStream);
        if (rv != cudaSuccess)
        {
          std::cerr << "Warning: cudaStreamDestroy failed" << std::endl;
        }
      }
    };

    void init_gpu_storage()
    {
      m_gpuStorages.resize(m_nGpus);

      auto maxRowHeight = m_scheduler.max_row_height();
      auto maxColWidth = m_scheduler.max_col_width();
      auto maxEntries = maxRowHeight * maxColWidth;

      for (size_t iGpu = 0; iGpu < m_nGpus; ++iGpu)
      {
        CHK_CUDA(cudaSetDevice(static_cast<int>(iGpu)));

        auto& storage = m_gpuStorages[iGpu];
        for (size_t buf = 0; buf < NUM_OUTPUT_BUFFERS; ++buf)
        {
          storage.matrix[buf] = CudaDeviceArray<Tv>(maxEntries);
          storage.hostScratch[buf] = PinnedHostBuffer<Tv>(maxEntries);
        }
        storage.opStorage = m_blockOp.init_gpu_storage(iGpu, m_scheduler);

        CHK_CUDA(cudaStreamCreate(&storage.downloadStream));
        storage.currentBuffer = 0;
        storage.hasPendingScatter = false;
      }
    }

    void flush_pending_scatter(GpuBlockStorage& storage)
    {
      if (!storage.hasPendingScatter)
      {
        return;
      }

      CHK_CUDA(cudaStreamSynchronize(storage.downloadStream));

      m_writer.scatter(
          storage.hostScratch[storage.pendingBuffer].data(),
          storage.pendingBlock);

      m_progressCb(count_pairs(storage.pendingBlock));

      storage.hasPendingScatter = false;
    }

    void exec_block(const BlockInfo& block, dim3 blockDim)
    {
      auto iGpu = m_gpuThreads.this_worker_id();
      CHK_CUDA(cudaSetDevice(static_cast<int>(iGpu)));

      auto& storage = m_gpuStorages[iGpu];
      size_t buf = storage.currentBuffer;

      // Clear output buffer and launch kernel (async — returns before kernel finishes)
      storage.matrix[buf].clear();
      m_blockOp.exec_block(storage.opStorage, block, storage.matrix[buf], blockDim);

      // While the kernel runs on the GPU, scatter the previous block's
      // results on the CPU. This is the core double-buffering overlap.
      flush_pending_scatter(storage);

      // Wait for this block's kernel to finish
      CHK_CUDA(cudaDeviceSynchronize());

      // Start async D2H copy of this block's results
      auto nEntries = block.rowHeight * block.colWidth;
      storage.matrix[buf].toHostAsync(
          storage.downloadStream,
          storage.hostScratch[buf].data(),
          nEntries);

      // Record pending scatter for this block
      storage.hasPendingScatter = true;
      storage.pendingBlock = block;
      storage.pendingBuffer = buf;

      // Alternate buffer for next block
      storage.currentBuffer = (buf + 1) % NUM_OUTPUT_BUFFERS;
    }

    size_t count_pairs(const BlockInfo& block) const
    {
      if (m_scheduler.triangle_mode() == BlockTriangleMode::Full)
      {
        return block.rowHeight * block.colWidth;
      }

      // Lower triangle: count pairs where iGlobal > jGlobal
      size_t rowEnd = block.rowStart + block.rowHeight;
      size_t colEnd = block.colStart + block.colWidth;

      if (block.rowStart >= colEnd)
      {
        return block.rowHeight * block.colWidth;
      }
      if (rowEnd <= block.colStart)
      {
        return 0;
      }

      // Block straddles the diagonal
      size_t pairs = 0;
      for (size_t i = block.rowStart; i < rowEnd; ++i)
      {
        size_t jEnd = std::min(i, colEnd);
        if (jEnd > block.colStart)
        {
          pairs += jEnd - block.colStart;
        }
      }
      return pairs;
    }

    tf::Executor& m_gpuThreads;
    BlockOp& m_blockOp;
    const CudaBlockScheduler& m_scheduler;
    ResultWriter& m_writer;
    std::function<void(size_t)> m_progressCb;
    std::function<bool()> m_stopRequested;
    size_t m_nGpus;

    std::vector<GpuBlockStorage> m_gpuStorages;
    std::atomic_bool m_canceled;
  };

} // namespace sb

#endif
