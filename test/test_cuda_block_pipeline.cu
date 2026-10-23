// Regression tests for the CUDA block pipeline (issues #188 and #195):
// - exceptions thrown inside exec_block must propagate out of execute()
//   instead of being swallowed by wait_for_all (issue #188);
// - a stop request must interrupt the pipeline at block granularity
//   (issue #195), and cancel() must keep working.
//
// Uses stub BlockOps against the real pipeline so the behavior is exercised
// without depending on kernel contents. Requires a CUDA device (the pipeline
// allocates device buffers up front).

#include <gtest/gtest.h>

#ifdef BUILD_WITH_CUDA

#include <sbear/cuda/cuda_block_executor.cuh>
#include <sbear/cuda/cuda_block_scheduler.hpp>
#include <sbear/executor.hpp>

#include <atomic>
#include <stdexcept>

namespace
{

  struct NullWriter
  {
    void scatter(const float*, const sb::BlockInfo&) {}
  };

  // exec_block launches nothing; it just counts invocations.
  struct CountingBlockOp
  {
    struct GpuStorage {};

    GpuStorage init_gpu_storage(size_t, const sb::CudaBlockScheduler&) { return {}; }

    void exec_block(GpuStorage&, const sb::BlockInfo&, sb::CudaDeviceArray<float>&, dim3)
    {
      ++calls;
    }

    std::atomic<int> calls{0};
  };

  struct ThrowingBlockOp
  {
    struct GpuStorage {};

    GpuStorage init_gpu_storage(size_t, const sb::CudaBlockScheduler&) { return {}; }

    void exec_block(GpuStorage&, const sb::BlockInfo&, sb::CudaDeviceArray<float>&, dim3)
    {
      throw std::runtime_error("simulated CUDA failure");
    }
  };

  class CudaBlockPipelineTest : public ::testing::Test
  {
  protected:
    void SetUp() override
    {
      if (sb::get_num_cuda_devices() == 0)
      {
        GTEST_SKIP() << "No CUDA devices available";
      }
    }

    static sb::CudaBlockScheduler make_scheduler()
    {
      return sb::CudaBlockScheduler({
          .nRows = 8, .nCols = 8,
          .maxOutputElements = 16,
          .nSplitsHint = 4,
          .triangleMode = sb::BlockTriangleMode::Full,
          .minBlockSide = 2
      });
    }
  };

  TEST_F(CudaBlockPipelineTest, ExecutesEveryBlock)
  {
    auto scheduler = make_scheduler();
    ASSERT_GT(scheduler.blocks().size(), 1u);

    tf::Executor gpuThreads(1);
    CountingBlockOp op;
    NullWriter writer;

    sb::CudaBlockPipeline<float, CountingBlockOp, NullWriter> pipeline(
        gpuThreads, op, scheduler, writer);
    pipeline.execute(dim3(8, 8, 1));

    EXPECT_EQ(op.calls.load(), static_cast<int>(scheduler.blocks().size()));
  }

  TEST_F(CudaBlockPipelineTest, ExceptionInBlockPropagatesFromExecute)
  {
    // Issue #188: the future from run() used to be discarded, so worker
    // exceptions vanished and execute() returned normally.
    auto scheduler = make_scheduler();

    tf::Executor gpuThreads(1);
    ThrowingBlockOp op;
    NullWriter writer;

    sb::CudaBlockPipeline<float, ThrowingBlockOp, NullWriter> pipeline(
        gpuThreads, op, scheduler, writer);

    EXPECT_THROW(pipeline.execute(dim3(8, 8, 1)), std::runtime_error);
  }

  TEST_F(CudaBlockPipelineTest, StopRequestSkipsRemainingBlocks)
  {
    // Issue #195: a stop request is polled per block; requested up front,
    // no block may execute.
    auto scheduler = make_scheduler();

    tf::Executor gpuThreads(1);
    CountingBlockOp op;
    NullWriter writer;

    sb::CudaBlockPipeline<float, CountingBlockOp, NullWriter> pipeline(
        gpuThreads, op, scheduler, writer,
        [](size_t) {}, [] { return true; });
    pipeline.execute(dim3(8, 8, 1));

    EXPECT_EQ(op.calls.load(), 0);
  }

  TEST_F(CudaBlockPipelineTest, CancelSkipsRemainingBlocks)
  {
    auto scheduler = make_scheduler();

    tf::Executor gpuThreads(1);
    CountingBlockOp op;
    NullWriter writer;

    sb::CudaBlockPipeline<float, CountingBlockOp, NullWriter> pipeline(
        gpuThreads, op, scheduler, writer);
    pipeline.cancel();
    pipeline.execute(dim3(8, 8, 1));

    EXPECT_EQ(op.calls.load(), 0);
  }

} // namespace

#endif // BUILD_WITH_CUDA
