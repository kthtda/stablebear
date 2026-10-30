// Tests for the CUDA block pipeline: block execution, exceptions, stop
// requests and cancel(). Stub BlockOps run against the real pipeline, which
// needs a CUDA device because it allocates device buffers up front.

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
    // An exception in a block propagates out of execute()
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
    // A stop request made before execute() skips every block
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
