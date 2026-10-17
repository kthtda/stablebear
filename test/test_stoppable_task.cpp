#include <gtest/gtest.h>

#include <sbear/executor.hpp>
#include <sbear/task.hpp>

#include <taskflow/algorithm/for_each.hpp>

#include <chrono>
#include <stdexcept>

namespace
{
  /// Runs a parallel loop in which the worker for index 2 throws.
  class FailingTask : public sb::StoppableTask<void>
  {
  private:
    tf::Future<void> run_async(sb::Executor& exec) override
    {
      tf::Taskflow flow;
      flow.for_each_index(size_t{0}, size_t{4}, size_t{1}, [](size_t i) {
        if (i == 2)
        {
          throw std::runtime_error("worker failed on 2");
        }
      });
      return exec.cpu()->run(std::move(flow));
    }
  };
}

TEST(StoppableTask, WaitRethrowsAWorkerExceptionOnce)
{
  FailingTask task;
  task.start_async(sb::default_executor());

  try
  {
    task.wait();
    FAIL() << "wait() did not rethrow the worker's exception";
  }
  catch (const std::runtime_error& error)
  {
    EXPECT_STREQ(error.what(), "worker failed on 2");
  }
  EXPECT_NO_THROW(task.wait());
}

TEST(StoppableTask, WaitForRethrowsAWorkerExceptionOnceFinished)
{
  FailingTask task;
  task.start_async(sb::default_executor());

  try
  {
    while (!task.wait_for(std::chrono::milliseconds(10)))
    {
    }
    FAIL() << "wait_for() did not rethrow the worker's exception";
  }
  catch (const std::runtime_error& error)
  {
    EXPECT_STREQ(error.what(), "worker failed on 2");
  }
}
