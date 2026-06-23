#ifndef STABLEBEAR_PY_ASYNC_SUPPORT_H
#define STABLEBEAR_PY_ASYNC_SUPPORT_H

#include <sbear/task.hpp>
#include <memory>
#include <utility>

namespace sb_py
{
  template <typename TaskT, typename... Args>
  [[nodiscard]] std::unique_ptr<TaskT> execute_stoppable_task(Args&&... args)
  {
    auto task = std::make_unique<TaskT>(std::forward<Args>(args)...);
    task->start_async(sb::default_executor());
    return task;
  }

  [[nodiscard]] inline std::unique_ptr<sb::EmptyTask<void>> execute_empty_task()
  {
    return execute_stoppable_task<sb::EmptyTask<void>>();
  }
}

#endif //STABLEBEAR_PY_ASYNC_SUPPORT_H
