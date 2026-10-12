#ifndef STABLEBEAR_PY_ASYNC_SUPPORT_H
#define STABLEBEAR_PY_ASYNC_SUPPORT_H

#include <sbear/task.hpp>

#include "pybind.hpp"

#include <memory>
#include <string>
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

  /// Bind a task class that exposes its output through result(). The task
  /// must have been started; result() is complete once the task has finished.
  template <typename TaskT>
  void register_result_task(pybind11::module_& m, const std::string& name)
  {
    pybind11::class_<TaskT, sb::StoppableTask<void>>(m, name.c_str())
      .def("result", [](const TaskT& task) { return task.result(); });
  }
}

#endif //STABLEBEAR_PY_ASYNC_SUPPORT_H
