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

#ifndef SB_TASK_H
#define SB_TASK_H

#include <atomic>
#include <functional>
#include <iostream>

#include <taskflow/taskflow.hpp>

#include "executor.hpp"

namespace sb
{
  template <typename RetT>
  class StoppableTask
  {
  public:
    virtual ~StoppableTask() = default;

    void request_stop()
    {
      m_stop_requested.store(true);
      on_stop_requested();
      // m_future.cancel(); // Some issue with this at the moment.
    }
    
    bool stop_requested() const
    {
      return m_stop_requested.load();
    }
    
    const tf::Future<RetT>& future() const
    {
      return m_future;
    }
    
    tf::Future<RetT>& future()
    {
      return m_future;
    }
    
    StoppableTask& start_async(sb::Executor& exec)
    {
      m_stop_requested.store(false);
      m_work_completed.store(0ul);
      m_future = run_async(exec);
      return *this;
    }
    
    size_t work_completed() const
    {
      return m_work_completed.load();
    }
    
    size_t work_total() const
    {
      return m_work_total.load();
    }

    size_t work_step() const
    {
      return m_work_step.load();
    }

    std::string work_step_desc() const
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      return m_work_step_desc;
    }

    std::string work_step_unit() const
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      return m_work_step_unit;
    }

    bool wait_for(std::chrono::milliseconds duration)
    {
      return m_future.wait_for(duration) == std::future_status::ready;
    }
    
    void wait()
    {
      m_future.wait();
    }

  protected:
    void add_progress(size_t n_items)
    {
      std::atomic_fetch_add(&m_work_completed, n_items);
    }
    
    void next_step(size_t n_items, std::string_view desc, std::string_view unit)
    {
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_work_step.fetch_add(1);
        m_work_total.store(n_items);
        m_work_step_desc = desc;
        m_work_step_unit = unit;
        m_work_completed.store(0);
      }
    }

  private:
    virtual tf::Future<RetT> run_async(sb::Executor& exec) = 0;
    
    virtual void on_stop_requested() { }
    
    tf::Future<RetT> m_future;
    std::atomic_bool m_stop_requested = false;
    
    std::atomic_uint64_t m_work_total{0};
    std::atomic_uint64_t m_work_completed{0};

    std::atomic_uint64_t m_work_step{0};
    std::string m_work_step_desc;
    std::string m_work_step_unit;

    mutable std::mutex m_mutex;
    std::condition_variable m_condition;
  };

  template <typename RetT = void>
  class EmptyTask : public StoppableTask<RetT>
  {
  private:
    tf::Future<RetT> run_async(sb::Executor& /*exec*/) override
    {
      std::promise<RetT> p;
      if constexpr (std::is_same_v<RetT, void>)
      {
        p.set_value();
      }
      else
      {
        p.set_value(RetT{});
      }

      return tf::Future<void>(p.get_future());
    }
  };
}

#endif
