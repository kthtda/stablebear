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

#ifndef STABLEBEAR_PY_FUTURE_H
#define STABLEBEAR_PY_FUTURE_H

#include "pybind.hpp"
#include <future>
#include <chrono>
#include <type_traits>
#include <utility>

namespace sb_py
{
  template <typename RetT>
  class Future
  {
  public:
    Future() = default;
    explicit Future(std::future<RetT>&& future)
      : m_future(std::move(future))
    {

    }

    Future(const Future&) = delete;
    Future(Future&& other) noexcept
      : m_future(std::move(other.m_future))
    { }

    Future& operator=(const Future&) = delete;
    Future& operator=(Future&& rhs) noexcept
    {
      m_future = std::move(rhs.m_future);
      return *this;
    }

    std::future_status wait_for(int timeoutMs)
    {
      return m_future.wait_for(std::chrono::milliseconds(timeoutMs));
    }

    auto get()
    {
      if constexpr (!std::is_same_v<RetT, void>)
      {
        return m_future.get();
      }
      else
      {
        m_future.get();
      }
    }

  private:
    std::future<RetT> m_future;
  };

  template <typename RetT>
  void register_bindings_future(pybind11::handle m, const std::string& suffix)
  {
    pybind11::class_<Future<RetT>>(m, ("Future" + suffix).c_str())
      .def(pybind11::init<>())
      .def("wait_for", &Future<RetT>::wait_for,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  }
}

#endif //STABLEBEAR_PY_FUTURE_H
