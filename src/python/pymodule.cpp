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

#include "pybind.hpp"

#include <mpcf/executor.hpp>
#include <mpcf/task.hpp>

#include "py_future.hpp"
#include "functional/py_pcf.hpp"
#include "py_io.hpp"
#include "functional/py_make_from_serial_content.hpp"
#include "functional/py_norms.hpp"
#include "py_tensor.hpp"
#include "functional/py_reductions.hpp"
#include "functional/py_distance.hpp"
#include "functional/py_inner_product.hpp"
#include "functional/py_random.hpp"
#include "py_np_tensor_convert.hpp"
#include "py_symmetric_matrix.hpp"
#include "py_distance_matrix.hpp"

#include "persistence/pymodule_persistence.hpp"
#include "point_process/pymodule_point_process.hpp"

#ifdef BUILD_WITH_CUDA
#include <cuda_runtime.h>
#include <mpcf/cuda/cuda_matrix_integrate_api.hpp>
#endif

#include <mpcf/settings.hpp>

namespace py = pybind11;

namespace
{

  // A test-only task that blocks until Python calls advance() n_steps times.
  // Used to verify the GIL is released during wait_for: if it isn't, the
  // thread calling wait_for holds the GIL, Python can never call advance(),
  // and the test deadlocks.
  class GatedTask : public mpcf::StoppableTask<void>
  {
  public:
    explicit GatedTask(size_t n_steps) : m_remaining(n_steps) { }

    void advance()
    {
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_remaining > 0)
          --m_remaining;
      }
      m_cv.notify_one();
    }

  private:
    tf::Future<void> run_async(mpcf::Executor& exec) override
    {
      next_step(m_remaining, "Waiting for gate", "step");
      m_flow.emplace([this]() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this]() { return m_remaining == 0 || stop_requested(); });
      });
      return exec.cpu()->run(std::move(m_flow));
    }

    void on_stop_requested() override
    {
      m_cv.notify_one();
    }

    std::mutex m_mutex;
    std::condition_variable m_cv;
    size_t m_remaining;
    tf::Taskflow m_flow;
  };

  int getNumGpus()
  {
#ifdef BUILD_WITH_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess)
    {
      throw std::runtime_error(cudaGetErrorString(error));
    }
    return deviceCount;
#else
    throw std::runtime_error("This version of masspcf is compiled without GPU support.");
#endif
  }

  template <typename RetT>
  static void register_bindings_stoppable_task(py::handle m, const std::string& suffix)
  {
    py::class_<mpcf::StoppableTask<RetT>> cls(m, ("StoppableTask" + suffix).c_str());

    cls
        .def("request_stop", &mpcf::StoppableTask<RetT>::request_stop)
        .def("wait_for", [](mpcf::StoppableTask<RetT>& self, int ms) { return self.wait_for(std::chrono::milliseconds(ms)); },
             py::call_guard<py::gil_scoped_release>())
        .def("work_total", &mpcf::StoppableTask<RetT>::work_total)
        .def("work_completed", &mpcf::StoppableTask<RetT>::work_completed)
        .def("work_step", &mpcf::StoppableTask<RetT>::work_step)
        .def("work_step_desc", &mpcf::StoppableTask<RetT>::work_step_desc)
        .def("work_step_unit", &mpcf::StoppableTask<RetT>::work_step_unit)
    ;
  }

}

PYBIND11_MODULE(MPCF_MODULE_NAME, m) {
  mpcf_py::register_pcf(m);

  register_bindings_stoppable_task<void>(m, "_void");

  py::class_<GatedTask, mpcf::StoppableTask<void>>(m, "_GatedTask")
    .def(py::init<size_t>())
    .def("advance", &GatedTask::advance);

  m.def("_create_gated_task", [](size_t n_steps) {
    auto task = std::make_unique<GatedTask>(n_steps);
    task->start_async(mpcf::default_executor());
    return task;
  });

  py::enum_<std::future_status>(m, "FutureStatus")
    .value("deferred", std::future_status::deferred)
    .value("ready", std::future_status::ready)
    .value("timeout", std::future_status::timeout)
    .export_values();

  py::class_<mpcf_py::Future<void>>(m, "Future_void")
    .def(py::init<>())
    .def("wait_for", &mpcf_py::Future<void>::wait_for,
         py::call_guard<py::gil_scoped_release>());

  m.def("force_cpu", [](bool on){ mpcf::settings().forceCpu = on; });
  m.def("set_cuda_threshold", [](size_t n){ mpcf::settings().cudaThreshold = n; });
  m.def("set_parallel_eval_threshold", [](size_t n){ mpcf::settings().parallelEvalThreshold = n; });
  m.def("get_parallel_eval_threshold", [](){ return mpcf::settings().parallelEvalThreshold; });
  m.def("set_device_verbose", [](bool on){ mpcf::settings().deviceVerbose = on; });
  m.def("set_block_dim", [](unsigned int x, unsigned int y) {
    mpcf::settings().blockDimX = x;
    mpcf::settings().blockDimY = y;
  });
  m.def("set_min_block_side", [](size_t n){ mpcf::settings().minBlockSide = n; });
#ifdef BUILD_WITH_CUDA
  m.def("limit_gpus", [](size_t n){ mpcf::default_executor().limit_cuda_workers(n); });
#endif
  m.def("get_ngpus", &getNumGpus);

  m.def("limit_cpus", [](size_t n){ mpcf::default_executor().limit_cpu_workers(n); });

  m.def("_build_type", [] {
#ifdef BUILD_WITH_CUDA
    return std::string("CUDA");
#else
    return std::string("CPU");
#endif
  });

  mpcf_py::register_random(m);

  mpcf_py::register_io(m);

  mpcf_py::register_tensor_bindings(m);
  mpcf_py::register_np_conversions(m);

  mpcf_py::register_make_from_serial_content(m);

  mpcf_py::register_reductions(m);
  mpcf_py::register_distance(m);
  mpcf_py::register_inner_product(m);
  mpcf_py::register_norms(m);
  mpcf_py::register_symmetric_matrix(m);
  mpcf_py::register_distance_matrix(m);

  mpcf_py::register_module_persistence(m);
  mpcf_py::register_module_point_process(m);
}
