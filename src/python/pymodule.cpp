#include "pybind.hpp"

#include <sbear/executor.hpp>
#include <sbear/task.hpp>

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
#include <sbear/cuda/cuda_matrix_integrate_api.hpp>
#endif

#include <sbear/settings.hpp>

namespace py = pybind11;

namespace
{

  // A test-only task that blocks until Python calls advance() n_steps times.
  // Used to verify the GIL is released during wait_for: if it isn't, the
  // thread calling wait_for holds the GIL, Python can never call advance(),
  // and the test deadlocks.
  class GatedTask : public sb::StoppableTask<void>
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
    tf::Future<void> run_async(sb::Executor& exec) override
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
    throw std::runtime_error("This version of stablebear is compiled without GPU support.");
#endif
  }

  template <typename RetT>
  static void register_bindings_stoppable_task(py::handle m, const std::string& suffix)
  {
    py::class_<sb::StoppableTask<RetT>> cls(m, ("StoppableTask" + suffix).c_str());

    cls
        .def("request_stop", &sb::StoppableTask<RetT>::request_stop)
        .def("wait_for", [](sb::StoppableTask<RetT>& self, int ms) { return self.wait_for(std::chrono::milliseconds(ms)); },
             py::call_guard<py::gil_scoped_release>())
        .def("work_total", &sb::StoppableTask<RetT>::work_total)
        .def("work_completed", &sb::StoppableTask<RetT>::work_completed)
        .def("work_step", &sb::StoppableTask<RetT>::work_step)
        .def("work_step_desc", &sb::StoppableTask<RetT>::work_step_desc)
        .def("work_step_unit", &sb::StoppableTask<RetT>::work_step_unit)
    ;
  }

}

PYBIND11_MODULE(SB_MODULE_NAME, m) {
  sb_py::register_pcf(m);

  register_bindings_stoppable_task<void>(m, "_void");

  py::class_<GatedTask, sb::StoppableTask<void>>(m, "_GatedTask")
    .def(py::init<size_t>())
    .def("advance", &GatedTask::advance);

  m.def("_create_gated_task", [](size_t n_steps) {
    auto task = std::make_unique<GatedTask>(n_steps);
    task->start_async(sb::default_executor());
    return task;
  });

  py::enum_<std::future_status>(m, "FutureStatus")
    .value("deferred", std::future_status::deferred)
    .value("ready", std::future_status::ready)
    .value("timeout", std::future_status::timeout)
    .export_values();

  py::class_<sb_py::Future<void>>(m, "Future_void")
    .def(py::init<>())
    .def("wait_for", &sb_py::Future<void>::wait_for,
         py::call_guard<py::gil_scoped_release>());

  m.def("force_cpu", [](bool on){ sb::settings().forceCpu = on; });
  m.def("set_cuda_threshold", [](size_t n){ sb::settings().cudaThreshold = n; });
  m.def("set_parallel_eval_threshold", [](size_t n){ sb::settings().parallelEvalThreshold = n; });
  m.def("get_parallel_eval_threshold", [](){ return sb::settings().parallelEvalThreshold; });
  m.def("set_device_verbose", [](bool on){ sb::settings().deviceVerbose = on; });
  m.def("set_block_dim", [](unsigned int x, unsigned int y) {
    sb::settings().blockDimX = x;
    sb::settings().blockDimY = y;
  });
  m.def("set_min_block_side", [](size_t n){ sb::settings().minBlockSide = n; });
#ifdef BUILD_WITH_CUDA
  m.def("limit_gpus", [](size_t n){ sb::default_executor().limit_cuda_workers(n); });
#endif
  m.def("get_ngpus", &getNumGpus);

  m.def("limit_cpus", [](size_t n){ sb::default_executor().limit_cpu_workers(n); });

  m.def("_build_type", [] {
#ifdef BUILD_WITH_CUDA
    return std::string("CUDA");
#else
    return std::string("CPU");
#endif
  });

  sb_py::register_random(m);

  sb_py::register_io(m);

  sb_py::register_tensor_bindings(m);
  sb_py::register_np_conversions(m);

  sb_py::register_make_from_serial_content(m);

  sb_py::register_reductions(m);
  sb_py::register_distance(m);
  sb_py::register_inner_product(m);
  sb_py::register_norms(m);
  sb_py::register_symmetric_matrix(m);
  sb_py::register_distance_matrix(m);

  sb_py::register_module_persistence(m);
  sb_py::register_module_point_process(m);
}
