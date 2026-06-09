// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "py_inner_product.hpp"

#include "tensor.hpp"
#include "functional/pcf.hpp"
#include "task.hpp"
#include "functional/operations.cuh"
#include "algorithms/functional/matrix_integrate.hpp"
#include "../py_async_support.hpp"
#include <sbear/settings.hpp>

#ifdef BUILD_WITH_CUDA
#include <sbear/cuda/cuda_matrix_integrate_api.hpp>
#endif

#include <sbear/symmetric_matrix.hpp>

#include <memory>

namespace py = pybind11;

namespace
{

  template <typename Tt, typename Tv>
  class PyInnerProductBindings
  {
  public:
    using PcfT = sb::Pcf<Tt, Tv>;
    using TensorT = sb::Tensor<PcfT>;

    static py::tuple l2(TensorT fs)
    {
      auto op = sb::OperationL2InnerProduct<Tt, Tv>();
      auto n = static_cast<size_t>(fs.shape(0));

      auto symmat = sb::SymmetricMatrix<Tv>(n);

      if (n == 0)
      {
        std::unique_ptr<sb::StoppableTask<void>> empty_task = sb_py::execute_empty_task();
        return py::make_tuple(std::move(empty_task), symmat);
      }

      auto begin = sb::begin1dValues(fs);
      auto end = sb::end1dValues(fs);

#ifdef BUILD_WITH_CUDA
      if (!sb::settings().forceCpu && static_cast<size_t>(std::distance(begin, end)) >= sb::settings().cudaThreshold)
      {
        if (sb::settings().deviceVerbose)
        {
          std::cout << "Kernel computation on CUDA device(s)" << std::endl;
        }

        std::vector<PcfT> pcfs(begin, end);
        auto task = sb::create_cuda_block_integrate_l2_kernel_task(symmat, pcfs, Tv(0), std::numeric_limits<Tv>::max());
        task->start_async(sb::default_executor());
        return py::make_tuple(std::move(task), symmat);
      }
#endif

      if (sb::settings().deviceVerbose)
      {
        std::cout << "Kernel computation on CPU(s)" << std::endl;
      }

      std::unique_ptr<sb::StoppableTask<void>> task = sb_py::execute_stoppable_task<sb::CpuPairwiseIntegrationTask<decltype(op), decltype(begin), sb::SymmetricMatrix<Tv>, true>>(symmat, begin, end, op);
      return py::make_tuple(std::move(task), symmat);
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyInnerProductBindings> cls(m, ("InnerProduct" + suffix).c_str());

      cls
          .def_static("l2", &PyInnerProductBindings::l2)
      ;
    }
  };

}

namespace sb_py
{

  void register_inner_product(py::module_& m)
  {
    PyInnerProductBindings<sb::float32_t, sb::float32_t>::register_bindings(m, "_f32_f32");
    PyInnerProductBindings<sb::float64_t, sb::float64_t>::register_bindings(m, "_f64_f64");
  }

}
