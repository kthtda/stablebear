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

#include "py_distance.hpp"

#include "tensor.hpp"
#include "functional/pcf.hpp"
#include "task.hpp"
#include "../py_async_support.hpp"
#include "algorithms/functional/apply_functional.hpp"

#include <memory>

namespace py = pybind11;

namespace
{

  template <typename Tt, typename Tv>
  class PyNormsBindings
  {
  public:
    using PcfT = sb::Pcf<Tt, Tv>;
    using TensorT = sb::Tensor<PcfT>;
    using OutTensorT = sb::Tensor<Tv>;

    static py::tuple lpnorm_l1(TensorT inTensor)
    {
      auto outTensor = OutTensorT(inTensor.shape(), Tv(0));

      auto f = [](const PcfT& v) {
        return sb::l1_norm(v);
      };

      std::unique_ptr<sb::StoppableTask<void>> task = sb_py::execute_stoppable_task<sb::ApplyFunctional<TensorT, OutTensorT, decltype(f)>>(inTensor, outTensor, f);
      return py::make_tuple(std::move(task), outTensor);
    }

    static py::tuple lpnorm_lp(TensorT inTensor, Tv p)
    {
      auto outTensor = OutTensorT(inTensor.shape(), Tv(0));

      auto f = [p](const PcfT& v) {
        return sb::lp_norm(v, p);
      };

      std::unique_ptr<sb::StoppableTask<void>> task = sb_py::execute_stoppable_task<sb::ApplyFunctional<TensorT, OutTensorT, decltype(f)>>(inTensor, outTensor, f);
      return py::make_tuple(std::move(task), outTensor);
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyNormsBindings> cls(m, ("Norms" + suffix).c_str());

      cls
          .def_static("lpnorm_l1", &PyNormsBindings::lpnorm_l1)
          .def_static("lpnorm_lp", &PyNormsBindings::lpnorm_lp)
      ;

    }
  };

}

namespace sb_py
{

  void register_norms(py::module_& m)
  {
    PyNormsBindings<sb::float32_t, sb::float32_t>::register_bindings(m, "_f32_f32");
    PyNormsBindings<sb::float64_t, sb::float64_t>::register_bindings(m, "_f64_f64");
  }

}