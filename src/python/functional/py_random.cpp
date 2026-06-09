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

#include "py_random.hpp"

#include <sbear/random.hpp>
#include <sbear/random_generator.hpp>

#include <pybind11/numpy.h>

#include "../py_np_support.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

namespace py = pybind11;

namespace
{

  template <typename Tt, typename Tv>
  class PyRandomBindings
  {
  public:
    using PcfT = sb::Pcf<Tt, Tv>;
    using TensorT = sb::Tensor<PcfT>;

    static void noisy_sin(TensorT& out, size_t nPoints, const sb::DefaultRandomGenerator* gen)
    {
      auto func = [](Tv t) { return sin(static_cast<Tv>(2. * M_PI) * t); };
      if (gen)
        sb::noisy_function(out, nPoints, func, static_cast<Tv>(0.1), *gen);
      else
        sb::noisy_function(out, nPoints, func);
    }

    static void noisy_cos(TensorT& out, size_t nPoints, const sb::DefaultRandomGenerator* gen)
    {
      auto func = [](Tv t) { return cos(static_cast<Tv>(2. * M_PI) * t); };
      if (gen)
        sb::noisy_function(out, nPoints, func, static_cast<Tv>(0.1), *gen);
      else
        sb::noisy_function(out, nPoints, func);
    }

    static void register_bindings(py::handle m, const std::string& suffix)
    {
      py::class_<PyRandomBindings> cls(m, ("Random" + suffix).c_str());

      cls
          .def_static("noisy_sin", &PyRandomBindings::noisy_sin,
                       py::arg("out"), py::arg("n_points"), py::arg("generator").none(true) = py::none())
          .def_static("noisy_cos", &PyRandomBindings::noisy_cos,
                       py::arg("out"), py::arg("n_points"), py::arg("generator").none(true) = py::none())
          ;
    }
  };

}

void sb_py::register_random(py::module_& m)
{
  py::class_<sb::DefaultRandomGenerator>(m, "RandomGenerator")
      .def(py::init<>())
      .def(py::init<uint64_t>())
      .def("seed", &sb::DefaultRandomGenerator::seed)
      ;

  m.def("seed", &sb::seed);

  PyRandomBindings<sb::float32_t, sb::float32_t>::register_bindings(m, "_f32_f32");
  PyRandomBindings<sb::float64_t, sb::float64_t>::register_bindings(m, "_f64_f64");
}
