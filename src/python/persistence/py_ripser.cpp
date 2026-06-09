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

#include "py_ripser.hpp"
#include "../py_async_support.hpp"

#include <sbear/tensor.hpp>
#include <sbear/distance_matrix.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/compute_persistence.hpp>

namespace py = pybind11;

namespace
{

  template <typename T>
  class PyRipserBindings
  {
  public:
    static std::unique_ptr<sb::StoppableTask<void>> spawn_ripser_pcloud_euclidean_task(const sb::Tensor<sb::PointCloud<T>>& pclouds, sb::Tensor<sb::ph::Barcode<T>>& out, size_t maxDim, bool reducedHomology)
    {
      return sb_py::execute_stoppable_task<sb::ph::RipserTask<T>>(pclouds, out, maxDim, reducedHomology);
    }

    static std::unique_ptr<sb::StoppableTask<void>> spawn_ripser_distmat_task(const sb::Tensor<sb::DistanceMatrix<T>>& dmats, sb::Tensor<sb::ph::Barcode<T>>& out, size_t maxDim, bool reducedHomology)
    {
      return sb_py::execute_stoppable_task<sb::ph::RipserDistMatTask<T>>(dmats, out, maxDim, reducedHomology);
    }

    static void register_bindings(py::module_& m, const std::string& suffix)
    {
      py::class_<PyRipserBindings>(m, ("PersistenceRipser" + suffix).c_str())
        .def_static("spawn_ripser_pcloud_euclidean_task", &PyRipserBindings::spawn_ripser_pcloud_euclidean_task)
        .def_static("spawn_ripser_distmat_task", &PyRipserBindings::spawn_ripser_distmat_task)
      ;
    }
  };

}

namespace sb_py
{
  void register_persistence_ripser(pybind11::module_ &m)
  {
    PyRipserBindings<sb::float32_t>::register_bindings(m, "32");
    PyRipserBindings<sb::float64_t>::register_bindings(m, "64");
  }
}
