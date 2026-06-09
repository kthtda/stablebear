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

#include "py_distance_matrix.hpp"
#include "py_tensor.hpp"

namespace sb_py
{

  void register_distance_matrix(pybind11::module_& m)
  {
    register_distance_matrix_bindings<sb::float32_t>(m, "_f32");
    register_distance_matrix_bindings<sb::float64_t>(m, "_f64");

    register_typed_tensor_bindings<sb::DistanceMatrix<sb::float32_t>>(m, "DistanceMatrix32", "");
    register_typed_tensor_bindings<sb::DistanceMatrix<sb::float64_t>>(m, "DistanceMatrix64", "");
  }

}
