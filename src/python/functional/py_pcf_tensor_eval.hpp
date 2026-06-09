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

#ifndef STABLEBEAR_PY_PCF_TENSOR_EVAL_H
#define STABLEBEAR_PY_PCF_TENSOR_EVAL_H

#include "../pybind.hpp"
#include "../py_np_support.hpp"
#include <pybind11/numpy.h>

#include <sbear/tensor.hpp>
#include <sbear/algorithms/tensor_eval.hpp>

namespace py = pybind11;

namespace sb_py
{

  template <sb::IsTensor TA, sb::IsTensor TB>
  std::vector<size_t> eval_out_shape(const TA& a, const TB& b)
  {
    std::vector<size_t> shape;
    for (auto s : a.shape()) shape.push_back(s);
    for (auto s : b.shape()) shape.push_back(s);
    return shape;
  }

  // Scalar t -> numpy array of shape tensor_shape
  template <typename Tt, typename Tv, sb::IsTensor TPcfTensor>
  py::array_t<Tv> pcf_tensor_eval_scalar(const TPcfTensor& pcfs, Tt t)
  {
    const auto& sh = pcfs.shape();
    std::vector<py::ssize_t> out_shape(sh.begin(), sh.end());
    py::array_t<Tv> result(out_shape);
    NumpyTensor<Tv> out(result);
    sb::tensor_eval<Tt, Tv>(pcfs, t, out);
    return result;
  }


} // namespace sb_py

#endif // STABLEBEAR_PY_PCF_TENSOR_EVAL_H
