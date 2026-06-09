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

//
// Created by bjorn on 2/25/2026.
//

#include "py_np_tensor_convert.hpp"

#include <sbear/tensor.hpp>

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace
{
  template <typename NumpyValueT, typename TensorValueT>
  sb::Tensor<TensorValueT> convert_numpy_to_tensor(py::array_t<NumpyValueT> arr)
  {
    auto const * arr_data = arr.template unchecked<>().data();
    auto const * arr_strides = arr.strides();
    auto arr_itemsize = arr.itemsize();

    sb::Tensor<TensorValueT> t(std::vector<size_t>(arr.shape(), arr.shape() + arr.ndim()));
    sb::walk(t, [&t, arr_data, arr_strides, arr_itemsize](const std::vector<size_t>& idx) {

      auto arr_idx = std::inner_product(idx.begin(), idx.end(), arr_strides, 0_z);
      arr_idx /= arr_itemsize;

      t(idx) = *(arr_data + arr_idx);
    });

    return t;
  }

  template <typename NumpyValueT, typename TensorValueT>
  void register_np_conversion_function(py::module_& m, const std::string& suffix)
  {
    m.def(("ndarray_to_tensor_" + suffix).c_str(), &convert_numpy_to_tensor<NumpyValueT, TensorValueT>);
  }
}

namespace sb_py
{

  void register_np_conversions(py::module_& m)
  {
    register_np_conversion_function<sb::float32_t, sb::float32_t>(m, "32");
    register_np_conversion_function<sb::float64_t, sb::float64_t>(m, "64");
    register_np_conversion_function<sb::int32_t, sb::int32_t>(m, "i32");
    register_np_conversion_function<sb::int64_t, sb::int64_t>(m, "i64");
    register_np_conversion_function<uint32_t, uint32_t>(m, "u32");
    register_np_conversion_function<uint64_t, uint64_t>(m, "u64");
    m.def("ndarray_to_bool_tensor", &convert_numpy_to_tensor<bool, bool>);
  }

}
