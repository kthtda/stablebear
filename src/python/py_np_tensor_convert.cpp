//
// Created by bjorn on 2/25/2026.
//

#include "py_np_tensor_convert.hpp"

#include <sbear/tensor.hpp>

#include "py_np_support.hpp"

namespace nb = nanobind;

namespace
{
  template <typename NumpyValueT, typename TensorValueT>
  sb::Tensor<TensorValueT> convert_numpy_to_tensor(sb_py::NpIn<NumpyValueT> arr)
  {
    NumpyTensor<const NumpyValueT> in(arr);

    sb::Tensor<TensorValueT> t(in.shape());
    sb::walk(t, [&t, &in](const std::vector<size_t>& idx) {
      t(idx) = in(idx);
    });

    return t;
  }

  template <typename NumpyValueT, typename TensorValueT>
  void register_np_conversion_function(nb::module_& m, const std::string& suffix)
  {
    m.def(("ndarray_to_tensor_" + suffix).c_str(), &convert_numpy_to_tensor<NumpyValueT, TensorValueT>);
  }
}

namespace sb_py
{

  void register_np_conversions(nb::module_& m)
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
