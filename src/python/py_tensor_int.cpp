#include "py_tensor.hpp"

namespace sb_py
{
  void register_tensor_int_bindings(pybind11::module_& m)
  {
    register_tensor_element<sb::int32_t>(m, "Int32");
    register_tensor_element<sb::int64_t>(m, "Int64");
    register_tensor_element<uint32_t>(m, "Uint32");
    register_tensor_element<uint64_t>(m, "Uint64");
  }
}
