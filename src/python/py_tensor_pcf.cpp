#include "py_tensor.hpp"

namespace sb_py
{
  void register_tensor_pcf_bindings(nb::module_& m)
  {
    register_tensor_element<sb::Pcf_f32>(m, "Pcf32");
    register_tensor_element<sb::Pcf_f64>(m, "Pcf64");
    register_tensor_element<sb::Pcf_i32>(m, "Pcf32i");
    register_tensor_element<sb::Pcf_i64>(m, "Pcf64i");
  }
}
