#include "py_tensor.hpp"

#include <sbear/functional/plf.hpp>

namespace sb_py
{
  void register_tensor_plf_bindings(nb::module_& m)
  {
    register_typed_tensor_bindings<sb::Plf_f32>(m, "Plf32", "");
    register_typed_tensor_bindings<sb::Plf_f64>(m, "Plf64", "");
  }
}
