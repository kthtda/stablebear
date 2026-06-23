#include "py_symmetric_matrix.hpp"
#include "py_tensor.hpp"

namespace sb_py
{

  void register_symmetric_matrix(pybind11::module_& m)
  {
    register_symmetric_matrix_bindings<sb::float32_t>(m, "_f32");
    register_symmetric_matrix_bindings<sb::float64_t>(m, "_f64");

    register_typed_tensor_bindings<sb::SymmetricMatrix<sb::float32_t>>(m, "SymmetricMatrix32", "");
    register_typed_tensor_bindings<sb::SymmetricMatrix<sb::float64_t>>(m, "SymmetricMatrix64", "");
  }

}
