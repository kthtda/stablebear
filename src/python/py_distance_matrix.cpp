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
