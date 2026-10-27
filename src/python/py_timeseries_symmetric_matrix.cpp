#include "py_timeseries_bind.hpp"

#include <sbear/symmetric_matrix.hpp>

namespace sb_py
{
  void register_timeseries_symmetric_matrix(pybind11::module_& m)
  {
    register_typed_timeseries<sb::SymmetricMatrix<sb::float32_t>>(m, "SymmetricMatrix32");
    register_typed_timeseries<sb::SymmetricMatrix<sb::float64_t>>(m, "SymmetricMatrix64");
  }
}
