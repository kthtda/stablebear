#include "py_timeseries_bind.hpp"

namespace sb_py
{
  void register_timeseries_float(pybind11::module_& m)
  {
    register_typed_timeseries<sb::float32_t>(m, "Float32");
    register_typed_timeseries<sb::float64_t>(m, "Float64");
  }
}
