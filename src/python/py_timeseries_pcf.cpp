#include "py_timeseries_bind.hpp"

#include <sbear/functional/pcf.hpp>

namespace sb_py
{
  void register_timeseries_pcf(pybind11::module_& m)
  {
    register_typed_timeseries<sb::Pcf_f32>(m, "Pcf32");
    register_typed_timeseries<sb::Pcf_f64>(m, "Pcf64");
    register_typed_timeseries<sb::Pcf_i32>(m, "Pcf32i");
    register_typed_timeseries<sb::Pcf_i64>(m, "Pcf64i");
  }
}
