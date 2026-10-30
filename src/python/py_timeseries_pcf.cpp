#include "py_timeseries_bind.hpp"

#include <sbear/functional/pcf.hpp>

namespace sb_py
{
  void register_timeseries_pcf(nb::module_& m)
  {
    register_typed_timeseries<sb::Pcf_f32>(m, "Pcf32");
    register_typed_timeseries<sb::Pcf_f64>(m, "Pcf64");
  }
}
