#include "py_timeseries_bind.hpp"

namespace sb_py
{
  void register_timeseries_int(nb::module_& m)
  {
    register_typed_timeseries<sb::int32_t>(m, "Int32");
    register_typed_timeseries<sb::int64_t>(m, "Int64");
  }
}
