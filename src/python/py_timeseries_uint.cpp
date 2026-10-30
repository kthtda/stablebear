#include "py_timeseries_bind.hpp"

namespace sb_py
{
  void register_timeseries_uint(nb::module_& m)
  {
    register_typed_timeseries<uint32_t>(m, "Uint32");
    register_typed_timeseries<uint64_t>(m, "Uint64");
  }
}
