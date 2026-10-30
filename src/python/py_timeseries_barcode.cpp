#include "py_timeseries_bind.hpp"

#include <sbear/persistence/barcode.hpp>

namespace sb_py
{
  void register_timeseries_barcode(nb::module_& m)
  {
    register_typed_timeseries<sb::ph::Barcode<sb::float32_t>>(m, "Barcode32");
    register_typed_timeseries<sb::ph::Barcode<sb::float64_t>>(m, "Barcode64");
  }
}
