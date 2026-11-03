#include "py_timeseries.hpp"
#include "py_timeseries_bind.hpp"

#include <string>

namespace nb = nanobind;

namespace sb_py
{
  void register_timeseries(nb::module_& m)
  {
    m.def("time_unit_name", [](const std::string& unit) {
      return std::string(sb::time_unit_name(sb::parse_time_unit(unit)));
    }, nb::arg("unit"), "The short name of a time unit given by its name or an alias.");

    nb::class_<sb::DateTime>(m, "DateTime")
      .def("__init__", [](sb::DateTime* self, int64_t count, const std::string& unit) {
        new (self) sb::DateTime(count, sb::parse_time_unit(unit));
      }, nb::arg("count"), nb::arg("unit"))
      .def_ro("count", &sb::DateTime::count)
      .def_prop_ro("unit", [](const sb::DateTime& self) {
        return std::string(sb::time_unit_name(self.unit));
      })
      .def("__eq__", [](const sb::DateTime& self, const sb::DateTime& other) { return self == other; });

    register_typed_timeseries<bool>(m, "Bool");
    register_timeseries_float(m);
    register_timeseries_int(m);
    register_timeseries_uint(m);
    register_timeseries_pcf(m);
    register_timeseries_pcf_int(m);
    register_timeseries_point_cloud(m);
    register_timeseries_distance_matrix(m);
    register_timeseries_symmetric_matrix(m);
    register_timeseries_barcode(m);
  }
}
