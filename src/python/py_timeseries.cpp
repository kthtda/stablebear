#include "py_timeseries.hpp"
#include "py_timeseries_bind.hpp"

#include <nanobind/ndarray.h>

#include <algorithm>
#include <optional>
#include <string>

namespace nb = nanobind;

namespace sb_py
{
  void register_timeseries(nb::module_& m)
  {
    nb::class_<sb::DateTime>(m, "DateTime")
      .def("__init__", [](sb::DateTime* self, int64_t count, const std::string& unit) {
        new (self) sb::DateTime(count, sb::parse_time_unit(unit));
      }, nb::arg("count"), nb::arg("unit"))
      .def_ro("count", &sb::DateTime::count)
      .def_prop_ro("unit", [](const sb::DateTime& self) {
        return std::string(sb::time_unit_name(self.unit));
      })
      .def("__eq__", [](const sb::DateTime& self, const sb::DateTime& other) { return self == other; });

    nb::class_<sb::Times>(m, "Times")
      .def("__init__", [](sb::Times* self, nb::ndarray<const sb::float64_t, nb::c_contig, nb::device::cpu> values,
                          const std::string& unit, const std::optional<sb::DateTime>& epoch) {
        if (values.ndim() != 1)
        {
          throw std::invalid_argument(
              "Timestamps must be one-dimensional, got " + std::to_string(values.ndim()) + " dimensions");
        }
        std::vector<sb::float64_t> data(values.data(), values.data() + values.size());
        new (self) sb::Times(std::move(data), sb::parse_time_unit(unit), epoch);
      }, nb::arg("values"), nb::arg("unit"), nb::arg("epoch") = nb::none())
      .def_prop_ro("values", [](const sb::Times& self) {
        // A copy: timestamps cannot change after validation.
        sb::Tensor<sb::float64_t> out({self.size()});
        std::copy(self.values().begin(), self.values().end(), out.data());
        return out;
      })
      .def_prop_ro("unit", [](const sb::Times& self) {
        return std::string(sb::time_unit_name(self.unit()));
      })
      .def_prop_ro("epoch", &sb::Times::epoch)
      .def_prop_ro("origin", &sb::Times::origin)
      .def("with_epoch", &sb::Times::with_epoch, nb::arg("epoch"))
      .def("to", [](const sb::Times& self, const std::string& unit) {
        return self.to(sb::parse_time_unit(unit));
      }, nb::arg("unit"))
      .def("__len__", &sb::Times::size)
      .def("__eq__", [](const sb::Times& self, const sb::Times& other) { return self == other; });

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
