#include "py_timeseries.hpp"
#include "py_timeseries_bind.hpp"

#include <pybind11/numpy.h>

#include <algorithm>
#include <optional>
#include <string>

namespace py = pybind11;

namespace sb_py
{
  void register_timeseries(py::module_& m)
  {
    py::class_<sb::DateTime>(m, "DateTime")
      .def(py::init([](int64_t count, const std::string& unit) { return sb::DateTime(count, sb::parse_time_unit(unit)); }),
           py::arg("count"), py::arg("unit"))
      .def_readonly("count", &sb::DateTime::count)
      .def_property_readonly("unit", [](const sb::DateTime& self) {
        return std::string(sb::time_unit_name(self.unit));
      })
      .def("__eq__", [](const sb::DateTime& self, const sb::DateTime& other) { return self == other; });

    py::class_<sb::Times>(m, "Times")
      .def(py::init([](py::array_t<sb::float64_t, py::array::c_style | py::array::forcecast> values,
                       const std::string& unit, const std::optional<sb::DateTime>& epoch) {
        if (values.ndim() != 1)
        {
          throw py::value_error(
              "Timestamps must be one-dimensional, got " + std::to_string(values.ndim()) + " dimensions");
        }
        std::vector<sb::float64_t> data(values.data(), values.data() + values.size());
        return sb::Times(std::move(data), sb::parse_time_unit(unit), epoch);
      }), py::arg("values"), py::arg("unit"), py::arg("epoch") = py::none())
      .def_property_readonly("values", [](const sb::Times& self) {
        // A copy: timestamps cannot change after validation.
        sb::Tensor<sb::float64_t> out({self.size()});
        std::copy(self.values().begin(), self.values().end(), out.data());
        return out;
      })
      .def_property_readonly("unit", [](const sb::Times& self) {
        return std::string(sb::time_unit_name(self.unit()));
      })
      .def_property_readonly("epoch", &sb::Times::epoch)
      .def("with_epoch", &sb::Times::with_epoch, py::arg("epoch"))
      .def("to", [](const sb::Times& self, const std::string& unit) {
        return self.to(sb::parse_time_unit(unit));
      }, py::arg("unit"))
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
