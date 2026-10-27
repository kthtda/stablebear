#ifndef STABLEBEAR_PY_TIMESERIES_BIND_H
#define STABLEBEAR_PY_TIMESERIES_BIND_H

#include "py_io.hpp"
#include "py_tensor.hpp"

#include <sbear/io.hpp>
#include <sbear/tensor.hpp>
#include <sbear/timeseries.hpp>

#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <vector>

// The element types are bound in several translation units so that they
// compile in parallel; each one instantiates the templates below.

namespace sb_py
{
  void register_timeseries_pcf(pybind11::module_& m);
  void register_timeseries_point_cloud(pybind11::module_& m);
  void register_timeseries_matrix_barcode(pybind11::module_& m);

  /// Bind the channel and series classes for element type T, as
  /// register_tensor_element does for tensors.
  template <typename T>
  void register_timeseries_element(pybind11::module_& m, const std::string& suffix)
  {
    namespace py = pybind11;
    using Channel = sb::Channel<T>;
    using Series = sb::TimeSeries<T>;

    py::class_<Channel>(m, ("TimeSeriesChannel" + suffix).c_str())
      .def_property_readonly("times", &Channel::times)
      .def_property_readonly("shape", [](const Channel& self) {
        return py::tuple(py::cast(self.shape()));
      })
      .def_property_readonly("values", &Channel::values)
      .def_property_readonly("snap_tolerance", &Channel::snap_tolerance)
      .def("__len__", &Channel::size);

    py::class_<Series>(m, ("TimeSeries" + suffix).c_str())
      .def(py::init<sb::Times, const sb::Tensor<T>&, sb::float64_t>(),
           py::arg("times"), py::arg("values"), py::arg("snap_tolerance"))
      .def_static("from_channels",
                  [](const std::vector<Series>& parts,
                     const std::optional<std::string>& unit) {
                    std::optional<sb::TimeUnit> parsed;
                    if (unit)
                    {
                      parsed = sb::parse_time_unit(*unit);
                    }
                    return Series::from_channels(parts, parsed);
                  },
                  py::arg("parts"), py::arg("unit"))
      .def_property_readonly("unit", [](const Series& self) {
        return std::string(sb::time_unit_name(self.unit()));
      })
      .def("to", [](const Series& self, const std::string& unit) { return self.to(sb::parse_time_unit(unit)); },
           py::arg("unit"))
      .def("times", &Series::times, py::arg("index"))
      .def("channel", &Series::channel, py::arg("index"), py::return_value_policy::reference_internal)
      .def_property_readonly("n_channels", &Series::n_channels)
      .def_property_readonly("domain", [](const Series& self) {
        const auto domain = self.domain();
        return py::make_tuple(domain.first, domain.last, domain.closedStart, domain.closedEnd);
      })
      .def("select", &Series::select, py::arg("indices"))
      .def("array_equal", [](const Series& self, const Series& other) { return self == other; }, py::arg("other"))
      .def("evaluate", [](const Series& self, sb::float64_t t, const std::optional<std::string>& unit) {
        return unit ? self.to(sb::parse_time_unit(*unit)).evaluate(t) : self.evaluate(t);
      }, py::arg("t"), py::arg("unit") = py::none(), py::call_guard<py::gil_scoped_release>())
      .def("evaluate", [](const Series& self, const sb::Tensor<sb::float64_t>& t,
                          const std::optional<std::string>& unit) {
        return unit ? self.to(sb::parse_time_unit(*unit)).evaluate(t) : self.evaluate(t);
      }, py::arg("t"), py::arg("unit") = py::none(), py::call_guard<py::gil_scoped_release>())
      .def("_save", [](const Series& self, py::object file) {
        sb_py::PythonOStreamBuf buf(file);
        std::ostream os(&buf);
        sb::write_object(self, os);
      }, py::arg("file"));
  }

  /// Bind series of element type T, and tensors and nested tensors of them,
  /// as windows give.
  template <typename T>
  void register_typed_timeseries(pybind11::module_& m, const std::string& suffix)
  {
    namespace py = pybind11;
    register_timeseries_element<T>(m, suffix);
    register_tensor_element<sb::TimeSeries<T>>(m, "TimeSeries" + suffix);
    py::reinterpret_borrow<py::class_<sb::Tensor<sb::TimeSeries<T>>>>(m.attr(("TimeSeries" + suffix + "Tensor").c_str()))
      .def("_save", [](const sb::Tensor<sb::TimeSeries<T>>& self, py::object file) {
        sb_py::PythonOStreamBuf buf(file);
        std::ostream os(&buf);
        sb::write(self, os);
      }, py::arg("file"));
  }
}

#endif // STABLEBEAR_PY_TIMESERIES_BIND_H
