#ifndef STABLEBEAR_PY_TIMESERIES_BIND_H
#define STABLEBEAR_PY_TIMESERIES_BIND_H

#include "py_io.hpp"
#include "py_tensor.hpp"

#include <sbear/io.hpp>
#include <sbear/tensor.hpp>
#include <sbear/timeseries.hpp>


#include <optional>
#include <string>
#include <vector>

// The element types are bound two to a translation unit, as each one costs
// about half a gigabyte to compile; each unit instantiates the templates
// below.

namespace sb_py
{
  void register_timeseries_float(nb::module_& m);
  void register_timeseries_int(nb::module_& m);
  void register_timeseries_uint(nb::module_& m);
  void register_timeseries_pcf(nb::module_& m);
  void register_timeseries_pcf_int(nb::module_& m);
  void register_timeseries_point_cloud(nb::module_& m);
  void register_timeseries_distance_matrix(nb::module_& m);
  void register_timeseries_symmetric_matrix(nb::module_& m);
  void register_timeseries_barcode(nb::module_& m);

  /// Bind the channel and series classes for element type T, as
  /// register_tensor_element does for tensors.
  template <typename T>
  void register_timeseries_element(nb::module_& m, const std::string& suffix)
  {
    namespace nb = nanobind;
    using Channel = sb::Channel<T>;
    using Series = sb::TimeSeries<T>;

    nb::class_<Channel>(m, ("TimeSeriesChannel" + suffix).c_str())
      .def_prop_ro("times", &Channel::times)
      .def_prop_ro("shape", [](const Channel& self) {
        return nb::tuple(nb::cast(self.shape()));
      })
      .def_prop_ro("values", &Channel::values)
      .def_prop_ro("snap_tolerance", &Channel::snap_tolerance)
      .def("__len__", &Channel::size);

    nb::class_<Series>(m, ("TimeSeries" + suffix).c_str())
      .def(nb::init<sb::Times, const sb::Tensor<T>&, sb::float64_t>(),
           nb::arg("times"), nb::arg("values"), nb::arg("snap_tolerance"))
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
                  nb::arg("parts"), nb::arg("unit"))
      .def_prop_ro("unit", [](const Series& self) {
        return std::string(sb::time_unit_name(self.unit()));
      })
      .def("to", [](const Series& self, const std::string& unit) { return self.to(sb::parse_time_unit(unit)); },
           nb::arg("unit"))
      .def("times", &Series::times, nb::arg("index"))
      .def("origin", [](const Series& self, size_t index) {
        const auto& channel = self.channel(index);
        return sb::convert_time(channel.origin(), channel.unit(), self.unit());
      }, nb::arg("index"))
      .def("channel", &Series::channel, nb::arg("index"), nb::rv_policy::reference_internal)
      .def_prop_ro("n_channels", &Series::n_channels)
      .def_prop_ro("epoch", [](const Series& self) { return self.channel(0).epoch(); })
      .def("with_epoch", &Series::with_epoch, nb::arg("epoch"))
      .def_prop_ro("domain", [](const Series& self) {
        const auto domain = self.domain();
        return nb::make_tuple(domain.first, domain.last, domain.closedStart, domain.closedEnd);
      })
      .def("select", &Series::select, nb::arg("indices"))
      .def("array_equal", [](const Series& self, const Series& other) { return self.same_observations(other); },
           nb::arg("other"))
      .def("evaluate", [](const Series& self, sb::float64_t t, const std::optional<std::string>& unit) {
        return unit ? self.to(sb::parse_time_unit(*unit)).evaluate(t) : self.evaluate(t);
      }, nb::arg("t"), nb::arg("unit") = nb::none(), nb::call_guard<nb::gil_scoped_release>())
      .def("evaluate", [](const Series& self, const sb::Tensor<sb::float64_t>& t,
                          const std::optional<std::string>& unit) {
        return unit ? self.to(sb::parse_time_unit(*unit)).evaluate(t) : self.evaluate(t);
      }, nb::arg("t"), nb::arg("unit") = nb::none(), nb::call_guard<nb::gil_scoped_release>())
      .def("_save", [](const Series& self, nb::object file) {
        sb_py::PythonOStreamBuf buf(file);
        std::ostream os(&buf);
        sb::write_object(self, os);
      }, nb::arg("file"));
  }

  /// Bind series of element type T, and tensors and nested tensors of them,
  /// as windows give.
  template <typename T>
  void register_typed_timeseries(nb::module_& m, const std::string& suffix)
  {
    namespace nb = nanobind;
    register_timeseries_element<T>(m, suffix);
    register_tensor_element<sb::TimeSeries<T>>(m, "TimeSeries" + suffix);
    nb::borrow<nb::class_<sb::Tensor<sb::TimeSeries<T>>>>(m.attr(("TimeSeries" + suffix + "Tensor").c_str()))
      .def("_save", [](const sb::Tensor<sb::TimeSeries<T>>& self, nb::object file) {
        sb_py::PythonOStreamBuf buf(file);
        std::ostream os(&buf);
        sb::write(self, os);
      }, nb::arg("file"));
  }
}

#endif // STABLEBEAR_PY_TIMESERIES_BIND_H
