#ifndef STABLEBEAR_PY_TIMESERIES_BIND_H
#define STABLEBEAR_PY_TIMESERIES_BIND_H

#include "py_io.hpp"
#include "py_tensor.hpp"

#include <sbear/io.hpp>
#include <sbear/tensor.hpp>
#include <sbear/timeseries.hpp>

#include <nanobind/ndarray.h>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

// The element types are bound two to a translation unit, as each one costs
// about half a gigabyte to compile; each unit instantiates the templates
// below.

namespace sb_py
{
  /// The short name of @p unit, or none for times without a unit.
  inline std::optional<std::string> unit_name(sb::TimeUnit unit)
  {
    if (unit == sb::TimeUnit::Unitless)
    {
      return std::nullopt;
    }
    return std::string(sb::time_unit_name(unit));
  }

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
      .def_prop_ro("shape", [](const Channel& self) {
        return nb::tuple(nb::cast(self.shape()));
      })
      .def_prop_ro("values", &Channel::values)
      .def_prop_ro("snap_tolerance", &Channel::snap_tolerance)
      .def("__len__", &Channel::size);

    nb::class_<Series>(m, ("TimeSeries" + suffix).c_str())
      .def("__init__", [](Series* self, nb::ndarray<const sb::float64_t, nb::c_contig, nb::device::cpu> times,
                          const std::optional<std::string>& unit, const std::optional<sb::DateTime>& epoch,
                          const sb::Tensor<T>& values, sb::float64_t snapTolerance) {
        if (times.ndim() != 1)
        {
          throw std::invalid_argument(
              "Timestamps must be one-dimensional, got " + std::to_string(times.ndim()) + " dimensions");
        }
        std::vector<sb::float64_t> data(times.data(), times.data() + times.size());
        new (self) Series(sb::Times(std::move(data)), values, snapTolerance,
                          unit ? sb::parse_time_unit(*unit) : sb::TimeUnit::Unitless, epoch);
      }, nb::arg("times"), nb::arg("unit"), nb::arg("epoch"), nb::arg("values"), nb::arg("snap_tolerance"))
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
      .def_prop_ro("unit", [](const Series& self) { return unit_name(self.unit()); })
      .def("to", [](const Series& self, const std::string& unit) { return self.to(sb::parse_time_unit(unit)); },
           nb::arg("unit"))
      .def("times", [](const Series& self, size_t index) {
        // A copy: timestamps cannot change after validation.
        const auto times = self.channel(index).times();
        sb::Tensor<sb::float64_t> out({times.size()});
        std::copy(times.values().begin(), times.values().end(), out.data());
        return out;
      }, nb::arg("index"))
      .def("origin", [](const Series& self, size_t index) { return self.channel(index).origin(); }, nb::arg("index"))
      .def("channel", &Series::channel, nb::arg("index"), nb::rv_policy::reference_internal)
      .def_prop_ro("n_channels", &Series::n_channels)
      .def_prop_ro("epoch", &Series::epoch)
      .def("with_epoch", &Series::with_epoch, nb::arg("epoch"))
      .def("shifted", &Series::shifted, nb::arg("delta"), nb::arg("epoch"))
      .def_prop_ro("domain", [](const Series& self) {
        const auto domain = self.domain();
        return nb::make_tuple(domain.first, domain.last, domain.closedStart, domain.closedEnd);
      })
      .def("select", &Series::select, nb::arg("indices"))
      .def("array_equal", [](const Series& self, const Series& other) { return self.same_observations(other); },
           nb::arg("other"))
      .def("evaluate", [](const Series& self, sb::float64_t t) { return self.evaluate(t); },
           nb::arg("t"), nb::call_guard<nb::gil_scoped_release>())
      .def("evaluate", [](const Series& self, const sb::Tensor<sb::float64_t>& t) { return self.evaluate(t); },
           nb::arg("t"), nb::call_guard<nb::gil_scoped_release>())
      .def("_save", &save_object<Series>, nb::arg("file"));
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
      .def("_save", &save_tensor<sb::Tensor<sb::TimeSeries<T>>>, nb::arg("file"));
  }
}

#endif // STABLEBEAR_PY_TIMESERIES_BIND_H
