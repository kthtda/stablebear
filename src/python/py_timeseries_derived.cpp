#include "py_timeseries_derived.hpp"

#include <sbear/distance_matrix.hpp>
#include <sbear/functional/pcf.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/point_cloud.hpp>
#include <sbear/symmetric_matrix.hpp>
#include <sbear/timeseries.hpp>
#include <sbear/timeseries/time_delay_embedding.hpp>
#include <sbear/timeseries/windowing.hpp>


#include <optional>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace
{
  template <typename T>
  void register_derived_element(nb::module_& m)
  {
    m.def("evaluate_series", [](const sb::Tensor<sb::TimeSeries<T>>& series, const sb::Tensor<sb::float64_t>& t,
                                const std::optional<std::string>& unit) {
      if (!unit)
      {
        return sb::evaluate(series, t);
      }
      // Each series reads the queries in @p unit; the copies share their
      // observations with @p series.
      const auto target = sb::parse_time_unit(*unit);
      sb::Tensor<sb::TimeSeries<T>> converted(std::vector<size_t>(series.shape().begin(), series.shape().end()));
      sb::walk(series, [&](const std::vector<size_t>& index) { converted(index) = series(index).to(target); });
      return sb::evaluate(converted, t);
    }, nb::arg("series"), nb::arg("t"), nb::arg("unit") = nb::none(), nb::call_guard<nb::gil_scoped_release>());
    m.def("delay_embed", [](const sb::TimeSeries<T>& series, size_t dimension, sb::float64_t delay, bool forward) {
      return sb::delay_embed(series, dimension, delay, forward);
    }, nb::arg("series"), nb::arg("dimension"), nb::arg("delay"), nb::arg("forward"),
       nb::call_guard<nb::gil_scoped_release>());
    m.def("windows", [](const sb::TimeSeries<T>& series, sb::float64_t size, sb::float64_t step, bool closedStart,
                        bool closedEnd, bool partial, bool anchorEnd) {
      return sb::windows(series, size, step, closedStart, closedEnd, partial, anchorEnd);
    }, nb::arg("series"), nb::arg("size"), nb::arg("step"), nb::arg("closed_start"), nb::arg("closed_end"),
       nb::arg("partial"), nb::arg("anchor_end"), nb::call_guard<nb::gil_scoped_release>());
  }
}

namespace sb_py
{
  void register_timeseries_derived(nb::module_& m)
  {
    register_derived_element<bool>(m);
    register_derived_element<sb::float32_t>(m);
    register_derived_element<sb::float64_t>(m);
    register_derived_element<sb::int32_t>(m);
    register_derived_element<sb::int64_t>(m);
    register_derived_element<uint32_t>(m);
    register_derived_element<uint64_t>(m);
    register_derived_element<sb::Pcf_f32>(m);
    register_derived_element<sb::Pcf_f64>(m);
    register_derived_element<sb::Pcf_i32>(m);
    register_derived_element<sb::Pcf_i64>(m);
    register_derived_element<sb::PointCloud<sb::float32_t>>(m);
    register_derived_element<sb::PointCloud<sb::float64_t>>(m);
    register_derived_element<sb::DistanceMatrix<sb::float32_t>>(m);
    register_derived_element<sb::DistanceMatrix<sb::float64_t>>(m);
    register_derived_element<sb::SymmetricMatrix<sb::float32_t>>(m);
    register_derived_element<sb::SymmetricMatrix<sb::float64_t>>(m);
    register_derived_element<sb::ph::Barcode<sb::float32_t>>(m);
    register_derived_element<sb::ph::Barcode<sb::float64_t>>(m);
  }
}
