#include "py_timeseries_derived.hpp"

#include <sbear/distance_matrix.hpp>
#include <sbear/functional/pcf.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/point_cloud.hpp>
#include <sbear/symmetric_matrix.hpp>
#include <sbear/timeseries.hpp>
#include <sbear/timeseries/point_clouds.hpp>
#include <sbear/timeseries/stack_channels.hpp>
#include <sbear/timeseries/time_delay_embedding.hpp>
#include <sbear/timeseries/windowing.hpp>


#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace nb = nanobind;

namespace
{
  /// The times to take points at, from the arguments of to_point_clouds:
  /// n, or step or times in unit (none for plain numbers), or dates in
  /// unit, or none of them.
  sb::PointTimes point_times(size_t n, sb::float64_t step, std::optional<std::vector<sb::float64_t>> times,
                             std::optional<std::vector<int64_t>> dates, const std::optional<std::string>& unit)
  {
    const auto timeUnit = unit ? sb::parse_time_unit(*unit) : sb::TimeUnit::Unitless;
    if (dates)
    {
      return {.n = n, .step = step, .times = std::nullopt, .unit = sb::TimeUnit::Unitless, .dates = std::move(dates),
              .dateUnit = timeUnit};
    }
    return {.n = n, .step = step, .times = std::move(times), .unit = timeUnit, .dates = std::nullopt,
            .dateUnit = sb::TimeUnit::Unitless};
  }

  template <typename T>
  void register_derived_element(nb::module_& m)
  {
    m.def("evaluate_series", [](const sb::Tensor<sb::TimeSeries<T>>& series, const sb::Tensor<sb::float64_t>& t,
                                const std::optional<std::string>& unit) {
      return sb::evaluate(series, t, unit ? std::optional(sb::parse_time_unit(*unit)) : std::nullopt);
    }, nb::arg("series"), nb::arg("t"), nb::arg("unit") = nb::none(), nb::call_guard<nb::gil_scoped_release>());
    m.def("evaluate_series_dates", [](const sb::Tensor<sb::TimeSeries<T>>& series, const sb::Tensor<int64_t>& dates,
                                      const std::string& unit) {
      return sb::evaluate_dates(series, dates, sb::parse_time_unit(unit));
    }, nb::arg("series"), nb::arg("dates"), nb::arg("unit"), nb::call_guard<nb::gil_scoped_release>());
    m.def("delay_embed", [](const sb::TimeSeries<T>& series, size_t dimension, sb::float64_t delay, bool forward) {
      return sb::delay_embed(series, dimension, delay, forward);
    }, nb::arg("series"), nb::arg("dimension"), nb::arg("delay"), nb::arg("forward"),
       nb::call_guard<nb::gil_scoped_release>());
    m.def("stack_channels", [](const sb::TimeSeries<T>& series, std::ptrdiff_t axis) {
      return sb::stack_channels(series, axis);
    }, nb::arg("series"), nb::arg("axis"), nb::call_guard<nb::gil_scoped_release>());
    m.def("concatenate_channels", [](const sb::TimeSeries<T>& series, std::ptrdiff_t axis) {
      return sb::concatenate_channels(series, axis);
    }, nb::arg("series"), nb::arg("axis"), nb::call_guard<nb::gil_scoped_release>());
    m.def("restrict", [](const sb::TimeSeries<T>& series, sb::float64_t from, sb::float64_t to, bool startClosed,
                         bool endClosed) {
      return sb::restrict(series, from, to, startClosed, endClosed);
    }, nb::arg("series"), nb::arg("start"), nb::arg("end"), nb::arg("start_closed"), nb::arg("end_closed"),
       nb::call_guard<nb::gil_scoped_release>());
    m.def("windows", [](const sb::TimeSeries<T>& series, sb::float64_t size, sb::float64_t step, bool closedStart,
                        bool closedEnd, bool partial, bool anchorEnd) {
      return sb::windows(series, size, step, closedStart, closedEnd, partial, anchorEnd);
    }, nb::arg("series"), nb::arg("size"), nb::arg("step"), nb::arg("closed_start"), nb::arg("closed_end"),
       nb::arg("partial"), nb::arg("anchor_end"), nb::call_guard<nb::gil_scoped_release>());
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
    {
      m.def("to_point_cloud", [](const sb::TimeSeries<T>& series, size_t n, sb::float64_t step,
                                 std::optional<std::vector<sb::float64_t>> times,
                                 std::optional<std::vector<int64_t>> dates, const std::optional<std::string>& unit) {
        return sb::to_point_cloud(series, point_times(n, step, std::move(times), std::move(dates), unit));
      }, nb::arg("series"), nb::arg("n") = 0, nb::arg("step") = 0, nb::arg("times") = nb::none(),
         nb::arg("dates") = nb::none(), nb::arg("unit") = nb::none(), nb::call_guard<nb::gil_scoped_release>());
      m.def("to_point_clouds", [](const sb::Tensor<sb::TimeSeries<T>>& seriesTensor, size_t n, sb::float64_t step,
                                  std::optional<std::vector<sb::float64_t>> times,
                                  std::optional<std::vector<int64_t>> dates, const std::optional<std::string>& unit) {
        return sb::to_point_clouds(seriesTensor, point_times(n, step, std::move(times), std::move(dates), unit));
      }, nb::arg("series_tensor"), nb::arg("n") = 0, nb::arg("step") = 0, nb::arg("times") = nb::none(),
         nb::arg("dates") = nb::none(), nb::arg("unit") = nb::none(), nb::call_guard<nb::gil_scoped_release>());
    }
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
