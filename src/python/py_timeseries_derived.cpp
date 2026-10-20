#include "py_timeseries_derived.hpp"

#include <sbear/distance_matrix.hpp>
#include <sbear/functional/pcf.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/point_cloud.hpp>
#include <sbear/symmetric_matrix.hpp>
#include <sbear/timeseries.hpp>
#include <sbear/timeseries/time_delay_embedding.hpp>
#include <sbear/timeseries/windowing.hpp>

namespace py = pybind11;

namespace
{
  template <typename T>
  void register_derived_element(py::module_& m)
  {
    m.def("delay_embed", [](const sb::TimeSeries<T>& series, size_t dimension, sb::float64_t delay) {
      return sb::delay_embed(series, dimension, delay);
    }, py::arg("series"), py::arg("dimension"), py::arg("delay"));
    m.def("windows", [](const sb::TimeSeries<T>& series, sb::float64_t size, sb::float64_t step) {
      return sb::windows(series, size, step);
    }, py::arg("series"), py::arg("size"), py::arg("step"));
  }
}

namespace sb_py
{
  void register_timeseries_derived(py::module_& m)
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
