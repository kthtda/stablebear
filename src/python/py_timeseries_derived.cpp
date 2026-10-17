#include "py_timeseries_derived.hpp"

#include <sbear/timeseries.hpp>
#include <sbear/timeseries/derived.hpp>

namespace py = pybind11;

namespace
{
  template <typename T>
  void register_derived_element(py::module_& m)
  {
    m.def("delay_embed", [](const sb::TimeSeries<T>& series, size_t dimension, sb::float64_t delay) {
      return sb::delay_embed(series, dimension, delay);
    }, py::arg("series"), py::arg("dimension"), py::arg("delay"));
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
  }
}
