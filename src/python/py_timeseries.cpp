#include "py_timeseries.hpp"

#include <sbear/tensor.hpp>
#include <sbear/timeseries.hpp>

#include <pybind11/numpy.h>

#include <algorithm>
#include <string>
#include <vector>

namespace py = pybind11;

namespace sb_py
{
  void register_timeseries(py::module_& m)
  {
    py::class_<sb::Times>(m, "Times")
      .def(py::init([](py::array_t<sb::float64_t, py::array::c_style | py::array::forcecast> values,
                       const std::string& unit) {
        if (values.ndim() != 1)
        {
          throw py::value_error(
              "Timestamps must be one-dimensional, got " + std::to_string(values.ndim()) + " dimensions");
        }
        std::vector<sb::float64_t> data(values.data(), values.data() + values.size());
        return sb::Times(std::move(data), sb::parse_time_unit(unit));
      }), py::arg("values"), py::arg("unit"))
      .def_property_readonly("values", [](const sb::Times& self) {
        // A copy: timestamps cannot change after validation.
        sb::Tensor<sb::float64_t> out({self.size()});
        std::copy(self.values().begin(), self.values().end(), out.data());
        return out;
      })
      .def_property_readonly("unit", [](const sb::Times& self) {
        return std::string(sb::time_unit_name(self.unit()));
      })
      .def("to", [](const sb::Times& self, const std::string& unit) {
        return self.to(sb::parse_time_unit(unit));
      }, py::arg("unit"))
      .def("__len__", &sb::Times::size)
      .def("__eq__", [](const sb::Times& self, const sb::Times& other) { return self == other; });
  }
}
