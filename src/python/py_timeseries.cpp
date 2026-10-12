#include "py_timeseries.hpp"

#include <sbear/distance_matrix.hpp>
#include <sbear/functional/pcf.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/point_cloud.hpp>
#include <sbear/symmetric_matrix.hpp>
#include <sbear/tensor.hpp>
#include <sbear/timeseries.hpp>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <string>
#include <vector>

namespace py = pybind11;

namespace
{
  /// Bind the channel and series classes for element type T, as
  /// register_nested_tensor_element does for nested tensors.
  template <typename T>
  void register_timeseries_element(py::module_& m, const std::string& suffix)
  {
    using Channel = sb::Channel<T>;
    using Series = sb::TimeSeries<T>;

    py::class_<Channel>(m, ("TimeSeriesChannel" + suffix).c_str())
      .def_property_readonly("times", &Channel::times)
      .def_property_readonly("shape", [](const Channel& self) {
        return py::tuple(py::cast(self.shape()));
      })
      .def_property_readonly("values", &Channel::values)
      .def("__len__", &Channel::size);

    py::class_<Series>(m, ("TimeSeries" + suffix).c_str())
      .def(py::init<sb::Times, const sb::Tensor<T>&>(), py::arg("times"), py::arg("values"))
      .def_property_readonly("channel_names", [](const Series& self) {
        return py::tuple(py::cast(self.channel_names()));
      })
      .def_property_readonly("unit", [](const Series& self) {
        return std::string(sb::time_unit_name(self.unit()));
      })
      .def("channel", &Series::channel, py::arg("index"), py::return_value_policy::reference_internal)
      .def_property_readonly("n_channels", &Series::n_channels);
  }
}

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

    register_timeseries_element<bool>(m, "Bool");
    register_timeseries_element<sb::float32_t>(m, "Float32");
    register_timeseries_element<sb::float64_t>(m, "Float64");
    register_timeseries_element<sb::int32_t>(m, "Int32");
    register_timeseries_element<sb::int64_t>(m, "Int64");
    register_timeseries_element<uint32_t>(m, "Uint32");
    register_timeseries_element<uint64_t>(m, "Uint64");
    register_timeseries_element<sb::Pcf_f32>(m, "Pcf32");
    register_timeseries_element<sb::Pcf_f64>(m, "Pcf64");
    register_timeseries_element<sb::Pcf_i32>(m, "Pcf32i");
    register_timeseries_element<sb::Pcf_i64>(m, "Pcf64i");
    register_timeseries_element<sb::PointCloud<sb::float32_t>>(m, "PointCloud32");
    register_timeseries_element<sb::PointCloud<sb::float64_t>>(m, "PointCloud64");
    register_timeseries_element<sb::DistanceMatrix<sb::float32_t>>(m, "DistanceMatrix32");
    register_timeseries_element<sb::DistanceMatrix<sb::float64_t>>(m, "DistanceMatrix64");
    register_timeseries_element<sb::SymmetricMatrix<sb::float32_t>>(m, "SymmetricMatrix32");
    register_timeseries_element<sb::SymmetricMatrix<sb::float64_t>>(m, "SymmetricMatrix64");
    register_timeseries_element<sb::ph::Barcode<sb::float32_t>>(m, "Barcode32");
    register_timeseries_element<sb::ph::Barcode<sb::float64_t>>(m, "Barcode64");
  }
}
