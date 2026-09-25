#include "py_point_cloud.hpp"

#include "py_tensor.hpp"

#include <sbear/point_cloud.hpp>
#include <sbear/tensor.hpp>

namespace py = pybind11;

namespace
{
  template <typename T>
  void register_point_cloud(py::module_& m, const std::string& suffix)
  {
    using PointCloud = sb::PointCloud<T>;
    py::class_<PointCloud>(m, ("PointCloud" + suffix).c_str())
      .def(py::init<const sb::Tensor<T>&>())
      .def_property_readonly("n_points", &PointCloud::n_points)
      .def_property_readonly("n_dims", &PointCloud::dim)
      .def_property_readonly("is_indexed", &PointCloud::is_indexed)
      .def_property_readonly("indices", &PointCloud::indices)
      .def_property_readonly("coords", &PointCloud::source_coordinates_copy)
      .def("_coordinate", [](const PointCloud& self, size_t row, size_t column) {
        return self(row, column);
      })
      .def("_storage_array", [](const PointCloud& self) {
        const auto source = self.source_view();
        return py::array_t<T>(
          {static_cast<py::ssize_t>(source.n_points()),
           static_cast<py::ssize_t>(source.dim())},
          {static_cast<py::ssize_t>(source.dim() * sizeof(T)),
           static_cast<py::ssize_t>(sizeof(T))},
          source.storage_data(), py::cast(source));
      })
      .def("copy", &PointCloud::copy);
  }
}

namespace sb_py
{
  void register_point_cloud_bindings(py::module_& m)
  {
    register_point_cloud<sb::float32_t>(m, "32");
    register_point_cloud<sb::float64_t>(m, "64");

    register_indexable_tensor_bindings<sb::PointCloud<sb::float32_t>>(
      m, "PointCloud32");
    register_indexable_tensor_bindings<sb::PointCloud<sb::float64_t>>(
      m, "PointCloud64");

    m.def("cast_pcloud32_pcloud64", [](const sb::Tensor<sb::PointCloud<sb::float32_t>>& src) {
      return sb::pcloud_cast<sb::float64_t>(src);
    });
    m.def("cast_pcloud64_pcloud32", [](const sb::Tensor<sb::PointCloud<sb::float64_t>>& src) {
      return sb::pcloud_cast<sb::float32_t>(src);
    });
  }
}
