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
      .def(py::init<const sb::Tensor<T>&, sb::Tensor<sb::uint64_t>>())
      .def_property_readonly("n_points", &PointCloud::n_points)
      .def_property_readonly("n_dims", &PointCloud::dim)
      .def_property_readonly("is_indexed", &PointCloud::is_indexed)
      .def_property_readonly("indices", &PointCloud::indices)
      .def_property_readonly("coords", &PointCloud::coords)
      .def("_mutable_coords", &PointCloud::mutable_coords,
        py::return_value_policy::reference_internal)
      .def("materialize", &PointCloud::materialize)
      .def("copy", &PointCloud::copy, py::arg("keep_source") = true);
  }
}

namespace sb_py
{
  void register_point_cloud_bindings(py::module_& m)
  {
    register_point_cloud<sb::float32_t>(m, "32");
    register_point_cloud<sb::float64_t>(m, "64");

    register_typed_tensor_bindings<sb::PointCloud<sb::float32_t>>(m, "PointCloud32", "");
    register_typed_tensor_bindings<sb::PointCloud<sb::float64_t>>(m, "PointCloud64", "");

    register_typed_tensor_bindings<
      sb::PointCloud<sb::float32_t>, sb::TensorProperty::Indexed>(
      m, "_IndexedPointCloud32", "");
    register_typed_tensor_bindings<
      sb::PointCloud<sb::float64_t>, sb::TensorProperty::Indexed>(
      m, "_IndexedPointCloud64", "");

    m.def("cast_pcloud32_pcloud64", [](const sb::Tensor<sb::PointCloud<sb::float32_t>>& src) {
      return sb::pcloud_cast<sb::float64_t>(src);
    });
    m.def("cast_pcloud64_pcloud32", [](const sb::Tensor<sb::PointCloud<sb::float64_t>>& src) {
      return sb::pcloud_cast<sb::float32_t>(src);
    });
  }
}
