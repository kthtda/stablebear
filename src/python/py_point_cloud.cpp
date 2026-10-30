#include "py_point_cloud.hpp"

#include "py_tensor.hpp"

#include <sbear/point_cloud.hpp>
#include <sbear/tensor.hpp>

namespace nb = nanobind;

namespace
{
  template <typename T>
  void register_point_cloud(nb::module_& m, const std::string& suffix)
  {
    using PointCloud = sb::PointCloud<T>;
    nb::class_<PointCloud>(m, ("PointCloud" + suffix).c_str())
      .def(nb::init<const sb::Tensor<T>&>())
      .def_prop_ro("n_points", &PointCloud::n_points)
      .def_prop_ro("n_dims", &PointCloud::dim)
      .def_prop_ro("is_indexed", &PointCloud::is_indexed)
      .def_prop_ro("indices", &PointCloud::indices)
      .def("to_float_tensor", &PointCloud::source_coordinates_copy,
           "Return an independent float tensor of the full source coordinates, without applying indices.")
      .def("_coordinate", [](const PointCloud& self, size_t row, size_t column) {
        return self(row, column);
      })
      .def("_storage_array", [](const PointCloud& self) {
        // The view shares the source storage; its Python object keeps it alive.
        nb::object source = nb::cast(self.view_source());
        const auto& view = nb::cast<const PointCloud&>(source);
        return sb_py::array_view<T>(const_cast<T*>(view.storage_data()), {view.n_points(), view.dim()},
                                    {static_cast<int64_t>(view.dim()), 1}, source);
      })
      .def("copy", &PointCloud::copy);
  }
}

namespace sb_py
{
  void register_point_cloud_bindings(nb::module_& m)
  {
    register_point_cloud<sb::float32_t>(m, "32");
    register_point_cloud<sb::float64_t>(m, "64");

    register_tensor_element<sb::PointCloud<sb::float32_t>>(m, "PointCloud32");
    register_tensor_element<sb::PointCloud<sb::float64_t>>(m, "PointCloud64");

    m.def("cast_pcloud32_pcloud64", [](const sb::Tensor<sb::PointCloud<sb::float32_t>>& src) {
      return sb::pcloud_cast<sb::float64_t>(src);
    });
    m.def("cast_pcloud64_pcloud32", [](const sb::Tensor<sb::PointCloud<sb::float64_t>>& src) {
      return sb::pcloud_cast<sb::float32_t>(src);
    });
  }
}
