#include "py_tensor.hpp"

#include <sbear/tensor.hpp>
#include <sbear/point_cloud.hpp>
#include <sbear/functional/pcf.hpp>

#include <sstream>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace
{

  void register_common_bindings(pybind11::module_& m)
  {
    pybind11::class_<sb_py::Shape>(m, "Shape")
      .def(pybind11::init<std::vector<size_t>>())
      .def(pybind11::init<>([](size_t n){ return sb_py::Shape{std::vector<size_t>{n}}; })) // 1d construction (Python recognizes (n) as "parenthesis int parenthesis" rather than a tuple of ints)

      .def("__eq__", [](const sb_py::Shape& self, pybind11::object other) {
        if (pybind11::isinstance<sb_py::Shape>(other))
        {
          return self.data == other.cast<sb_py::Shape>().data;
        }
        else if (pybind11::isinstance<std::vector<size_t>>(other))
        {
          return self.data == other.cast<std::vector<size_t>>();
        }
        else if (pybind11::isinstance<size_t>(other) || pybind11::isinstance<pybind11::int_>(other))
        {
          return self == sb_py::Shape(other.cast<size_t>());
        }
        else if (pybind11::isinstance<pybind11::tuple>(other))
        {
          auto t = other.cast<pybind11::tuple>();

          if (t.size() != self.data.size())
          {
            return false;
          }

          return std::equal(self.data.begin(), self.data.end(), t.begin(), [](size_t a, pybind11::handle b) {
            if (pybind11::isinstance<pybind11::int_>(b))
            {
              return a == b.cast<size_t>();
            }

            return false;
          });

        }

        std::string type_name = pybind11::str(pybind11::type::handle_of(other).attr("__name__"));
        throw std::runtime_error("Unsupported comparison with object of type " + type_name);
      })

      .def("__getitem__", &sb_py::Shape::dunder_getitem)
      .def("__len__", &sb_py::Shape::dunder_len)
      .def("__repr__", &sb_py::Shape::dunder_repr)
      .def("__str__", &sb_py::Shape::dunder_str)
    ;

    pybind11::class_<sb::SliceAll>(m, "SliceAll");
    pybind11::class_<sb::SliceIndex>(m, "SliceIndex");
    pybind11::class_<sb::SliceRange>(m, "SliceRange");

    m.def("slice_all", [](){ return sb::all(); });
    m.def("slice_index", [](ptrdiff_t idx){ return sb::index(idx); });
    m.def("slice_range", [](std::optional<ptrdiff_t> start, std::optional<ptrdiff_t> stop, std::optional<ptrdiff_t> step){ return sb::range(start, stop, step); });

    // tensor_cast bindings — each pair registered as a module-level function
    auto tc = [&m]<typename To, typename From>(const char* name) {
      m.def(name, [](const sb::Tensor<From>& src) { return sb::tensor_cast<To>(src); });
    };

    // Float precision
    tc.template operator()<sb::float64_t, sb::float32_t>("cast_f32_f64");
    tc.template operator()<sb::float32_t, sb::float64_t>("cast_f64_f32");
    // Int precision
    tc.template operator()<sb::int64_t, sb::int32_t>("cast_i32_i64");
    tc.template operator()<sb::int32_t, sb::int64_t>("cast_i64_i32");
    tc.template operator()<uint64_t, uint32_t>("cast_u32_u64");
    tc.template operator()<uint32_t, uint64_t>("cast_u64_u32");
    // Float <-> Int
    tc.template operator()<sb::int32_t, sb::float64_t>("cast_f64_i32");
    tc.template operator()<sb::int64_t, sb::float64_t>("cast_f64_i64");
    tc.template operator()<sb::int32_t, sb::float32_t>("cast_f32_i32");
    tc.template operator()<sb::int64_t, sb::float32_t>("cast_f32_i64");
    tc.template operator()<sb::float32_t, sb::int32_t>("cast_i32_f32");
    tc.template operator()<sb::float64_t, sb::int32_t>("cast_i32_f64");
    tc.template operator()<sb::float32_t, sb::int64_t>("cast_i64_f32");
    tc.template operator()<sb::float64_t, sb::int64_t>("cast_i64_f64");
    // Int <-> UInt
    tc.template operator()<uint32_t, sb::int32_t>("cast_i32_u32");
    tc.template operator()<uint64_t, sb::int32_t>("cast_i32_u64");
    tc.template operator()<uint32_t, sb::int64_t>("cast_i64_u32");
    tc.template operator()<uint64_t, sb::int64_t>("cast_i64_u64");
    tc.template operator()<sb::int32_t, uint32_t>("cast_u32_i32");
    tc.template operator()<sb::int64_t, uint32_t>("cast_u32_i64");
    tc.template operator()<sb::int32_t, uint64_t>("cast_u64_i32");
    tc.template operator()<sb::int64_t, uint64_t>("cast_u64_i64");
    // PCF precision
    tc.template operator()<sb::Pcf_f64, sb::Pcf_f32>("cast_pcf32_pcf64");
    tc.template operator()<sb::Pcf_f32, sb::Pcf_f64>("cast_pcf64_pcf32");
    tc.template operator()<sb::Pcf_i64, sb::Pcf_i32>("cast_pcf32i_pcf64i");
    tc.template operator()<sb::Pcf_i32, sb::Pcf_i64>("cast_pcf64i_pcf32i");

    // PointCloud precision: Tensor<Tensor<float>> <-> Tensor<Tensor<double>>
    m.def("cast_pcloud32_pcloud64", [](const sb::Tensor<sb::PointCloud<sb::float32_t>>& src) {
      return sb::pcloud_cast<sb::float64_t>(src);
    });
    m.def("cast_pcloud64_pcloud32", [](const sb::Tensor<sb::PointCloud<sb::float64_t>>& src) {
      return sb::pcloud_cast<sb::float32_t>(src);
    });

  }

  // The element type of a PointCloud tensor. Registered as a subclass of the
  // scalar Tensor<T> so it can be returned to Python; an element may be an indexed
  // view sharing a source cloud (materialized lazily — see stablebear.tensor.PointCloud).
  template <typename T>
  void register_point_cloud_element(py::module_& m, const std::string& suffix)
  {
    using PC = sb::PointCloud<T>;
    py::class_<PC, sb::Tensor<T>>(m, ("PointCloud" + suffix + "_inner").c_str())
        .def(py::init<const sb::Tensor<T>&>())
        .def_property_readonly("n_points", &PC::n_points)
        .def_property_readonly("n_dims", &PC::dim)
        .def_property_readonly("is_indexed", &PC::is_indexed)
        .def_property_readonly("indices", &PC::indices)
        .def("materialize", &PC::materialize);
  }

}

namespace sb_py
{
  void register_tensor_bindings(py::module_& m)
  {
    register_common_bindings(m);

    register_typed_tensor_bindings<bool>(m, "Bool", "");

    register_typed_tensor_bindings<sb::float64_t>(m, "Float64", "");
    register_typed_tensor_bindings<sb::float32_t>(m, "Float32", "");

    register_typed_tensor_bindings<sb::int32_t>(m, "Int32", "");
    register_typed_tensor_bindings<sb::int64_t>(m, "Int64", "");
    register_typed_tensor_bindings<uint32_t>(m, "Uint32", "");
    register_typed_tensor_bindings<uint64_t>(m, "Uint64", "");

    register_typed_tensor_bindings<sb::Pcf_f32>(m, "Pcf32", "");
    register_typed_tensor_bindings<sb::Pcf_f64>(m, "Pcf64", "");
    register_typed_tensor_bindings<sb::Pcf_i32>(m, "Pcf32i", "");
    register_typed_tensor_bindings<sb::Pcf_i64>(m, "Pcf64i", "");

    register_typed_tensor_bindings<sb::PointCloud<sb::float32_t>>(m, "PointCloud32", "");
    register_typed_tensor_bindings<sb::PointCloud<sb::float64_t>>(m, "PointCloud64", "");

    register_point_cloud_element<sb::float32_t>(m, "32");
    register_point_cloud_element<sb::float64_t>(m, "64");
  }
}
