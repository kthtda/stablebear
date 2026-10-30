#include "py_tensor.hpp"

#include <sbear/tensor.hpp>
#include <sbear/functional/pcf.hpp>

#include <sstream>


namespace nb = nanobind;

namespace
{

  void register_common_bindings(nb::module_& m)
  {
    nb::class_<sb_py::Shape>(m, "Shape")
      .def(nb::init<std::vector<size_t>>())
      .def("__init__", [](sb_py::Shape* self, size_t n){ new (self) sb_py::Shape{std::vector<size_t>{n}}; }) // 1d construction (Python recognizes (n) as "parenthesis int parenthesis" rather than a tuple of ints)

      .def("__eq__", [](const sb_py::Shape& self, nb::object other) {
        if (nb::isinstance<sb_py::Shape>(other))
        {
          return self.data == nb::cast<const sb_py::Shape&>(other).data;
        }
        else if (nb::isinstance<nb::list>(other))
        {
          std::vector<size_t> values;
          return nb::try_cast(other, values) && self.data == values;
        }
        else if (nb::isinstance<nb::int_>(other))
        {
          return self == sb_py::Shape(nb::cast<size_t>(other));
        }
        else if (nb::isinstance<nb::tuple>(other))
        {
          auto t = nb::borrow<nb::tuple>(other);

          if (t.size() != self.data.size())
          {
            return false;
          }

          return std::equal(self.data.begin(), self.data.end(), t.begin(), [](size_t a, nb::handle b) {
            if (nb::isinstance<nb::int_>(b))
            {
              return a == nb::cast<size_t>(b);
            }

            return false;
          });

        }

        auto type_name = nb::cast<std::string>(other.type().attr("__name__"));
        throw std::runtime_error("Unsupported comparison with object of type " + type_name);
      })

      .def("__getitem__", &sb_py::Shape::dunder_getitem)
      .def("__len__", &sb_py::Shape::dunder_len)
      .def("__repr__", &sb_py::Shape::dunder_repr)
      .def("__str__", &sb_py::Shape::dunder_str)
    ;

    nb::class_<sb::SliceAll>(m, "SliceAll");
    nb::class_<sb::SliceIndex>(m, "SliceIndex");
    nb::class_<sb::SliceRange>(m, "SliceRange");

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

  }

}

namespace sb_py
{
  void register_tensor_bindings(nb::module_& m)
  {
    register_common_bindings(m);

    register_tensor_element<bool>(m, "Bool");

    register_tensor_element<sb::float64_t>(m, "Float64");
    register_tensor_element<sb::float32_t>(m, "Float32");

    register_tensor_int_bindings(m);
    register_tensor_pcf_bindings(m);
  }
}
