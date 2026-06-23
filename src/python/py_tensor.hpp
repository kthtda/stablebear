#pragma once

#ifndef STABLEBEAR_PY_TENSOR_H
#define STABLEBEAR_PY_TENSOR_H

#include "pybind.hpp"
#include "py_np_support.hpp"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <sbear/tensor.hpp>
#include <sbear/concepts.hpp>
#include <sbear/functional/pcf.hpp>
#include "functional/py_pcf_tensor_eval.hpp"

#include <algorithm>
#include <numeric>

namespace sb_py
{
  void register_tensor_bindings(pybind11::module_& m);

  template <typename T>
  struct scalar_of
  {
    using type = T;
  };

  template <typename T>
  requires requires { typename T::value_type; }
  struct scalar_of<T>
  {
    using type = typename T::value_type;
  };

  template <typename T>
  using scalar_of_t = typename scalar_of<T>::type;

  template <typename T>
  struct time_of
  {
    using type = T;
  };

  template <typename T>
  requires requires { typename T::time_type; }
  struct time_of<T>
  {
    using type = typename T::time_type;
  };

  template <typename T>
  using time_of_t = typename time_of<T>::type;

  class Shape
  {
  public:
    std::vector<size_t> data;

    explicit Shape(std::vector<size_t>&& shape)
      : data(std::move(shape))
    { }

    explicit Shape(const std::vector<size_t>& shape)
      : data(shape)
    { }

    explicit Shape(size_t sz) // 1d shape
      : data({sz})
    { }

    [[nodiscard]] bool operator==(const Shape& rhs) const
    {
      return data == rhs.data;
    }

    [[nodiscard]] size_t dunder_getitem(size_t idx) const
    {
      if (idx >= data.size())
      {
        throw pybind11::index_error("Attempted to get index >= len");
      }
      return data[idx];
    }

    [[nodiscard]] size_t dunder_len() const noexcept
    {
      return data.size();
    }

    [[nodiscard]] std::string dunder_repr() const
    {
      std::stringstream ss;
      ss << "(";
      for (auto it = data.begin(); it != data.end(); ++it)
      {
        if (it != data.begin())
        {
          ss << ", ";
        }
        ss << *it;
      }
      ss << ")";
      return ss.str();
    }

    [[nodiscard]] std::string dunder_str() const
    {
      return "Shape" + dunder_repr();
    }
  };



  template <typename TTensor>
  void assert_valid_index(const TTensor& tensor, const std::vector<size_t>& index)
  {
    if (index.size() != tensor.shape().size()
      || !std::equal(index.begin(), index.end(), tensor.shape().begin(), [](size_t i, size_t s){ return i < s; })) // Check that all indices i are < shape[i]
    {
      throw pybind11::index_error("Index out of range");
    }
  }

  template <typename TTensor>
  void assert_valid_index(const TTensor& tensor, size_t index)
  {
    bool ok = tensor.shape().size() == 1 && index < tensor.shape()[0];
    if (!ok)
    {
      throw pybind11::index_error("Index out of range");
    }
  }

  template <typename T>
  void register_typed_tensor_bindings(pybind11::module_& m, const std::string& prefix, const std::string& suffix)
  {
    using TTensor = sb::Tensor<T>;

    pybind11::class_<TTensor> cls = [&m, &prefix, &suffix]
    {
      if constexpr (std::is_trivially_copyable_v<T>)
      {
        pybind11::class_<TTensor> cls(m, (prefix + "Tensor" + suffix).c_str(), pybind11::buffer_protocol());

        cls.def_buffer([](const TTensor& self) -> pybind11::buffer_info
        {
          if (!self.is_contiguous())
          {
            throw std::runtime_error("Noncontiguous tensor not supported.");
          }

          std::vector<pybind11::ssize_t> shape(self.shape().size(), 0);
          std::transform(self.shape().begin(), self.shape().end(), shape.begin(),
              [](size_t v) { return static_cast<pybind11::ssize_t>(v); });

          std::vector<pybind11::ssize_t> strides(self.strides().size(), 0);
          std::transform(self.strides().begin(), self.strides().end(), strides.begin(),
              [](ptrdiff_t v) { return static_cast<pybind11::ssize_t>(v * sizeof(T)); });

          return pybind11::buffer_info(
              static_cast<void*>(self.data() + self.offset()),
              sizeof(T),
              pybind11::format_descriptor<T>::format(),
              self.rank(),
              shape,
              strides
          );
        });

        return cls;
      }
      else
      {
        return pybind11::class_<TTensor>(m, (prefix + "Tensor" + suffix).c_str());
      }
    }();

    cls
      .def(pybind11::init([](const Shape& shape)
        {
          return TTensor(shape.data);
        }))

      .def(pybind11::init([](const Shape& shape, const T& init)
        {
          return TTensor(shape.data, init);
        }))

      .def_property_readonly("shape", [](const TTensor& self){ return Shape{self.shape()}; })
      .def_property_readonly("strides", [](const TTensor& self){ return self.strides(); })
      .def_property_readonly("offset", [](const TTensor& self){ return self.offset(); })

      .def("__getitem__", [](const TTensor& self, const std::vector<sb::Slice>& slices) {
          return self[slices];
        })

      .def("__setitem__", [](TTensor& self, const std::vector<sb::Slice>& slices, const TTensor& vals) {
          self[slices].assign_from(vals);
        })

      .def("__eq__", [](const TTensor& self, const TTensor& rhs){
          return sb::elementwise_eq(self, rhs);
        })
      .def("__ne__", [](const TTensor& self, const TTensor& rhs){
          return sb::elementwise_ne(self, rhs);
        })
      .def("array_equal", [](const TTensor& self, const TTensor& rhs){
          return self == rhs;
        })

      .def("_get_element", [](const TTensor& self, const std::vector<size_t>& index) {
          assert_valid_index(self, index);
          return self(index);
        })

      .def("_get_element", [](TTensor& self, size_t index) {
          assert_valid_index(self, index);
          return self(index);
        })

      .def("_set_element", [](TTensor& self, const std::vector<size_t>& index, const T& val) {
          assert_valid_index(self, index);
          self(index) = val;
        })

      .def("copy", &TTensor::copy)
      .def("flatten", &TTensor::flatten)
      .def("reshape", &TTensor::reshape)
      .def("transpose", &TTensor::transpose, pybind11::arg("axes") = std::vector<size_t>{})
      .def("swapaxes", &TTensor::swapaxes, pybind11::arg("axis1"), pybind11::arg("axis2"))
      .def("squeeze", [](const TTensor& self) { return self.squeeze(); })
      .def("squeeze", [](const TTensor& self, size_t axis) { return self.squeeze(axis); }, pybind11::arg("axis"))
      .def("expand_dims", &TTensor::expand_dims, pybind11::arg("axis"))
      .def_static("concatenate", [](const std::vector<TTensor>& tensors, size_t axis) {
        return sb::concatenate(tensors, axis);
      }, pybind11::arg("tensors"), pybind11::arg("axis") = 0)
      .def_static("stack", [](const std::vector<TTensor>& tensors, ptrdiff_t axis) {
        return sb::stack(tensors, axis);
      }, pybind11::arg("tensors"), pybind11::arg("axis") = 0)
      .def_static("split_sections", [](const TTensor& tensor, size_t n_sections, size_t axis) {
        return sb::split(tensor, n_sections, axis);
      }, pybind11::arg("tensor"), pybind11::arg("n_sections"), pybind11::arg("axis") = 0)
      .def_static("split_indices", [](const TTensor& tensor, const std::vector<size_t>& indices, size_t axis) {
        return sb::split(tensor, indices, axis);
      }, pybind11::arg("tensor"), pybind11::arg("indices"), pybind11::arg("axis") = 0)
      .def_static("array_split", [](const TTensor& tensor, size_t n_sections, size_t axis) {
        return sb::array_split(tensor, n_sections, axis);
      }, pybind11::arg("tensor"), pybind11::arg("n_sections"), pybind11::arg("axis") = 0)
      .def("is_contiguous", &TTensor::is_contiguous)
    ;

    // Unary negation
    if constexpr (sb::CanNegate<T>)
    {
      cls.def("__neg__", [](const TTensor& self){ return -self; });
    }

    if constexpr (sb::FloatType<T>)
    {
      cls.def("allclose", [](const TTensor& self, const TTensor& rhs, double atol, double rtol){
        return sb::allclose(self, rhs, T(atol), T(rtol));
      }, py::arg("other"), py::arg("atol") = 1e-8, py::arg("rtol") = 1e-5);
    }

    // Ordered comparisons (broadcasting, returns BoolTensor)
    if constexpr (sb::CanOrder<T>)
    {
      cls
        .def("__lt__", [](const TTensor& self, const TTensor& rhs){ return sb::elementwise_lt(self, rhs); })
        .def("__le__", [](const TTensor& self, const TTensor& rhs){ return sb::elementwise_le(self, rhs); })
        .def("__gt__", [](const TTensor& self, const TTensor& rhs){ return sb::elementwise_gt(self, rhs); })
        .def("__ge__", [](const TTensor& self, const TTensor& rhs){ return sb::elementwise_ge(self, rhs); })
      ;
    }

    // Tensor-Tensor arithmetic (broadcasting)
    if constexpr (sb::CanAddTo<T, T, T>)
    {
      cls
        .def("__add__", [](const TTensor& self, const TTensor& rhs){ return self + rhs; })
        .def("__iadd__", [](TTensor& self, const TTensor& rhs) -> TTensor& { self += rhs; return self; })
      ;
    }

    if constexpr (sb::CanSubtractTo<T, T, T>)
    {
      cls
        .def("__sub__", [](const TTensor& self, const TTensor& rhs){ return self - rhs; })
        .def("__isub__", [](TTensor& self, const TTensor& rhs) -> TTensor& { self -= rhs; return self; })
      ;
    }

    if constexpr (sb::CanMultiplyTo<T, T, T>)
    {
      cls
        .def("__mul__", [](const TTensor& self, const TTensor& rhs){ return self * rhs; })
        .def("__imul__", [](TTensor& self, const TTensor& rhs) -> TTensor& { self *= rhs; return self; })
      ;
    }

    if constexpr (sb::CanDivideTo<T, T, T>)
    {
      cls
        .def("__truediv__", [](const TTensor& self, const TTensor& rhs){ return self / rhs; })
        .def("__itruediv__", [](TTensor& self, const TTensor& rhs) -> TTensor& { self /= rhs; return self; })
      ;
    }

    cls.def("broadcast_to", [](const TTensor& self, const std::vector<size_t>& shape){ return self.broadcast_to(shape); });

    // Masked operations
    cls.def("masked_select", [](const TTensor& self, const sb::Tensor<bool>& mask) {
      return sb::masked_select(self, mask);
    });
    cls.def("masked_assign", [](TTensor& self, const sb::Tensor<bool>& mask, const TTensor& values) {
      sb::masked_assign(self, mask, values);
    });
    cls.def("masked_fill", [](TTensor& self, const sb::Tensor<bool>& mask, const T& value) {
      sb::masked_fill(self, mask, value);
    });
    cls.def("axis_select", [](const TTensor& self, size_t axis, const sb::Tensor<bool>& mask) {
      return sb::axis_select(self, axis, mask);
    });
    cls.def("axis_assign", [](TTensor& self, size_t axis, const sb::Tensor<bool>& mask, const TTensor& values) {
      sb::axis_assign(self, axis, mask, values);
    });
    cls.def("axis_fill", [](TTensor& self, size_t axis, const sb::Tensor<bool>& mask, const T& value) {
      sb::axis_fill(self, axis, mask, value);
    });
    cls.def("multi_axis_select", [](const TTensor& self, const std::vector<std::pair<size_t, sb::Tensor<bool>>>& axis_masks) {
      return sb::multi_axis_select(self, axis_masks);
    });
    cls.def("multi_axis_assign", [](TTensor& self, const std::vector<std::pair<size_t, sb::Tensor<bool>>>& axis_masks, const TTensor& values) {
      sb::multi_axis_assign(self, axis_masks, values);
    });
    cls.def("multi_axis_fill", [](TTensor& self, const std::vector<std::pair<size_t, sb::Tensor<bool>>>& axis_masks, const T& value) {
      sb::multi_axis_fill(self, axis_masks, value);
    });

    cls.def("outer_select", [](const TTensor& self, const std::vector<std::pair<size_t, sb::AxisSelector>>& selectors) {
      return sb::outer_select(self, selectors);
    });
    cls.def("outer_assign", [](TTensor& self, const std::vector<std::pair<size_t, sb::AxisSelector>>& selectors, const TTensor& values) {
      sb::outer_assign(self, selectors, values);
    });
    cls.def("outer_fill", [](TTensor& self, const std::vector<std::pair<size_t, sb::AxisSelector>>& selectors, const T& value) {
      sb::outer_fill(self, selectors, value);
    });

    // Index-based gather/scatter (always use int64 as index type)
    cls.def("index_select", [](const TTensor& self, size_t axis, const sb::Tensor<sb::int64_t>& indices) {
      return sb::index_select(self, axis, indices);
    });
    cls.def("index_assign", [](TTensor& self, size_t axis, const sb::Tensor<sb::int64_t>& indices, const TTensor& values) {
      sb::index_assign(self, axis, indices, values);
    });
    cls.def("index_fill", [](TTensor& self, size_t axis, const sb::Tensor<sb::int64_t>& indices, const T& value) {
      sb::index_fill(self, axis, indices, value);
    });

    using Tv = scalar_of_t<T>;
    using Tt = time_of_t<T>;

    if constexpr (sb::CanAddTo<T, T, T>)
    {
      cls
        .def("__add__", [](const TTensor& self, const T& rhs){ return self + rhs; })
        .def("__radd__", [](const TTensor& self, const T& lhs){ return lhs + self; })
        .def("__iadd__", [](TTensor& self, const T& rhs) -> TTensor& { self += rhs; return self; })
      ;
    }

    if constexpr (sb::CanAddTo<T, T, Tv>)
    {
      cls
        .def("__add__", [](const TTensor& self, Tv rhs){ return self + rhs; })
        .def("__iadd__", [](TTensor& self, Tv rhs) -> TTensor& { self += rhs; return self; })
      ;
    }

    if constexpr (sb::CanAddTo<T, Tv, T>)
    {
      cls.def("__radd__", [](const TTensor& self, Tv lhs){ return lhs + self; });
    }

    if constexpr (sb::CanSubtractTo<T, T, T>)
    {
      cls
        .def("__sub__", [](const TTensor& self, const T& rhs){ return self - rhs; })
        .def("__rsub__", [](const TTensor& self, const T& lhs){ return lhs - self; })
        .def("__isub__", [](TTensor& self, const T& rhs) -> TTensor& { self -= rhs; return self; })
      ;
    }

    if constexpr (sb::CanSubtractTo<T, T, Tv>)
    {
      cls
        .def("__sub__", [](const TTensor& self, Tv rhs){ return self - rhs; })
        .def("__isub__", [](TTensor& self, Tv rhs) -> TTensor& { self -= rhs; return self; })
      ;
    }

    if constexpr (sb::CanSubtractTo<T, Tv, T>)
    {
      cls.def("__rsub__", [](const TTensor& self, Tv lhs){ return lhs - self; });
    }

    if constexpr (sb::CanMultiplyTo<T, T, Tv>)
    {
      cls
        .def("__mul__", [](const TTensor& self, Tv rhs){ return self * rhs; })
        .def("__imul__", [](TTensor& self, Tv rhs) -> TTensor& { self *= rhs; return self; })
      ;
    }

    if constexpr (sb::CanMultiplyTo<T, Tv, T>)
    {
      cls.def("__rmul__", [](const TTensor& self, Tv lhs){ return lhs * self; });
    }

    if constexpr (sb::CanDivideTo<T, T, Tv>)
    {
      cls
        .def("__truediv__", [](const TTensor& self, Tv rhs){ return self / rhs; })
        .def("__itruediv__", [](TTensor& self, Tv rhs) -> TTensor& { self /= rhs; return self; })
      ;
    }

    if constexpr (sb::CanDivideTo<T, Tv, T>)
    {
      cls.def("__rtruediv__", [](const TTensor& self, Tv lhs){ return lhs / self; });
    }

    if constexpr (sb::CanPow<T, Tv>)
    {
      cls.def("__pow__", [](const TTensor& self, Tv exponent) {
        auto result = sb::pow(self, exponent);
        bool warned = result.any_of([](const T& elem) {
          if constexpr (sb::PcfLike<T>)
            return std::ranges::any_of(elem.points(), [](const auto& pt) {
              return std::isnan(pt.v) || std::isinf(pt.v);
            });
          else
            return std::isnan(elem) || std::isinf(elem);
        });
        if (warned)
        {
          PyErr_WarnEx(PyExc_RuntimeWarning,
            "invalid or infinite value encountered in pow", 1);
        }
        return result;
      });

      cls.def("__ipow__", [](TTensor& self, Tv exponent) -> TTensor& {
        sb::ipow(self, exponent);
        return self;
      });
    }

    if constexpr (sb::PcfLike<T>)
    {
      cls.def("__call__", [](const TTensor& self, Tt t) {
        return sb_py::pcf_tensor_eval_scalar<Tt, Tv>(self, t);
      });

      cls.def("__call__", [](const TTensor& self, py::array_t<Tt> times) {
        NumpyTensor<Tt> t_in(times);
        auto sh = sb_py::eval_out_shape(self, t_in);
        std::vector<py::ssize_t> out_shape(sh.begin(), sh.end());
        py::array_t<Tv> result(out_shape);
        NumpyTensor<Tv> out(result);
        sb::tensor_eval<Tt, Tv>(self, t_in, out);
        return result;
      });

      cls.def("__call__", [](const TTensor& self, const sb::Tensor<Tt>& times) {
        sb::Tensor<Tv> out(sb_py::eval_out_shape(self, times));
        sb::tensor_eval<Tt, Tv>(self, times, out);
        return out;
      });
    }

  }
}

#endif //STABLEBEAR_PY_TENSOR_H