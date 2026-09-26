#pragma once

#ifndef STABLEBEAR_PY_TENSOR_H
#define STABLEBEAR_PY_TENSOR_H

#include "pybind.hpp"
#include "py_np_support.hpp"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <sbear/tensor.hpp>
#include <sbear/nested_tensor.hpp>
#include <sbear/concepts.hpp>
#include <sbear/functional/pcf.hpp>
#include "functional/py_pcf_tensor_eval.hpp"

#include <algorithm>
#include <numeric>
#include <type_traits>

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

  template <typename TTensor, typename Index>
  decltype(auto) writable_element(TTensor& tensor, const Index& index)
  {
    if constexpr (TTensor::IsIndexed)
      return tensor.writable_at(index);
    else
      return tensor(index);
  }

  template <typename TTensor>
  void bind_tensor_view_operations(pybind11::class_<TTensor>& cls)
  {
    cls
      .def_property_readonly("shape", [](const TTensor& self){ return Shape{self.shape()}; })
      .def_property_readonly("strides", [](const TTensor& self){ return self.strides(); })
      .def_property_readonly("offset", [](const TTensor& self) {
        if constexpr (TTensor::IsIndexed)
          return ptrdiff_t{0};
        else
          return self.offset();
      })
      .def("__getitem__", [](const TTensor& self, const std::vector<sb::Slice>& slices) {
        return self[slices];
      })
      .def("_get_element", [](const TTensor& self, const std::vector<size_t>& index) {
        assert_valid_index(self, index);
        return self(index);
      })
      .def("_get_element", [](const TTensor& self, size_t index) {
        assert_valid_index(self, index);
        return self(index);
      })
      .def("_get_writeable_element", [](TTensor& self,
        const std::vector<size_t>& index) -> decltype(auto) {
        assert_valid_index(self, index);
        return writable_element(self, index);
      }, pybind11::return_value_policy::reference_internal)
      .def("_get_writeable_element", [](TTensor& self, size_t index) -> decltype(auto) {
        assert_valid_index(self, index);
        return writable_element(self, index);
      }, pybind11::return_value_policy::reference_internal)
      .def("copy", &TTensor::copy)
      .def("flatten", &TTensor::flatten)
      .def("reshape", &TTensor::reshape)
      .def("transpose", &TTensor::transpose, pybind11::arg("axes") = std::vector<size_t>{})
      .def("swapaxes", &TTensor::swapaxes, pybind11::arg("axis1"), pybind11::arg("axis2"))
      .def("squeeze", [](const TTensor& self) { return self.squeeze(); })
      .def("squeeze", [](const TTensor& self, size_t axis) { return self.squeeze(axis); }, pybind11::arg("axis"))
      .def("expand_dims", &TTensor::expand_dims, pybind11::arg("axis"))
      .def("broadcast_to", [](const TTensor& self, const std::vector<size_t>& shape) {
        return self.broadcast_to(shape);
      })
      .def("is_contiguous", &TTensor::is_contiguous);

    if constexpr (TTensor::IsIndexed)
      cls.def("_ensure_materialized", &TTensor::ensure_materialized);
    
    cls.def("has_indices", [](const TTensor& t) { 
      if constexpr (TTensor::IsIndexed)
        return t.has_indices();
      else
        return false; 
    });
  }

  /// Bind operations that depend only on the logical Tensor interface.  The
  /// property bitmask controls storage behavior in C++; Python sees the same
  /// API for ordinary and indexed tensors.
  template <typename T, sb::TensorProperties Properties>
  void bind_tensor_operations(pybind11::class_<sb::Tensor<T, Properties>>& cls)
  {
    using TTensor = sb::Tensor<T, Properties>;
    static constexpr sb::TensorProperties PlainProperties =
      Properties & ~sb::TensorProperty::Indexed;
    using TPlain = sb::Tensor<T, PlainProperties>;
    using TIndexed = sb::Tensor<T, PlainProperties | sb::TensorProperty::Indexed>;

    cls
      .def("__setitem__", [](TTensor& self, const std::vector<sb::Slice>& slices,
          const TPlain& values) { self[slices].assign_from(values); })
      .def("__eq__", [](const TTensor& self, const TPlain& rhs) {
        return sb::elementwise_eq(self, rhs);
      })
      .def("__ne__", [](const TTensor& self, const TPlain& rhs) {
        return sb::elementwise_ne(self, rhs);
      })
      .def("array_equal", [](const TTensor& self, const TPlain& rhs) {
        return self == rhs;
      })
      .def("_set_element", [](TTensor& self,
          const std::vector<size_t>& index, const T& value) {
        assert_valid_index(self, index);
        writable_element(self, index) = sb::detail::store_copy(value);
      })
      .def_static("concatenate", [](const pybind11::iterable& values, size_t axis) {
        std::vector<TPlain> tensors;
        for (const auto item : values)
        {
          if (pybind11::isinstance<TPlain>(item))
            tensors.push_back(pybind11::cast<TPlain>(item));
          else if constexpr (sb::IndexableTensorElement<T>)
          {
            if (pybind11::isinstance<TIndexed>(item))
              tensors.push_back(pybind11::cast<TIndexed>(item).materialize());
            else
              throw pybind11::type_error("all tensors must have the same element type and properties");
          }
          else
            throw pybind11::type_error("all tensors must have the same element type and properties");
        }
        return sb::concatenate(tensors, axis);
      }, pybind11::arg("tensors"), pybind11::arg("axis") = 0)
      .def_static("stack", [](const pybind11::iterable& values, ptrdiff_t axis) {
        std::vector<TPlain> tensors;
        for (const auto item : values)
        {
          if (pybind11::isinstance<TPlain>(item))
            tensors.push_back(pybind11::cast<TPlain>(item));
          else if constexpr (sb::IndexableTensorElement<T>)
          {
            if (pybind11::isinstance<TIndexed>(item))
              tensors.push_back(pybind11::cast<TIndexed>(item).materialize());
            else
              throw pybind11::type_error("all tensors must have the same element type and properties");
          }
          else
            throw pybind11::type_error("all tensors must have the same element type and properties");
        }
        return sb::stack(tensors, axis);
      }, pybind11::arg("tensors"), pybind11::arg("axis") = 0)
      .def_static("split_sections", [](const TTensor& tensor,
          size_t sections, size_t axis) { return sb::split(tensor, sections, axis); },
          pybind11::arg("tensor"), pybind11::arg("n_sections"), pybind11::arg("axis") = 0)
      .def_static("split_indices", [](const TTensor& tensor,
          const std::vector<size_t>& indices, size_t axis) {
        return sb::split(tensor, indices, axis);
      }, pybind11::arg("tensor"), pybind11::arg("indices"), pybind11::arg("axis") = 0)
      .def_static("array_split", [](const TTensor& tensor,
          size_t sections, size_t axis) { return sb::array_split(tensor, sections, axis); },
          pybind11::arg("tensor"), pybind11::arg("n_sections"), pybind11::arg("axis") = 0)
      .def("masked_select", [](const TTensor& self, const sb::Tensor<bool>& mask) {
        return sb::masked_select(self, mask);
      })
      .def("masked_assign", [](TTensor& self, const sb::Tensor<bool>& mask,
          const TPlain& values) { sb::masked_assign(self, mask, values); })
      .def("masked_fill", [](TTensor& self, const sb::Tensor<bool>& mask,
          const T& value) { sb::masked_fill(self, mask, value); })
      .def("axis_select", [](const TTensor& self, size_t axis,
          const sb::Tensor<bool>& mask) { return sb::axis_select(self, axis, mask); })
      .def("axis_assign", [](TTensor& self, size_t axis, const sb::Tensor<bool>& mask,
          const TPlain& values) { sb::axis_assign(self, axis, mask, values); })
      .def("axis_fill", [](TTensor& self, size_t axis, const sb::Tensor<bool>& mask,
          const T& value) { sb::axis_fill(self, axis, mask, value); })
      .def("multi_axis_select", [](const TTensor& self,
          const std::vector<std::pair<size_t, sb::Tensor<bool>>>& masks) {
        return sb::multi_axis_select(self, masks);
      })
      .def("multi_axis_assign", [](TTensor& self,
          const std::vector<std::pair<size_t, sb::Tensor<bool>>>& masks,
          const TPlain& values) { sb::multi_axis_assign(self, masks, values); })
      .def("multi_axis_fill", [](TTensor& self,
          const std::vector<std::pair<size_t, sb::Tensor<bool>>>& masks,
          const T& value) { sb::multi_axis_fill(self, masks, value); })
      .def("outer_select", [](const TTensor& self,
          const std::vector<std::pair<size_t, sb::AxisSelector>>& selectors) {
        return sb::outer_select(self, selectors);
      })
      .def("outer_assign", [](TTensor& self,
          const std::vector<std::pair<size_t, sb::AxisSelector>>& selectors,
          const TPlain& values) { sb::outer_assign(self, selectors, values); })
      .def("outer_fill", [](TTensor& self,
          const std::vector<std::pair<size_t, sb::AxisSelector>>& selectors,
          const T& value) { sb::outer_fill(self, selectors, value); })
      .def("index_select", [](const TTensor& self, size_t axis,
          const sb::Tensor<sb::int64_t>& indices) {
        return sb::index_select(self, axis, indices);
      })
      .def("index_assign", [](TTensor& self, size_t axis,
          const sb::Tensor<sb::int64_t>& indices, const TPlain& values) {
        sb::index_assign(self, axis, indices, values);
      })
      .def("index_fill", [](TTensor& self, size_t axis,
          const sb::Tensor<sb::int64_t>& indices, const T& value) {
        sb::index_fill(self, axis, indices, value);
      });

    if constexpr (sb::IndexableTensorElement<T>)
    {
      cls
        .def("__setitem__", [](TTensor& self, const std::vector<sb::Slice>& slices,
            const TIndexed& values) { self[slices].assign_from(values); })
        .def("__eq__", [](const TTensor& self, const TIndexed& rhs) {
          return sb::elementwise_eq(self, rhs);
        })
        .def("__ne__", [](const TTensor& self, const TIndexed& rhs) {
          return sb::elementwise_ne(self, rhs);
        })
        .def("array_equal", [](const TTensor& self, const TIndexed& rhs) {
          return self == rhs;
        })
        .def("masked_assign", [](TTensor& self, const sb::Tensor<bool>& mask,
            const TIndexed& values) { sb::masked_assign(self, mask, values); })
        .def("axis_assign", [](TTensor& self, size_t axis, const sb::Tensor<bool>& mask,
            const TIndexed& values) { sb::axis_assign(self, axis, mask, values); })
        .def("multi_axis_assign", [](TTensor& self,
            const std::vector<std::pair<size_t, sb::Tensor<bool>>>& masks,
            const TIndexed& values) { sb::multi_axis_assign(self, masks, values); })
        .def("outer_assign", [](TTensor& self,
            const std::vector<std::pair<size_t, sb::AxisSelector>>& selectors,
            const TIndexed& values) { sb::outer_assign(self, selectors, values); })
        .def("index_assign", [](TTensor& self, size_t axis,
            const sb::Tensor<sb::int64_t>& indices, const TIndexed& values) {
          sb::index_assign(self, axis, indices, values);
        })
        .def("_index_elements", [](const TTensor& self,
            const typename TTensor::index_tensor_type& indices,
            bool exactLeadingDimensions) {
          const auto alignment = exactLeadingDimensions
            ? sb::IndexedTensorAlignment::ExactLeadingDimensions
            : sb::IndexedTensorAlignment::Broadcast;
          if constexpr (TTensor::IsIndexed)
          {
            sb::validate_indexed_tensor_shape(self, indices, alignment);
            return sb::make_indexed_tensor(
              self.materialize(), indices, alignment);
          }
          else
          {
            return sb::make_indexed_tensor(self, indices, alignment);
          }
        }, pybind11::arg("indices"),
          pybind11::arg("exact_leading_dimensions") = false);
    }

    using ScalarT = scalar_of_t<T>;
    if constexpr (std::is_constructible_v<T, sb::Tensor<ScalarT>>)
    {
      cls.def("_set_element", [](TTensor& self,
          const std::vector<size_t>& index, const sb::Tensor<ScalarT>& value) {
        assert_valid_index(self, index);
        writable_element(self, index) = sb::detail::store_copy(T(value));
      });
      cls.def("masked_fill", [](TTensor& self, const sb::Tensor<bool>& mask,
          const sb::Tensor<ScalarT>& value) {
        sb::masked_fill(self, mask, T(value));
      });
      cls.def("axis_fill", [](TTensor& self, size_t axis,
          const sb::Tensor<bool>& mask, const sb::Tensor<ScalarT>& value) {
        sb::axis_fill(self, axis, mask, T(value));
      });
      cls.def("multi_axis_fill", [](TTensor& self,
          const std::vector<std::pair<size_t, sb::Tensor<bool>>>& masks,
          const sb::Tensor<ScalarT>& value) {
        sb::multi_axis_fill(self, masks, T(value));
      });
      cls.def("outer_fill", [](TTensor& self,
          const std::vector<std::pair<size_t, sb::AxisSelector>>& selectors,
          const sb::Tensor<ScalarT>& value) {
        sb::outer_fill(self, selectors, T(value));
      });
      cls.def("index_fill", [](TTensor& self, size_t axis,
          const sb::Tensor<sb::int64_t>& indices,
          const sb::Tensor<ScalarT>& value) {
        sb::index_fill(self, axis, indices, T(value));
      });
    }

  }

  template <typename T,
    sb::TensorProperties Properties = sb::TensorProperty::None>
  void register_typed_tensor_bindings(pybind11::module_& m, const std::string& prefix, const std::string& suffix)
  {
    using TTensor = sb::Tensor<T, Properties>;

    pybind11::class_<TTensor> cls = [&m, &prefix, &suffix]
    {
      if constexpr (!TTensor::IsIndexed && std::is_trivially_copyable_v<T>)
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

    bind_tensor_view_operations(cls);
    bind_tensor_operations<T, Properties>(cls);

    if constexpr (TTensor::IsIndexed)
    {
      cls.def("materialize", &TTensor::materialize);
    }
    else
    {
      cls
        .def(pybind11::init([](const Shape& shape)
        {
          return TTensor(shape.data);
        }))
        .def(pybind11::init([](const Shape& shape, const T& init)
        {
          return TTensor(shape.data, init);
        }));
    }

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

  template <sb::IndexableTensorElement T>
  void register_indexable_tensor_bindings(
      pybind11::module_& m, const std::string& name,
      const std::string& suffix = "")
  {
    register_typed_tensor_bindings<T>(m, name, suffix);
    register_typed_tensor_bindings<T, sb::TensorProperty::Indexed>(
      m, "_Indexed" + name, suffix);
  }

}

#endif //STABLEBEAR_PY_TENSOR_H
