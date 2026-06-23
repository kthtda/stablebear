#include <sbear/tensor.hpp>
#include <sbear/functional/pcf.hpp>

#include "py_make_from_serial_content.hpp"
#include "../py_tensor.hpp"
#include "../py_np_support.hpp"

#include <pybind11/numpy.h>

#include <iostream>

namespace py = pybind11;

namespace sb_py
{

  namespace detail
  {
    using EnumerationDt = long long int;
  }


  template <typename Tt, typename Tv>
  sb::Tensor<sb::Pcf<Tt, Tv>>
  make_from_serial_content(py::array_t<Tt> content, py::array_t<detail::EnumerationDt> enumeration)
  {
    using PcfT = sb::Pcf<Tt, Tv>;
    using PointT = typename PcfT::point_type;
    using TensorT = sb::Tensor<PcfT>;

    auto content_buf = content.request();
    auto enumeration_buf = enumeration.request();


    if (content_buf.ndim != 2)
    {
      throw std::runtime_error("content should have 2 dimensions (content has shape " + shape_to_string(content) + ").");
    }

    if (enumeration_buf.ndim < 2)
    {
      throw std::runtime_error("enumeration must have at least 2 dimensions");
    }

    std::vector<size_t> targetShape(enumeration_buf.ndim - 1);
    for (auto i = 0; i < enumeration_buf.ndim - 1; ++i) // Last dim is always 2 for [start, end)
    {
      targetShape[i] = enumeration.shape(i);
    }

    TensorT target(targetShape);

    sb::walk(target, [&target, &content, &enumeration](const std::vector<size_t>& idx) {

      auto enumerationBaseOffset = std::inner_product(idx.begin(), idx.end(), enumeration.strides(), 0_uz);
      enumerationBaseOffset /= enumeration.itemsize();

      auto* enumerationBuf = enumeration.unchecked().data();

      auto lastStride = enumeration.strides(enumeration.ndim() - 1) / enumeration.itemsize();

      auto start = *(enumerationBuf + enumerationBaseOffset);
      auto stop = *(enumerationBuf + enumerationBaseOffset + lastStride);

      if (start >= stop)
      {
        throw py::value_error("Item in index " + sb::index_to_string(idx) + " in the enumeration has start >= stop (" + std::to_string(start) + " >= " + std::to_string(stop) + ")");
      }

      auto numRows = static_cast<detail::EnumerationDt>(content.shape(0));
      if (start < 0 || stop > numRows)
      {
        throw py::value_error("Item in index " + sb::index_to_string(idx) + " in the enumeration is out of range for content with " + std::to_string(numRows) + " rows (start=" + std::to_string(start) + ", stop=" + std::to_string(stop) + "); requires 0 <= start < stop <= " + std::to_string(numRows) + ".");
      }

      std::vector<PointT> pts;
      pts.reserve(stop - start);

      for (auto i = start; i < stop; ++i)
      {
        auto t = get_element(content, { static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(0) });
        auto v = get_element(content, { static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(1) });
        pts.emplace_back(t, v);
      }

      target(idx) = PcfT(std::move(pts));
    });

    return target;
  }

  void register_make_from_serial_content(py::module_& m)
  {
    m.def("make_from_serial_content_32", &make_from_serial_content<sb::float32_t, sb::float32_t>);
    m.def("make_from_serial_content_64", &make_from_serial_content<sb::float64_t, sb::float64_t>);
  }

}