#include <sbear/tensor.hpp>
#include <sbear/functional/pcf.hpp>

#include "py_make_from_serial_content.hpp"
#include "../py_tensor.hpp"
#include "../py_np_support.hpp"

#include <nanobind/ndarray.h>

#include <iostream>

namespace nb = nanobind;

namespace sb_py
{

  namespace detail
  {
    using EnumerationDt = long long int;
  }


  template <typename Tt, typename Tv>
  sb::Tensor<sb::Pcf<Tt, Tv>>
  make_from_serial_content(sb_py::NpIn<Tt> content, sb_py::NpIn<detail::EnumerationDt> enumeration)
  {
    using PcfT = sb::Pcf<Tt, Tv>;
    using PointT = typename PcfT::point_type;
    using TensorT = sb::Tensor<PcfT>;

    if (content.ndim() != 2)
    {
      throw std::runtime_error("content should have 2 dimensions (content has shape " + shape_to_string(content) + ").");
    }

    if (enumeration.ndim() < 2)
    {
      throw std::runtime_error("enumeration must have at least 2 dimensions");
    }

    std::vector<size_t> targetShape(enumeration.ndim() - 1);
    for (size_t i = 0; i < enumeration.ndim() - 1; ++i) // Last dim is always 2 for [start, end)
    {
      targetShape[i] = enumeration.shape(i);
    }

    TensorT target(targetShape);

    NumpyTensor<const Tt> contentView(content);
    NumpyTensor<const detail::EnumerationDt> enumerationView(enumeration);

    sb::walk(target, [&target, &content, &contentView, &enumerationView](const std::vector<size_t>& idx) {

      auto enumerationIdx = idx;
      enumerationIdx.push_back(0);
      auto start = enumerationView(enumerationIdx);
      enumerationIdx.back() = 1;
      auto stop = enumerationView(enumerationIdx);

      if (start >= stop)
      {
        throw std::invalid_argument("Item in index " + sb::index_to_string(idx) + " in the enumeration has start >= stop (" + std::to_string(start) + " >= " + std::to_string(stop) + ")");
      }

      auto numRows = static_cast<detail::EnumerationDt>(content.shape(0));
      if (start < 0 || stop > numRows)
      {
        throw std::invalid_argument("Item in index " + sb::index_to_string(idx) + " in the enumeration is out of range for content with " + std::to_string(numRows) + " rows (start=" + std::to_string(start) + ", stop=" + std::to_string(stop) + "); requires 0 <= start < stop <= " + std::to_string(numRows) + ".");
      }

      std::vector<PointT> pts;
      pts.reserve(stop - start);

      for (auto i = start; i < stop; ++i)
      {
        auto t = contentView(static_cast<size_t>(i), size_t{0});
        auto v = contentView(static_cast<size_t>(i), size_t{1});
        pts.emplace_back(t, v);
      }

      target(idx) = PcfT(std::move(pts));
    });

    return target;
  }

  void register_make_from_serial_content(nb::module_& m)
  {
    m.def("make_from_serial_content_32", &make_from_serial_content<sb::float32_t, sb::float32_t>);
    m.def("make_from_serial_content_64", &make_from_serial_content<sb::float64_t, sb::float64_t>);
  }

}