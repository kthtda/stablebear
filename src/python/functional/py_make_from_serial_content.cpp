#include <sbear/tensor.hpp>
#include <sbear/functional/pcf.hpp>
#include <sbear/persistence/barcode.hpp>

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


  /// A tensor of shape enumeration.shape[:-1] whose element at idx is
  /// build(rows), for the rows (a, b) of content from enumeration[idx, 0] to
  /// enumeration[idx, 1]. Unless allowEmpty, each element needs a row.
  template <typename Elem, typename Tt, typename Build>
  sb::Tensor<Elem> tensor_from_serial_content(
      sb_py::NpIn<Tt> content, sb_py::NpIn<detail::EnumerationDt> enumeration, bool allowEmpty, Build build)
  {
    using TensorT = sb::Tensor<Elem>;

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

    sb::walk(target, [&](const std::vector<size_t>& idx) {

      auto enumerationIdx = idx;
      enumerationIdx.push_back(0);
      auto start = enumerationView(enumerationIdx);
      enumerationIdx.back() = 1;
      auto stop = enumerationView(enumerationIdx);

      if (start > stop || (start == stop && !allowEmpty))
      {
        throw std::invalid_argument("Item in index " + sb::index_to_string(idx) + " in the enumeration has start " + (allowEmpty ? "> " : ">= ") + "stop (" + std::to_string(start) + ", " + std::to_string(stop) + ")");
      }

      auto numRows = static_cast<detail::EnumerationDt>(content.shape(0));
      if (start < 0 || stop > numRows)
      {
        throw std::invalid_argument("Item in index " + sb::index_to_string(idx) + " in the enumeration is out of range for content with " + std::to_string(numRows) + " rows (start=" + std::to_string(start) + ", stop=" + std::to_string(stop) + "); requires 0 <= start < stop <= " + std::to_string(numRows) + ".");
      }

      std::vector<std::pair<Tt, Tt>> rows;
      rows.reserve(static_cast<size_t>(stop - start));
      for (auto i = start; i < stop; ++i)
      {
        rows.emplace_back(contentView(static_cast<size_t>(i), size_t{0}), contentView(static_cast<size_t>(i), size_t{1}));
      }

      target(idx) = build(rows);
    });

    return target;
  }

  template <typename Tt, typename Tv>
  sb::Tensor<sb::Pcf<Tt, Tv>>
  make_from_serial_content(sb_py::NpIn<Tt> content, sb_py::NpIn<detail::EnumerationDt> enumeration)
  {
    using PcfT = sb::Pcf<Tt, Tv>;
    return tensor_from_serial_content<PcfT>(content, enumeration, false, [](const auto& rows) {
      std::vector<typename PcfT::point_type> pts;
      pts.reserve(rows.size());
      for (const auto& [t, v] : rows)
      {
        pts.emplace_back(t, static_cast<Tv>(v));
      }
      return PcfT(std::move(pts));
    });
  }

  template <typename T>
  sb::Tensor<sb::ph::Barcode<T>>
  barcodes_from_serial_content(sb_py::NpIn<T> content, sb_py::NpIn<detail::EnumerationDt> enumeration)
  {
    return tensor_from_serial_content<sb::ph::Barcode<T>>(content, enumeration, true, [](const auto& rows) {
      std::vector<sb::ph::PersistencePair<T>> bars;
      bars.reserve(rows.size());
      for (const auto& [birth, death] : rows)
      {
        bars.emplace_back(birth, death);
      }
      return sb::ph::Barcode<T>(std::move(bars));
    });
  }

  /// The inverse of tensor_from_serial_content: (content, enumeration) with
  /// the rows of the elements, rows(element) as (a, b) pairs, concatenated in
  /// C order of the tensor's elements.
  template <typename Tt, typename Elem, typename Rows>
  nb::tuple tensor_to_serial_content(const sb::Tensor<Elem>& tensor, Rows rows)
  {
    const auto& shape = tensor.shape();
    std::vector<size_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i-- > 1;)
    {
      strides[i - 1] = strides[i] * shape[i];
    }
    auto flat = [&strides](const std::vector<size_t>& idx) {
      size_t k = 0;
      for (size_t i = 0; i < idx.size(); ++i)
      {
        k += idx[i] * strides[i];
      }
      return k;
    };

    std::vector<size_t> sizes(tensor.size());
    sb::walk(tensor, [&](const std::vector<size_t>& idx) { sizes[flat(idx)] = rows(tensor(idx)).size(); });

    std::vector<size_t> enumerationShape(shape.begin(), shape.end());
    enumerationShape.push_back(2);
    auto enumeration = new_array<detail::EnumerationDt>(enumerationShape);
    auto* e = enumeration.data();
    size_t total = 0;
    for (size_t k = 0; k < sizes.size(); ++k)
    {
      e[2 * k] = static_cast<detail::EnumerationDt>(total);
      total += sizes[k];
      e[2 * k + 1] = static_cast<detail::EnumerationDt>(total);
    }

    auto content = new_array<Tt>({ total, 2 });
    auto* c = content.data();
    sb::walk(tensor, [&](const std::vector<size_t>& idx) {
      auto row = static_cast<size_t>(e[2 * flat(idx)]);
      for (const auto& [a, b] : rows(tensor(idx)))
      {
        c[2 * row] = a;
        c[2 * row + 1] = b;
        ++row;
      }
    });

    return nb::make_tuple(content, enumeration);
  }

  template <typename Tt, typename Tv>
  nb::tuple to_serial_content(const sb::Tensor<sb::Pcf<Tt, Tv>>& tensor)
  {
    return tensor_to_serial_content<Tt>(tensor, [](const sb::Pcf<Tt, Tv>& f) {
      std::vector<std::pair<Tt, Tt>> rows;
      rows.reserve(f.points().size());
      for (const auto& pt : f.points())
      {
        rows.emplace_back(pt.t, static_cast<Tt>(pt.v));
      }
      return rows;
    });
  }

  template <typename T>
  nb::tuple barcodes_to_serial_content(const sb::Tensor<sb::ph::Barcode<T>>& tensor)
  {
    return tensor_to_serial_content<T>(tensor, [](const sb::ph::Barcode<T>& bc) {
      std::vector<std::pair<T, T>> rows;
      rows.reserve(bc.bars().size());
      for (const auto& bar : bc.bars())
      {
        rows.emplace_back(bar.birth, bar.death);
      }
      return rows;
    });
  }

  void register_make_from_serial_content(nb::module_& m)
  {
    m.def("to_serial_content", &to_serial_content<sb::float32_t, sb::float32_t>);
    m.def("to_serial_content", &to_serial_content<sb::float64_t, sb::float64_t>);
    m.def("make_from_serial_content_32", &make_from_serial_content<sb::float32_t, sb::float32_t>);
    m.def("make_from_serial_content_64", &make_from_serial_content<sb::float64_t, sb::float64_t>);
    m.def("barcodes_to_serial_content", &barcodes_to_serial_content<sb::float32_t>);
    m.def("barcodes_to_serial_content", &barcodes_to_serial_content<sb::float64_t>);
    m.def("barcodes_from_serial_content_32", &barcodes_from_serial_content<sb::float32_t>);
    m.def("barcodes_from_serial_content_64", &barcodes_from_serial_content<sb::float64_t>);
  }

}