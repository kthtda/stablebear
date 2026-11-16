#include <sbear/algorithms/argsort_segments.hpp>
#include <sbear/executor.hpp>
#include <sbear/tensor.hpp>
#include <sbear/walk.hpp>
#include <sbear/functional/pcf.hpp>
#include <sbear/algorithms/functional/iterate_rectangles.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/point_cloud.hpp>

#include "py_make_from_serial_content.hpp"
#include "../py_tensor.hpp"
#include "../py_np_support.hpp"

#include <nanobind/ndarray.h>

#include <taskflow/algorithm/for_each.hpp>

#include <span>
#include <vector>

#include <iostream>

namespace nb = nanobind;

namespace sb_py
{

  namespace detail
  {
    using EnumerationDt = long long int;
  }


  template <typename Tt>
  void require_columns(const sb_py::NpIn<Tt>& content, size_t columns)
  {
    if (content.ndim() == 2 && content.shape(1) != columns)
    {
      throw std::runtime_error("content should have " + std::to_string(columns) + " columns (content has shape " + shape_to_string(content) + ").");
    }
  }

  /// A tensor of shape enumeration.shape[:-1] whose element at idx is
  /// build(count, at), where at(i, j) gives column j of row i of the count
  /// rows of content from enumeration[idx, 0] to enumeration[idx, 1].
  /// Unless allowEmpty, each element needs a row. The elements are built
  /// in parallel.
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

    auto range = [&enumerationView](const std::vector<size_t>& idx) {
      auto enumerationIdx = idx;
      enumerationIdx.push_back(0);
      auto start = enumerationView(enumerationIdx);
      enumerationIdx.back() = 1;
      return std::pair(start, enumerationView(enumerationIdx));
    };

    // Check the ranges first, so that the parallel pass cannot throw.
    const auto numRows = static_cast<detail::EnumerationDt>(content.shape(0));
    sb::walk(target, [&](const std::vector<size_t>& idx) {
      auto [start, stop] = range(idx);

      if (start > stop || (start == stop && !allowEmpty))
      {
        throw std::invalid_argument("Item in index " + sb::index_to_string(idx) + " in the enumeration has start " + (allowEmpty ? "> " : ">= ") + "stop (" + std::to_string(start) + ", " + std::to_string(stop) + ")");
      }

      if (start < 0 || stop > numRows)
      {
        throw std::invalid_argument("Item in index " + sb::index_to_string(idx) + " in the enumeration is out of range for content with " + std::to_string(numRows) + " rows (start=" + std::to_string(start) + ", stop=" + std::to_string(stop) + "); requires 0 <= start < stop <= " + std::to_string(numRows) + ".");
      }
    });

    nb::gil_scoped_release release;
    sb::parallel_walk(target, [&](const std::vector<size_t>& idx) {
      auto [start, stop] = range(idx);
      target(idx) = build(static_cast<size_t>(stop - start), [&contentView, start](size_t i, size_t j) {
        return contentView(static_cast<size_t>(start) + i, j);
      });
    }, sb::default_executor());

    return target;
  }

  template <typename Tt, typename Tv>
  sb::Tensor<sb::Pcf<Tt, Tv>>
  make_from_serial_content(sb_py::NpIn<Tt> content, sb_py::NpIn<detail::EnumerationDt> enumeration)
  {
    using PcfT = sb::Pcf<Tt, Tv>;
    require_columns(content, 2);
    return tensor_from_serial_content<PcfT>(content, enumeration, false, [](size_t count, const auto& at) {
      std::vector<typename PcfT::point_type> pts;
      pts.reserve(count);
      for (size_t i = 0; i < count; ++i)
      {
        pts.emplace_back(at(i, 0), static_cast<Tv>(at(i, 1)));
      }
      return PcfT(std::move(pts));
    });
  }

  template <typename T>
  sb::Tensor<sb::ph::Barcode<T>>
  barcodes_from_serial_content(sb_py::NpIn<T> content, sb_py::NpIn<detail::EnumerationDt> enumeration)
  {
    require_columns(content, 2);
    return tensor_from_serial_content<sb::ph::Barcode<T>>(content, enumeration, true, [](size_t count, const auto& at) {
      std::vector<sb::ph::PersistencePair<T>> bars;
      bars.reserve(count);
      for (size_t i = 0; i < count; ++i)
      {
        bars.emplace_back(at(i, 0), at(i, 1));
      }
      return sb::ph::Barcode<T>(std::move(bars));
    });
  }

  /// Point clouds whose points are the rows of content: its columns are
  /// the coordinates.
  template <typename T>
  sb::Tensor<sb::PointCloud<T>>
  point_clouds_from_serial_content(sb_py::NpIn<T> content, sb_py::NpIn<detail::EnumerationDt> enumeration)
  {
    const size_t dim = content.ndim() == 2 ? content.shape(1) : 0;
    return tensor_from_serial_content<sb::PointCloud<T>>(content, enumeration, true, [dim](size_t count, const auto& at) {
      sb::PointCloud<T> cloud(std::vector<size_t>{count, dim});
      for (size_t i = 0; i < count; ++i)
      {
        for (size_t j = 0; j < dim; ++j)
        {
          cloud(i, j) = at(i, j);
        }
      }
      return cloud;
    });
  }

  /// The inverse of tensor_from_serial_content: (content, enumeration) with
  /// the rows of the elements concatenated in C order of the tensor's
  /// elements. size(element) is its number of rows, and write(element, out)
  /// writes them, each of the given width, to out. The rows are written in
  /// parallel.
  template <typename Tt, typename Elem, typename Size, typename Write>
  nb::tuple tensor_to_serial_content(const sb::Tensor<Elem>& tensor, Size size, Write write, size_t width = 2)
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
    sb::walk(tensor, [&](const std::vector<size_t>& idx) { sizes[flat(idx)] = size(tensor(idx)); });

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

    auto content = new_array<Tt>({ total, width });
    auto* c = content.data();
    {
      nb::gil_scoped_release release;
      sb::parallel_walk(tensor, [&](const std::vector<size_t>& idx) {
        write(tensor(idx), c + width * static_cast<size_t>(e[2 * flat(idx)]));
      }, sb::default_executor());
    }

    return nb::make_tuple(content, enumeration);
  }

  template <typename Tt, typename Tv>
  nb::tuple to_serial_content(const sb::Tensor<sb::Pcf<Tt, Tv>>& tensor)
  {
    return tensor_to_serial_content<Tt>(tensor,
      [](const sb::Pcf<Tt, Tv>& f) { return f.points().size(); },
      [](const sb::Pcf<Tt, Tv>& f, Tt* out) {
        for (const auto& pt : f.points())
        {
          *out++ = pt.t;
          *out++ = static_cast<Tt>(pt.v);
        }
      });
  }

  template <typename T>
  nb::tuple barcodes_to_serial_content(const sb::Tensor<sb::ph::Barcode<T>>& tensor)
  {
    return tensor_to_serial_content<T>(tensor,
      [](const sb::ph::Barcode<T>& bc) { return bc.bars().size(); },
      [](const sb::ph::Barcode<T>& bc, T* out) {
        for (const auto& bar : bc.bars())
        {
          *out++ = bar.birth;
          *out++ = bar.death;
        }
      });
  }

  /// The points of all clouds as the rows of the content. The clouds that
  /// have points must have the same dimension.
  template <typename T>
  nb::tuple point_clouds_to_serial_content(const sb::Tensor<sb::PointCloud<T>>& tensor)
  {
    size_t dim = 0;
    sb::walk(tensor, [&](const std::vector<size_t>& idx) {
      const auto& cloud = tensor(idx);
      if (cloud.n_points() == 0)
      {
        return;
      }
      if (dim != 0 && cloud.dim() != dim)
      {
        throw std::invalid_argument("The point clouds must have the same dimension (got " + std::to_string(dim) + " and " + std::to_string(cloud.dim()) + ").");
      }
      dim = cloud.dim();
    });

    return tensor_to_serial_content<T>(tensor,
      [](const sb::PointCloud<T>& cloud) { return cloud.n_points(); },
      [dim](const sb::PointCloud<T>& cloud, T* out) {
        for (size_t i = 0; i < cloud.n_points(); ++i)
        {
          for (size_t j = 0; j < dim; ++j)
          {
            *out++ = cloud(i, j);
          }
        }
      }, dim);
  }

  /// The union of the breakpoint times of two tensors of PCFs of the same
  /// shape, per element, from the rectangles of each pair, in parallel.
  /// Returns (times, sizes, fRows, gRows, fromG): the times of all elements,
  /// concatenated in C order; the number of times of each element; and for
  /// each time the positions, within their PCFs, of the breakpoints of f and
  /// of g that hold at it, and whether the time is that of g's breakpoint
  /// and not of f's.
  template <typename Tt, typename Tv>
  nb::tuple merge_breakpoints(const sb::Tensor<sb::Pcf<Tt, Tv>>& f, const sb::Tensor<sb::Pcf<Tt, Tv>>& g)
  {
    using RectangleT = sb::Rectangle<Tt, Tv>;

    if (f.shape() != g.shape())
    {
      throw std::invalid_argument("merge_breakpoints: the tensors must have the same shape");
    }

    const auto& shape = f.shape();
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
    // Calls emit(rect, fRow, gRow) for each rectangle of nonzero width.
    // Equal times within one PCF, as in a stable rank with equal lifetimes,
    // give a rectangle of zero width, which is not a time of the union.
    auto for_each_time = [&](const std::vector<size_t>& idx, auto&& emit) {
      sb::iterate_rectangles_with_rows(f(idx).points(), g(idx).points(),
        [&emit](const RectangleT& rect, size_t fRow, size_t gRow) {
          if (rect.right > rect.left)
          {
            emit(rect, fRow, gRow);
          }
        });
    };

    const size_t n = f.size();
    auto sizes = new_array<int64_t>({n});
    {
      nb::gil_scoped_release release;
      sb::parallel_walk(f, [&](const std::vector<size_t>& idx) {
        int64_t count = 0;
        for_each_time(idx, [&count](const RectangleT&, size_t, size_t) { ++count; });
        sizes.data()[flat(idx)] = count;
      }, sb::default_executor());
    }

    std::vector<size_t> offsets(n + 1, 0);
    for (size_t k = 0; k < n; ++k)
    {
      offsets[k + 1] = offsets[k] + static_cast<size_t>(sizes.data()[k]);
    }

    auto times = new_array<Tt>({offsets[n]});
    auto fRows = new_array<int64_t>({offsets[n]});
    auto gRows = new_array<int64_t>({offsets[n]});
    auto fromG = new_array<bool>({offsets[n]});
    {
      nb::gil_scoped_release release;
      sb::parallel_walk(f, [&](const std::vector<size_t>& idx) {
        const auto& fpts = f(idx).points();
        size_t out = offsets[flat(idx)];
        for_each_time(idx, [&](const RectangleT& rect, size_t fRow, size_t gRow) {
          times.data()[out] = rect.left;
          fRows.data()[out] = static_cast<int64_t>(fRow);
          gRows.data()[out] = static_cast<int64_t>(gRow);
          fromG.data()[out] = fpts[fRow].t != rect.left;
          ++out;
        });
      }, sb::default_executor());
    }

    return nb::make_tuple(times, sizes, fRows, gRows, fromG);
  }

  /// The order that sorts the values of each element of serial content:
  /// element k has the values in rows enumeration[k, 0] to enumeration[k, 1]
  /// of @p values. Returns, for all elements concatenated, the rows of each
  /// element arranged by increasing value, equal values keeping their
  /// order. The elements are sorted in parallel.
  template <typename T>
  NpArray<int64_t> argsort_segments(
      nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu> values,
      nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> enumeration)
  {
    const size_t n = enumeration.shape(0);
    const size_t total = values.shape(0);
    if (enumeration.shape(1) != 2)
    {
      throw std::invalid_argument("argsort_segments: enumeration must have shape (n, 2)");
    }
    const auto* e = enumeration.data();
    for (size_t k = 0; k < n; ++k)
    {
      // The elements must tile the values, so that each row is written once.
      const auto expectedStart = static_cast<int64_t>(k == 0 ? 0 : e[2 * k - 1]);
      if (e[2 * k] != expectedStart || e[2 * k + 1] < e[2 * k])
      {
        throw std::invalid_argument("argsort_segments: the enumeration must list consecutive rows");
      }
    }
    if ((n == 0 ? size_t{0} : static_cast<size_t>(e[2 * n - 1])) != total)
    {
      throw std::invalid_argument("argsort_segments: the enumeration must cover all values");
    }

    auto order = new_array<int64_t>({total});
    {
      nb::gil_scoped_release release;
      tf::Taskflow flow;
      flow.for_each_index(size_t{0}, n, size_t{1}, [&](size_t k) {
        const auto start = e[2 * k];
        const auto count = static_cast<size_t>(e[2 * k + 1] - start);
        sb::argsort_segment(std::span<const T>(values.data() + start, count), start,
                            std::span<int64_t>(order.data() + start, count));
      });
      sb::default_executor().cpu()->run(flow).get();
    }
    return order;
  }

  void register_make_from_serial_content(nb::module_& m)
  {
    m.def("to_serial_content", &to_serial_content<sb::float32_t, sb::float32_t>);
    m.def("to_serial_content", &to_serial_content<sb::float64_t, sb::float64_t>);
    m.def("make_from_serial_content_32", &make_from_serial_content<sb::float32_t, sb::float32_t>);
    m.def("make_from_serial_content_64", &make_from_serial_content<sb::float64_t, sb::float64_t>);
    m.def("argsort_segments", &argsort_segments<sb::float32_t>);
    m.def("argsort_segments", &argsort_segments<sb::float64_t>);
    m.def("merge_breakpoints", &merge_breakpoints<sb::float32_t, sb::float32_t>);
    m.def("merge_breakpoints", &merge_breakpoints<sb::float64_t, sb::float64_t>);
    m.def("barcodes_to_serial_content", &barcodes_to_serial_content<sb::float32_t>);
    m.def("barcodes_to_serial_content", &barcodes_to_serial_content<sb::float64_t>);
    m.def("barcodes_from_serial_content_32", &barcodes_from_serial_content<sb::float32_t>);
    m.def("barcodes_from_serial_content_64", &barcodes_from_serial_content<sb::float64_t>);
    m.def("point_clouds_to_serial_content", &point_clouds_to_serial_content<sb::float32_t>);
    m.def("point_clouds_to_serial_content", &point_clouds_to_serial_content<sb::float64_t>);
    m.def("point_clouds_from_serial_content_32", &point_clouds_from_serial_content<sb::float32_t>);
    m.def("point_clouds_from_serial_content_64", &point_clouds_from_serial_content<sb::float64_t>);
  }

}