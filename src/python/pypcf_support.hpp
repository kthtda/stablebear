#ifndef SB_PYPCF_SUPPORT_H
#define SB_PYPCF_SUPPORT_H

#include <sbear/functional/pcf.hpp>
#include <sbear/walk.hpp>

#include <algorithm>
#include <numeric>
#include <sstream>
#include <vector>

#include "nb.hpp"
#include "py_np_support.hpp"


namespace nb = nanobind;

namespace sb
{
  namespace detail
  {
    template <typename T>
    std::string shape_to_string(T shape)
    {
      std::stringstream ss;
      ss << "(";
      for (auto it = std::begin(shape); it != std::end(shape); ++it)
      {
        if (it != std::begin(shape))
        {
          ss << ", ";
        }
        ss << *it;
      }
      ss << ")";
      return ss.str();
    }

    template <typename Tt, typename Tv>
    sb::Pcf<Tt, Tv> construct_pcf(sb_py::NpIn<Tt> arr)
    {
      using point_type = typename sb::Pcf<Tt, Tv>::point_type;

      std::vector<sb::TimePoint<Tt, Tv>> points;

      if (arr.size() == 0)
      {
        points.emplace_back(static_cast<Tt>(0), static_cast<Tv>(0));
        return sb::Pcf<Tt, Tv>(std::move(points));
      }

      if (arr.ndim() != 2)
      {
        throw std::runtime_error("Input array should have two dimensions (time + value).");
      }

      if (arr.shape(1) == 2)
      {
        NumpyTensor<const Tt> data(arr);
        points.resize(arr.shape(0));
        for (size_t i = 0; i < arr.shape(0); ++i)
        {
          points[i].t = data(i, 0);
          points[i].v = data(i, 1);
        }
      }
      else
      {
        throw std::runtime_error("Input array should be Nx2 (supplied shape is " + ::shape_to_string(arr) + ").");
      }

      // Breakpoint times must be strictly increasing. Rather than silently
      // sorting (which can reorder rows in a way that hides a misplaced or
      // negative time), reject input that is not already ordered.
      auto notIncreasing = [](const point_type& a, const point_type& b) {
        return a.t >= b.t;
      };
      if (std::adjacent_find(points.begin(), points.end(), notIncreasing) != points.end())
      {
        throw std::invalid_argument(
          "Breakpoint times must be strictly increasing.");
      }

      // PCFs are defined on [0, inf), so the first (smallest) breakpoint time
      // must be 0; any negative time (or a missing t=0) is rejected here.
      if (points.front().t != static_cast<Tt>(0))
      {
        throw std::invalid_argument(
          "The first breakpoint must have time t=0 (got t="
          + std::to_string(static_cast<double>(points.front().t)) + ").");
      }

      return sb::Pcf<Tt, Tv>(std::move(points));
    }

    /// Evaluate `f` at each of `times`, giving a C-ordered array of the same
    /// shape.
    template <typename TFunction>
    auto evaluate_at(const TFunction& f, sb_py::NpIn<typename TFunction::time_type> times)
    {
      using Tt = typename TFunction::time_type;
      using Tv = typename TFunction::value_type;

      // Read the times in row-major order, whatever their strides
      NumpyTensor<const Tt> in(times);
      std::vector<Tt> flat_in;
      flat_in.reserve(in.size());
      sb::walk(in, [&](const std::vector<size_t>& idx) { flat_in.push_back(in(idx)); });
      auto n = flat_in.size();

      // Argsort
      std::vector<size_t> order(n);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&flat_in](size_t a, size_t b) {
        return flat_in[a] < flat_in[b];
      });

      // Build sorted times and evaluate
      std::vector<Tt> sorted_in(n);
      for (size_t i = 0; i < n; ++i)
      {
        sorted_in[i] = flat_in[order[i]];
      }

      std::vector<Tv> sorted_out(n);
      f.evaluate(sorted_in, sorted_out);

      // Unsort results back to original order
      auto result = sb_py::new_array<Tv>(in.shape());
      Tv* out = result.data();
      for (size_t i = 0; i < n; ++i)
      {
        out[order[i]] = sorted_out[i];
      }

      return result;
    }

    /// A read-only (n, 2) view of the breakpoints of `pcf`, kept alive by
    /// the Python object `owner`.
    template <typename TPcf>
    auto to_numpy(const TPcf& pcf, nb::handle owner)
    {
      using TTime = typename TPcf::time_type;
      using TVal = typename TPcf::value_type;
      static_assert(std::is_same<TTime, TVal>::value, "time and value type must be the same");

      return nb::ndarray<nb::numpy, const TVal>(
        reinterpret_cast<const TVal*>(pcf.points().data()),
        { pcf.points().size(), size_t{2} }, owner);
    }

    /// Bind what Pcf and Plf share: construction with no breakpoints or from
    /// an (n, 2) array, which `construct` turns into a function, the array
    /// view, size, and evaluation at a scalar or an array.
    template <typename Cls, typename Construct>
    void bind_piecewise_function(Cls& cls, Construct construct)
    {
      using TFunction = typename Cls::Type;
      using Tt = typename TFunction::time_type;
      using Tv = typename TFunction::value_type;

      sb_py::bind_array_view(cls, [](const TFunction& self, nb::handle owner) {
        return to_numpy(self, owner);
      });
      cls
        .def(nb::init<>())
        .def("__init__", [construct](TFunction* self, sb_py::NpIn<Tt> arr) {
          new (self) TFunction(construct(arr));
        })
        .def("size", [](const TFunction& self) { return self.points().size(); })
        .def("__call__", [](const TFunction& self, Tt t) -> Tv { return self.evaluate(t); })
        .def("__call__", [](const TFunction& self, sb_py::NpIn<Tt> times) {
          return evaluate_at(self, times);
        });
    }
  }
}

#endif
