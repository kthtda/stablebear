#ifndef SB_PYPCF_SUPPORT_H
#define SB_PYPCF_SUPPORT_H

#include <sbear/functional/pcf.hpp>

#include <sstream>

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

      auto sortByTime = [](const point_type& a, const point_type & b){
        return a.t < b.t;
      };

      // Breakpoints must be supplied in non-decreasing time order. Rather than
      // silently sorting (which can reorder rows in a way that hides a
      // misplaced/negative time), reject input that is not already ordered.
      if (!std::is_sorted(points.begin(), points.end(), sortByTime))
      {
        throw std::invalid_argument(
          "Breakpoints must be given in non-decreasing time order.");
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
  }
}

#endif
