#ifndef SB_PYPCF_SUPPORT_H
#define SB_PYPCF_SUPPORT_H

#include <sbear/functional/pcf.hpp>

#include <sstream>

#include "pybind.hpp"


namespace py = pybind11;

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
    sb::Pcf<Tt, Tv> construct_pcf(py::array_t<Tt> arr)
    {
      using point_type = typename sb::Pcf<Tt, Tv>::point_type;

      std::vector<sb::TimePoint<Tt, Tv>> points;

      py::buffer_info buf = arr.request();
      if (buf.size == 0)
      {
        // A PCF must have at least one breakpoint at t=0; an empty input array
        // cannot represent one. (Use Pcf() for the all-zero constant PCF.)
        throw std::invalid_argument(
          "Cannot construct a Pcf from an empty array; a PCF must have at "
          "least one breakpoint at t=0.");
      }

      if (buf.ndim != 2)
      {
        throw std::runtime_error("Input array should have two dimensions (time + value).");
      }

      auto data = arr.template unchecked<2>();

      if (buf.shape.size() == 2 && buf.shape[1] == 2)
      {
        points.resize(buf.shape[0]);
        for (auto i = 0; i < buf.shape[0]; ++i)
        {
          points[i].t = data(i, 0);
          points[i].v = data(i, 1);
        }
      }
      else
      {
        throw std::runtime_error("Input array should be Nx2 (supplied shape is " + shape_to_string(buf.shape) + ").");
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

      // Times must be strictly increasing (t_0 < t_1 < ...): a duplicate time
      // is ambiguous (two values at the same breakpoint) and is rejected
      // rather than silently kept.
      if (std::adjacent_find(points.begin(), points.end(),
            [](const point_type& a, const point_type& b){ return a.t == b.t; })
          != points.end())
      {
        throw std::invalid_argument(
          "Breakpoints must have strictly increasing times (a duplicate time "
          "was supplied).");
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

#if 0
    template <typename TPcf>
    py::memoryview to_numpy(const TPcf& pcf)
    {
      using TTime = typename TPcf::time_type;
      using TVal = typename TPcf::value_type;
      static_assert(std::is_same<TTime, TVal>::value, "time and value type must be the same");

      return py::memoryview::from_buffer(
        reinterpret_cast<const TTime*>(pcf.points().data()), 
        { Py_ssize_t(2), Py_ssize_t(pcf.points().size())},
        { Py_ssize_t(sizeof(TTime)), Py_ssize_t(sizeof(TTime) * 2)});
    }
#endif

    template <typename TPcf>
    py::buffer_info to_numpy(const TPcf& pcf)
    {
      using TTime = typename TPcf::time_type;
      using TVal = typename TPcf::value_type;
      static_assert(std::is_same<TTime, TVal>::value, "time and value type must be the same");

      return py::buffer_info(
        const_cast<void*>(reinterpret_cast<const void*>(pcf.points().data())),
        sizeof(TVal),
        py::format_descriptor<TVal>::format(),
        py::ssize_t(2),
        { py::ssize_t(pcf.points().size()), py::ssize_t(2) },
        { py::ssize_t(2 * sizeof(TVal)), py::ssize_t(sizeof(TVal)) },
        true
      );
    }


  }
}

#endif
