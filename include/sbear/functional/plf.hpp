#ifndef SB_PLF_H
#define SB_PLF_H

#include "piecewise_function.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sb
{
  /**
   * A piecewise linear function on [0, inf).
   *
   * The function is given by breakpoints (t_0, v_0), ..., (t_n, v_n) with
   * t_0 = 0 and strictly increasing times. It interpolates linearly between
   * consecutive breakpoints and, after t_n, continues its last segment. A
   * single breakpoint gives a constant function.
   */
  template <FloatType Tt, FloatType Tv = Tt>
  class Plf
  {
  public:
    using point_type = TimePoint<Tt, Tv>;
    using time_type = typename point_type::time_type;
    using value_type = typename point_type::value_type;

    /// The zero function.
    Plf()
    {
      m_points.emplace_back(static_cast<Tt>(0), static_cast<Tv>(0));
    }

    /// Throws std::invalid_argument unless the breakpoints are valid.
    explicit Plf(std::vector<point_type>&& pts)
      : m_points(std::move(pts))
    {
      validate();
    }

    Plf(std::initializer_list<std::pair<Tt, Tv>> pts)
    {
      m_points.reserve(pts.size());
      for (auto const& pt : pts)
      {
        m_points.emplace_back(pt.first, pt.second);
      }
      validate();
    }

    bool operator==(const Plf& rhs) const
    {
      return m_points == rhs.m_points;
    }

    bool operator!=(const Plf& rhs) const
    {
      return m_points != rhs.m_points;
    }

    [[nodiscard]] value_type evaluate(time_type t) const
    {
      if (t < static_cast<time_type>(0))
      {
        throw std::domain_error("Cannot evaluate PLF before time 0.");
      }

      auto it = std::upper_bound(m_points.begin(), m_points.end(), t,
        [](time_type time, const point_type& pt) { return time < pt.t; });

      return value_after(static_cast<size_t>(std::distance(m_points.begin(), it)) - 1, t);
    }

    /// Evaluate at sorted times in one pass, writing out[k] for each
    /// sorted_times[k]. out must be at least as long as sorted_times.
    void evaluate(std::span<const time_type> sorted_times, std::span<value_type> out) const
    {
      size_t i = 0;

      for (size_t k = 0; k < sorted_times.size(); ++k)
      {
        auto t = sorted_times[k];

        if (t < static_cast<time_type>(0))
        {
          throw std::domain_error("Cannot evaluate PLF before time 0.");
        }

        while (i + 1 < m_points.size() && m_points[i + 1].t <= t)
        {
          ++i;
        }

        out[k] = value_after(i, t);
      }
    }

    [[nodiscard]] const std::vector<point_type>& points() const { return m_points; }
    [[nodiscard]] size_t size() const noexcept { return m_points.size(); }

  private:
    /// The value at t, where breakpoint i is the last one at or before t.
    [[nodiscard]] value_type value_after(size_t i, time_type t) const
    {
      const auto& p = m_points[i];
      if (i + 1 < m_points.size())
      {
        const auto& q = m_points[i + 1];
        return p.v + (q.v - p.v) * static_cast<value_type>((t - p.t) / (q.t - p.t));
      }
      if (i == 0)
      {
        return p.v;
      }

      // Continue the last segment from the last breakpoint, so that the value
      // there is exactly v_n.
      const auto& q = m_points[i - 1];
      return p.v + (p.v - q.v) * static_cast<value_type>((t - p.t) / (p.t - q.t));
    }

    void validate() const
    {
      if (m_points.empty())
      {
        throw std::invalid_argument("A PLF needs at least one breakpoint.");
      }
      for (const auto& pt : m_points)
      {
        if (!std::isfinite(pt.t))
        {
          throw std::invalid_argument("PLF breakpoint times must be finite.");
        }
      }
      if (m_points.front().t != static_cast<Tt>(0))
      {
        throw std::invalid_argument("The first PLF breakpoint must be at time 0.");
      }
      for (size_t i = 1; i < m_points.size(); ++i)
      {
        if (m_points[i].t <= m_points[i - 1].t)
        {
          throw std::invalid_argument("PLF breakpoint times must be strictly increasing.");
        }
      }
    }

    std::vector<point_type> m_points;
  };

  using Plf_f32 = Plf<float32_t, float32_t>;
  using Plf_f64 = Plf<float64_t, float64_t>;
}

#endif
