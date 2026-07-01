#ifndef SB_PCF_H
#define SB_PCF_H

#include "piecewise_function.hpp"
#include "rectangle.hpp"
#include "../algorithms/functional/reduce.hpp"

#include <vector>
#include <iostream>
#include <ostream>
#include <cmath>

namespace sb
{
  template <typename Tt, typename Tv = Tt>
  class Pcf
  {
  public:
    using point_type = TimePoint<Tt, Tv>;
    using time_type = typename point_type::time_type;
    using value_type = typename point_type::value_type;

    using reduction_functional_type = value_type(time_type, time_type, value_type, value_type);
    using rectangle_type = Rectangle<Tt, Tv>;
    using segment_type = Segment<Tt, Tv>;

    Pcf()
    {
      m_points.emplace_back(static_cast<Tt>(0), static_cast<Tv>(0));
    }

    explicit Pcf(value_type v)
    {
      m_points.emplace_back(static_cast<Tt>(0), static_cast<Tv>(v));
    }

    explicit Pcf(std::vector<TimePoint<Tt, Tv>>&& pts)
      : m_points(std::move(pts))
    {

    }

    /// Converting constructor: convert from a Pcf with different precision.
    template <typename Ut, typename Uv>
    explicit Pcf(const Pcf<Ut, Uv>& other)
    {
      m_points.reserve(other.size());
      for (size_t i = 0; i < other.size(); ++i)
        m_points.emplace_back(static_cast<Tt>(other.points()[i].t),
                              static_cast<Tv>(other.points()[i].v));
    }

    Pcf(std::initializer_list<std::pair<Tt, Tv>> pts)
      requires (!FloatType<Tt> || !FloatType<Tv>)
    {
      // We can't have this constructor if we work with floating point types because of the overloads below, but we
      // still want to be able to construct from pairs if we are using non-float types.
      initFromPairs(pts);
    }

    Pcf(std::initializer_list<std::pair<float64_t, float64_t>> pts)
      requires (std::is_same_v<Tt, float64_t> && std::is_same_v<Tv, float64_t>)
    {
      initFromPairs(pts);
    }

    Pcf(std::initializer_list<std::pair<float32_t, float32_t>> pts)
      requires (std::is_same_v<Tt, float32_t> && std::is_same_v<Tv, float32_t>)
    {
      initFromPairs(pts);
    }

    std::string to_string() const
    {
      std::stringstream ss;
      for (auto i = 0ul; i < m_points.size(); ++i)
      {
        if (i != 0ul)
        {
          ss << ", ";
        }
        ss << "f(" << m_points[i].t << ") = " << m_points[i].v;
      }
      return ss.str();
    }
    
    void debug_print() const
    {
      std::cout << to_string() << std::endl;
    }

    Pcf& operator+=(const Pcf& rhs)
    {
      *this = combine(*this, rhs, [](const typename Pcf<Tt, Tv>::rectangle_type& rect) {
        return rect.f_value + rect.g_value;
        });
      return *this;
    }

    Pcf& operator-=(const Pcf& rhs)
    {
      *this = combine(*this, rhs, [](const typename Pcf<Tt, Tv>::rectangle_type& rect) {
        return rect.f_value - rect.g_value;
        });
      return *this;
    }

    Pcf& operator*=(const Pcf& rhs)
    {
      *this = combine(*this, rhs, [](const typename Pcf<Tt, Tv>::rectangle_type& rect) {
        return rect.f_value * rect.g_value;
        });
      return *this;
    }

    Pcf& operator/=(const Pcf& rhs)
    {
      *this = combine(*this, rhs, [](const typename Pcf<Tt, Tv>::rectangle_type& rect) {
        return rect.f_value / rect.g_value;
        });
      return *this;
    }

    template <Arithmetic T>
    Pcf& operator+=(T v)
    {
      for (auto & pt : m_points)
      {
        pt.v += v;
      }
      return *this;
    }

    template <Arithmetic T>
    Pcf& operator-=(T v)
    {
      for (auto & pt : m_points)
      {
        pt.v -= v;
      }
      return *this;
    }

    template <Arithmetic T>
    Pcf& operator/=(T v)
    {
      for (auto & pt : m_points)
      {
        pt.v /= v;
      }
      return *this;
    }

    template <Arithmetic T>
    Pcf& operator*=(T v)
    {
      for (auto & pt : m_points)
      {
        pt.v *= v;
      }
      return *this;
    }
    
    [[nodiscard]] Pcf operator-() const
    {
      Pcf ret = *this;
      for (auto& pt : ret.m_points)
      {
        pt.v = -pt.v;
      }
      return ret;
    }

    bool operator==(const Pcf& rhs) const
    {
      return m_points == rhs.m_points;
    }
    
    bool operator!=(const Pcf& rhs) const
    {
      return m_points != rhs.m_points;
    }

    [[nodiscard]] value_type evaluate(time_type t) const
    {
      // PCFs are defined on [0, inf); reject negative times against 0 directly
      // rather than against the first breakpoint's time.
      if (t < static_cast<time_type>(0))
      {
        throw std::domain_error("Cannot evaluate PCF before time 0.");
      }

      auto it = std::upper_bound(m_points.begin(), m_points.end(), t,
        [](time_type time, const point_type& pt) { return time < pt.t; });

      if (it == m_points.begin())
      {
        throw std::domain_error("Cannot evaluate PCF before time 0.");
      }

      --it;
      return it->v;
    }

    template <typename TIn, typename TOut>
    void evaluate(const TIn& sorted_times, TOut& out, size_t n) const
    {
      auto it = m_points.begin();
      auto end = m_points.end();

      for (size_t i = 0; i < n; ++i)
      {
        auto t = sorted_times(i);

        // PCFs are defined on [0, inf); guard against 0, not the first
        // breakpoint's time (which the t=0 construction invariant pins to 0).
        if (t < static_cast<time_type>(0))
        {
          throw std::domain_error("Cannot evaluate PCF before time 0.");
        }

        while (std::next(it) != end && std::next(it)->t <= t)
        {
          ++it;
        }

        out(i) = it->v;
      }
    }

    /**
     * Raise every value of the PCF to a power.
     *
     * Returns a new PCF whose value at every breakpoint is raised to
     * `exponent`. The domain (time coordinates) is unchanged.
     * Squaring (exponent == 2) avoids `std::pow` for performance.
     *
     * @param exponent the exponent to raise each value to
     * @return a new PCF with transformed values
     */
    template <Arithmetic T>
    [[nodiscard]] Pcf pow(T exponent) const
    {
      auto pts = m_points;
      if (exponent == 2)
      {
        for (auto& pt : pts)
        {
          pt.v = pt.v * pt.v;
        }
      }
      else
      {
        for (auto& pt : pts)
        {
          pt.v = static_cast<Tv>(std::pow(pt.v, exponent));
        }
      }
      return Pcf(std::move(pts));
    }

    [[nodiscard]] const std::vector<point_type>& points() const { return m_points; }
    [[nodiscard]] size_t size() const noexcept { return m_points.size(); }
    
    void swap(Pcf& other) noexcept
    {
      m_points.swap(other.m_points);
    }
    
    template <typename Ut, typename Uv>
    friend void PrintTo(const Pcf& f, std::ostream* os);

  private:
    template <typename T1, typename T2>
    void initFromPairs(std::initializer_list<std::pair<T1, T2>> pts)
    {
      m_points.reserve(pts.size());
      for (auto const& pt : pts)
      {
        m_points.emplace_back(static_cast<Tt>(pt.first), static_cast<Tv>(pt.second));
      }
    }

    std::vector<point_type> m_points;
  };
  
  using Pcf_f32 = Pcf<float32_t, float32_t>;
  using Pcf_f64 = Pcf<float64_t, float64_t>;
  using Pcf_i32 = Pcf<int32_t, int32_t>;
  using Pcf_i64 = Pcf<int64_t, int64_t>;
  
  template <typename Tt, typename Tv>
  [[nodiscard]] Pcf<Tt, Tv> operator+(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g)
  {
    return combine(f, g, [](const typename Pcf<Tt, Tv>::rectangle_type& rect){
      return rect.f_value + rect.g_value;
    });
  }

  template <typename Tt, typename Tv>
  [[nodiscard]] Pcf<Tt, Tv> operator-(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g)
  {
    Pcf<Tt, Tv> ret = f;
    ret -= g;
    return ret;
  }

  template <typename Tt, typename Tv>
  [[nodiscard]] Pcf<Tt, Tv> operator*(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g)
  {
    return combine(f, g, [](const typename Pcf<Tt, Tv>::rectangle_type& rect){
      return rect.f_value * rect.g_value;
    });
  }

  template <typename Tt, typename Tv>
  [[nodiscard]] Pcf<Tt, Tv> operator/(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g)
  {
    return combine(f, g, [](const typename Pcf<Tt, Tv>::rectangle_type& rect){
      return rect.f_value / rect.g_value;
    });
  }

  template <typename Tt, typename Tv>
  [[nodiscard]] Pcf<Tt, Tv> average(const std::vector<Pcf<Tt, Tv>>& fs, size_t chunksz = 2ul)
  {
    auto f = parallel_reduce(fs.begin(), fs.end(), [](const typename Pcf<Tt, Tv>::rectangle_type& rect) {
      return rect.f_value + rect.g_value;
    }, chunksz);
    return f / static_cast<Tv>(fs.size());
  }

  template <typename Tt, typename Tv>
  void PrintTo(const Pcf<Tt, Tv>& f, std::ostream* os)
  {
    *os << f.to_string();
  }

  template <typename F>
  concept PcfLike = PiecewiseFunctionLike<F> && requires
  {
    typename F::rectangle_type;
    typename F::segment_type;
  };
}

#endif
