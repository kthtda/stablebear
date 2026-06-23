#ifndef SB_PIECEWISE_FUNCTION_H
#define SB_PIECEWISE_FUNCTION_H

#include "time_point.hpp"
#include "../config.hpp"

#include <vector>
#include <cmath>

namespace sb
{
  // Forward declaration for type traits
  template <typename Tt, typename Tv> class Pcf;

  // --- Type traits ---

  template <typename T> struct is_pcf : std::false_type {};
  template <typename Tt, typename Tv> struct is_pcf<Pcf<Tt, Tv>> : std::true_type {};
  template <typename T> inline constexpr bool is_pcf_v = is_pcf<T>::value;

  /// Concept for any piecewise function type.
  template <typename F>
  concept PiecewiseFunctionLike = requires(const F cf)
  {
    typename F::point_type;
    typename F::time_type;
    typename F::value_type;

    requires TimePointLike<typename F::point_type>;

    { cf.points() } -> std::convertible_to<const std::vector<typename F::point_type>&>;
    { cf.size()   } -> std::convertible_to<std::size_t>;

    { cf == cf } -> std::convertible_to<bool>;
    { cf != cf } -> std::convertible_to<bool>;
  };

  // --- Scalar free operators for any PiecewiseFunctionLike type ---

  template <PiecewiseFunctionLike F, Arithmetic Ts>
  [[nodiscard]] F operator+(const F& f, Ts val)
  {
    F ret = f;
    ret += val;
    return ret;
  }

  template <PiecewiseFunctionLike F, Arithmetic Ts>
  [[nodiscard]] F operator+(Ts val, const F& f)
  {
    F ret = f;
    ret += val;
    return ret;
  }

  template <PiecewiseFunctionLike F, Arithmetic Ts>
  [[nodiscard]] F operator-(const F& f, Ts val)
  {
    F ret = f;
    ret -= val;
    return ret;
  }

  template <PiecewiseFunctionLike F, Arithmetic Ts>
  [[nodiscard]] F operator-(Ts val, const F& f)
  {
    F ret = -f;
    ret += val;
    return ret;
  }

  template <PiecewiseFunctionLike F, Arithmetic Ts>
  [[nodiscard]] F operator*(const F& f, Ts val)
  {
    F ret = f;
    ret *= val;
    return ret;
  }

  template <PiecewiseFunctionLike F, Arithmetic Ts>
  [[nodiscard]] F operator*(Ts val, const F& f)
  {
    F ret = f;
    ret *= val;
    return ret;
  }

  template <PiecewiseFunctionLike F, Arithmetic Ts>
  [[nodiscard]] F operator/(const F& f, Ts val)
  {
    F ret = f;
    ret /= val;
    return ret;
  }

  template <PiecewiseFunctionLike F, Arithmetic Ts>
  [[nodiscard]] F operator/(Ts val, const F& f)
  {
    auto pts = f.points();
    for (auto& pt : pts)
    {
      pt.v = static_cast<typename F::value_type>(val) / pt.v;
    }
    return F(std::move(pts));
  }

}

#endif
