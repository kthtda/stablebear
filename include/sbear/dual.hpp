#ifndef SB_DUAL_H
#define SB_DUAL_H

#include <cmath>

namespace sb
{
  /// Forward-mode dual number v + d*eps with eps^2 = 0. Evaluating a function
  /// templated on its argument type at Dual{x, 1} gives the value at x in v and
  /// the derivative with respect to x in d.
  template <typename T>
  struct Dual
  {
    T v{ 0 };
    T d{ 0 };

    constexpr Dual() = default;
    constexpr Dual(T value, T deriv = T{ 0 }) : v(value), d(deriv) { }

    constexpr Dual& operator+=(const Dual& o)
    {
      v += o.v;
      d += o.d;
      return *this;
    }
  };

  template <typename T>
  constexpr Dual<T> operator*(const Dual<T>& a, const Dual<T>& b)
  {
    return { a.v * b.v, a.d * b.v + a.v * b.d };
  }

  template <typename T>
  constexpr Dual<T> operator*(T a, const Dual<T>& b)
  {
    return { a * b.v, a * b.d };
  }

  template <typename T>
  constexpr Dual<T> operator/(const Dual<T>& a, const Dual<T>& b)
  {
    return { a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v) };
  }

  /// a^b. Taken as 0 with derivative 0 at a = 0, which is the limit for b > 0.
  template <typename T>
  Dual<T> pow(const Dual<T>& a, const Dual<T>& b)
  {
    if (a.v == 0)
    {
      return {};
    }
    T v = std::pow(a.v, b.v);
    return { v, v * (b.d * std::log(a.v) + b.v * a.d / a.v) };
  }

  template <typename T>
  Dual<T> pow(T a, const Dual<T>& b)
  {
    return pow(Dual<T>(a), b);
  }
}

#endif
