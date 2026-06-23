#ifndef SB_MATH_OPS_H
#define SB_MATH_OPS_H

#include <cmath>
#include <type_traits>

namespace sb
{
  /**
   * Raise an arithmetic value to a power.
   *
   * Thin wrapper around `std::pow` that participates in the `sb::pow`
   * overload set, allowing generic code (tensors, concepts) to use a
   * single qualified call for both scalar and user-defined types.
   *
   * @param base     the base value
   * @param exponent the exponent
   * @return `std::pow(base, exponent)`
   */
  template <typename T, typename U>
  requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
  [[nodiscard]] auto pow(T base, U exponent)
  {
    return std::pow(base, exponent);
  }

  /**
   * Raise an object to a power by delegating to its `.pow()` member.
   *
   * Any type that provides a `.pow(exponent)` const member function
   * (e.g. `Pcf`) is automatically supported by this overload. This
   * keeps `sb::pow` extensible without modifying this header.
   *
   * @param t        the object to raise
   * @param exponent the exponent
   * @return `t.pow(exponent)`
   */
  template <typename T, typename U>
  requires requires(const T& t, U u) { t.pow(u); }
  [[nodiscard]] auto pow(const T& t, U exponent)
  {
    return t.pow(exponent);
  }
}

#endif // SB_MATH_OPS_H
