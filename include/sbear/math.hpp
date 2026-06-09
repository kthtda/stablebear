/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

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
