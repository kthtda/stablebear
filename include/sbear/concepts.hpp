// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STABLEBEAR_CONCEPTS_H
#define STABLEBEAR_CONCEPTS_H

#include "math.hpp"

#include <concepts>
#include <iterator>
#include <vector>

namespace sb
{
  template <typename T, typename U>
  concept CanDivide = requires(T t, U u)
  {
    t / u;
  };

  template <typename T, typename U>
  concept CanMultiply = requires(T t, U u)
  {
    t * u;
  };

  template <typename T, typename U>
  concept CanAdd = requires(T t, U u)
  {
    t + u;
  };

  template <typename T, typename U>
  concept CanSubtract = requires(T t, U u)
  {
    t - u;
  };

  // "Into" variants: check that the result of A op B is convertible to R.
  // Use these for operators whose body stores the result back into an element
  // of type R (e.g. compound assignment, elementwise free operators).

  template <typename R, typename A, typename B>
  concept CanAddTo = requires(A a, B b)
  {
    { a + b } -> std::convertible_to<R>;
  };

  template <typename R, typename A, typename B>
  concept CanSubtractTo = requires(A a, B b)
  {
    { a - b } -> std::convertible_to<R>;
  };

  template <typename R, typename A, typename B>
  concept CanMultiplyTo = requires(A a, B b)
  {
    { a * b } -> std::convertible_to<R>;
  };

  template <typename R, typename A, typename B>
  concept CanDivideTo = requires(A a, B b)
  {
    { a / b } -> std::convertible_to<R>;
  };

  /// Satisfied when `sb::pow(T, U)` is a valid expression.
  /// This is true for arithmetic scalars (via `std::pow`) and for any
  /// type that provides a `.pow()` member (e.g. `Pcf`).
  template <typename T, typename U>
  concept CanPow = requires(T t, U u)
  {
    { sb::pow(t, u) };
  };

  template <typename T>
  concept CanNegate = requires(T t)
  {
    { -t } -> std::convertible_to<T>;
  };

  template <typename T>
  concept CanOrder = requires(const T& a, const T& b)
  {
    { a < b } -> std::convertible_to<bool>;
  };

  // Any type that can be evaluated at a point in DomainT, yielding a value convertible to CodomainT.
  template <typename T, typename DomainT, typename CodomainT>
  concept Evaluable = requires(T t, DomainT x)
  {
    { t.evaluate(x) } -> std::convertible_to<CodomainT>;
  };

  template <typename T>
  concept Iterable = requires(T t)
  {
    std::begin(t);
    std::end(t);
  };

  template <typename T>
  concept IsTensor = requires(T t, std::vector<size_t> indices)
  {
    { t.shape() } -> Iterable;
    { t.strides() } -> Iterable;
    { t.rank() } -> std::convertible_to<size_t>;
    { t.size() } -> std::convertible_to<size_t>;
    { t(indices) } -> std::common_with<typename T::value_type>;

    typename T::value_type;
  };
  template <typename T>
  struct is_compressed_matrix : std::false_type {};

  template <typename T>
  inline constexpr bool is_compressed_matrix_v = is_compressed_matrix<T>::value;
}

#endif //STABLEBEAR_CONCEPTS_H
