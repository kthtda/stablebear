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

#ifndef STABLEBEAR_TENSOR_1D_VALUE_ITERATOR_H
#define STABLEBEAR_TENSOR_1D_VALUE_ITERATOR_H

#include <cstddef>
#include <iterator>

namespace sb
{

  template<typename TensorT>
  class Tensor1dValueIterator
  {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = typename TensorT::value_type;
    using pointer = value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using self_type = Tensor1dValueIterator;
    using iterator_category = std::random_access_iterator_tag;

    constexpr Tensor1dValueIterator() noexcept
        : m_cur(nullptr), m_stride(0) {
    }

    constexpr Tensor1dValueIterator(TensorT& tensor, size_t idx) noexcept
        : m_cur(&tensor(idx)), m_stride(tensor.stride(0)) {
    }

    constexpr Tensor1dValueIterator(const Tensor1dValueIterator& other) noexcept
        : m_cur(other.m_cur), m_stride(other.m_stride) {
    }

    constexpr Tensor1dValueIterator& operator=(const Tensor1dValueIterator& other) noexcept {
      m_cur = other.m_cur;
      m_stride = other.m_stride;
      return *this;
    }

    constexpr Tensor1dValueIterator& operator=(Tensor1dValueIterator&& other) noexcept {
      m_cur = other.m_cur;
      m_stride = other.m_stride;
      return *this;
    }

    [[nodiscard]] constexpr bool operator==(const Tensor1dValueIterator& rhs) const noexcept {
      return m_cur == rhs.m_cur && m_stride == rhs.m_stride;
    }

    [[nodiscard]] constexpr reference operator*() const noexcept {
      return *m_cur;
    }

    [[nodiscard]] constexpr pointer operator->() const noexcept {
      return m_cur;
    }

    [[nodiscard]] constexpr reference operator[](difference_type n) const noexcept {
      return *(*this + n);
    }

    constexpr self_type& operator++() noexcept {
      m_cur += m_stride;
      return *this;
    }

    constexpr self_type operator++(int) noexcept {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    constexpr self_type& operator--() noexcept {
      m_cur -= m_stride;
      return *this;
    }

    constexpr self_type operator--(int) noexcept {
      auto tmp = *this;
      --*this;
      return tmp;
    }

    constexpr self_type& operator+=(difference_type n) noexcept {
      m_cur += n * static_cast<difference_type>(m_stride);
      return *this;
    }

    constexpr self_type& operator-=(difference_type n) noexcept {
      m_cur -= n * static_cast<difference_type>(m_stride);
      return *this;
    }

    constexpr self_type operator+(difference_type n) const noexcept {
      auto tmp = *this;
      tmp += n;
      return tmp;
    }

    [[nodiscard]] constexpr difference_type operator-(const self_type& other) const noexcept {
      if (m_stride == 0)
      {
        return 0; // Prevent division by zero in constexpr
      }
      return static_cast<difference_type>(m_cur - other.m_cur) / m_stride;
    }

    constexpr self_type operator-(difference_type n) const noexcept {
      auto tmp = *this;
      tmp -= n;
      return tmp;
    }

    [[nodiscard]] constexpr bool operator<(const self_type& rhs) const noexcept {
      return m_cur < rhs.m_cur;
    }

    [[nodiscard]] constexpr bool operator<=(const self_type& rhs) const noexcept {
      return m_cur <= rhs.m_cur;
    }

    [[nodiscard]] constexpr bool operator>(const self_type& rhs) const noexcept {
      return m_cur > rhs.m_cur;
    }

    [[nodiscard]] constexpr bool operator>=(const self_type& rhs) const noexcept {
      return m_cur >= rhs.m_cur;
    }

    [[nodiscard]] constexpr bool operator!=(const self_type& rhs) const noexcept {
      return m_cur != rhs.m_cur || m_stride != rhs.m_stride;
    }

  private:
    pointer m_cur;
    size_t m_stride;
  };

  template<typename TensorT>
  constexpr Tensor1dValueIterator<TensorT> operator+(ptrdiff_t n, Tensor1dValueIterator<TensorT> it) {
    it += n;
    return it;
  }

  template <IsTensor TensorT>
  Tensor1dValueIterator<TensorT> begin1dValues(TensorT& tensor)
  {
    return Tensor1dValueIterator<TensorT>(tensor, 0);
  }

  template <IsTensor TensorT>
  Tensor1dValueIterator<TensorT> end1dValues(TensorT& tensor)
  {
    return Tensor1dValueIterator<TensorT>(tensor, tensor.shape(0));
  }

}

#endif //STABLEBEAR_TENSOR_1D_VALUE_ITERATOR_H
