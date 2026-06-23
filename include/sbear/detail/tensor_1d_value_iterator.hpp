#ifndef STABLEBEAR_TENSOR_1D_VALUE_ITERATOR_H
#define STABLEBEAR_TENSOR_1D_VALUE_ITERATOR_H

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

namespace sb
{

  // Random-access iterator over a single tensor dimension. It walks the
  // elements base[pos * stride] for an increasing logical position `pos`, where
  // `stride` is the tensor's stride along the chosen dimension. Tracking the
  // logical position (rather than only a raw pointer) keeps it well-defined for
  // a broadcast dimension whose stride is 0: such a range still spans the
  // requested number of (identical) elements instead of collapsing to empty.
  //
  // TensorT may be const-qualified; when it is, the iterator yields const
  // references, so it can read a `const Tensor&` without copying its elements.
  template<typename TensorT>
  class Tensor1dValueIterator
  {
    using element_type = std::conditional_t<
        std::is_const_v<TensorT>,
        const typename std::remove_const_t<TensorT>::value_type,
        typename TensorT::value_type>;

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = std::remove_const_t<element_type>;
    using pointer = element_type*;
    using reference = element_type&;
    using const_reference = const element_type&;
    using self_type = Tensor1dValueIterator;
    using iterator_category = std::random_access_iterator_tag;

    constexpr Tensor1dValueIterator() noexcept
        : m_base(nullptr), m_stride(0), m_pos(0) {
    }

    // Iterate dimension 0 of `tensor`, positioned at `pos`. The base is pinned
    // to the start of the dimension so that begin/end iterators (which differ
    // only in `pos`) compare equal once they reach the same position.
    Tensor1dValueIterator(TensorT& tensor, size_t pos) noexcept
        : m_base(&tensor(0)), m_stride(tensor.stride(0)), m_pos(static_cast<difference_type>(pos)) {
    }

    // Iterate dimension `dim` of `tensor` over the slice whose remaining indices
    // are fixed by `startIndex` (its `dim` component is the origin, i.e. 0),
    // positioned at `pos`.
    Tensor1dValueIterator(TensorT& tensor, const std::vector<size_t>& startIndex, size_t dim, size_t pos) noexcept
        : m_base(&tensor(startIndex)), m_stride(tensor.stride(dim)), m_pos(static_cast<difference_type>(pos)) {
    }

    constexpr Tensor1dValueIterator(const Tensor1dValueIterator& other) noexcept = default;
    constexpr Tensor1dValueIterator& operator=(const Tensor1dValueIterator& other) noexcept = default;
    constexpr Tensor1dValueIterator(Tensor1dValueIterator&& other) noexcept = default;
    constexpr Tensor1dValueIterator& operator=(Tensor1dValueIterator&& other) noexcept = default;

    [[nodiscard]] constexpr reference operator*() const noexcept {
      return m_base[m_pos * m_stride];
    }

    [[nodiscard]] constexpr pointer operator->() const noexcept {
      return m_base + m_pos * m_stride;
    }

    [[nodiscard]] constexpr reference operator[](difference_type n) const noexcept {
      return m_base[(m_pos + n) * m_stride];
    }

    constexpr self_type& operator++() noexcept {
      ++m_pos;
      return *this;
    }

    constexpr self_type operator++(int) noexcept {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    constexpr self_type& operator--() noexcept {
      --m_pos;
      return *this;
    }

    constexpr self_type operator--(int) noexcept {
      auto tmp = *this;
      --*this;
      return tmp;
    }

    constexpr self_type& operator+=(difference_type n) noexcept {
      m_pos += n;
      return *this;
    }

    constexpr self_type& operator-=(difference_type n) noexcept {
      m_pos -= n;
      return *this;
    }

    constexpr self_type operator+(difference_type n) const noexcept {
      auto tmp = *this;
      tmp += n;
      return tmp;
    }

    constexpr self_type operator-(difference_type n) const noexcept {
      auto tmp = *this;
      tmp -= n;
      return tmp;
    }

    [[nodiscard]] constexpr difference_type operator-(const self_type& other) const noexcept {
      return m_pos - other.m_pos;
    }

    [[nodiscard]] constexpr bool operator==(const self_type& rhs) const noexcept {
      return m_pos == rhs.m_pos && m_base == rhs.m_base && m_stride == rhs.m_stride;
    }

    [[nodiscard]] constexpr bool operator!=(const self_type& rhs) const noexcept {
      return !(*this == rhs);
    }

    [[nodiscard]] constexpr bool operator<(const self_type& rhs) const noexcept {
      return m_pos < rhs.m_pos;
    }

    [[nodiscard]] constexpr bool operator<=(const self_type& rhs) const noexcept {
      return m_pos <= rhs.m_pos;
    }

    [[nodiscard]] constexpr bool operator>(const self_type& rhs) const noexcept {
      return m_pos > rhs.m_pos;
    }

    [[nodiscard]] constexpr bool operator>=(const self_type& rhs) const noexcept {
      return m_pos >= rhs.m_pos;
    }

  private:
    pointer m_base;
    difference_type m_stride;
    difference_type m_pos;
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
