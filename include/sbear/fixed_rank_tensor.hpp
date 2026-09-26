#ifndef STABLEBEAR_FIXED_RANK_TENSOR_H
#define STABLEBEAR_FIXED_RANK_TENSOR_H

#include <algorithm>
#include <array>
#include <concepts>
#include <compare>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace sb
{
  template <size_t Rank>
  class ContiguousFixedRankLayout
  {
  public:
    using shape_type = std::array<size_t, Rank>;

    ContiguousFixedRankLayout() = default;
    explicit ContiguousFixedRankLayout(shape_type shape)
      : m_shape(std::move(shape))
    { }

    [[nodiscard]] const shape_type& shape() const noexcept { return m_shape; }

    [[nodiscard]] size_t storage_size() const noexcept
    {
      return std::accumulate(
        m_shape.begin(), m_shape.end(), size_t{1}, std::multiplies<size_t>());
    }

    [[nodiscard]] size_t offset(const shape_type& indices) const
    {
      size_t result = 0;
      for (size_t axis = 0; axis < Rank; ++axis)
      {
        if (indices[axis] >= m_shape[axis])
          throw std::out_of_range("FixedRankTensor index out of range");
        result = result * m_shape[axis] + indices[axis];
      }
      return result;
    }

  private:
    shape_type m_shape{};
  };

  template <typename Layout, size_t Rank>
  concept FixedRankLayout = requires(
      const Layout& layout, const std::array<size_t, Rank>& indices)
  {
    { layout.shape() } -> std::convertible_to<std::array<size_t, Rank>>;
    { layout.storage_size() } -> std::convertible_to<size_t>;
    { layout.offset(indices) } -> std::convertible_to<size_t>;
  };

  /// Shared contiguous storage with compile-time logical rank and an adapter
  /// that maps logical coordinates into that storage.
  template <typename T, size_t Rank,
      FixedRankLayout<Rank> Layout = ContiguousFixedRankLayout<Rank>>
  class FixedRankTensor
  {
  public:
    using value_type = T;
    using shape_type = std::array<size_t, Rank>;
    using layout_type = Layout;

    FixedRankTensor() : FixedRankTensor(Layout{}) { }

    explicit FixedRankTensor(shape_type shape, const T& init = {})
      requires std::constructible_from<Layout, shape_type>
      : FixedRankTensor(Layout(std::move(shape)), init)
    { }

    explicit FixedRankTensor(Layout layout, const T& init = {})
      : m_layout(std::move(layout))
      , m_storage(std::make_shared<T[]>(storage_size()))
    {
      std::fill_n(m_storage.get(), storage_size(), init);
    }

    [[nodiscard]] static constexpr size_t rank() noexcept { return Rank; }
    [[nodiscard]] shape_type shape() const noexcept { return m_layout.shape(); }

    [[nodiscard]] size_t shape(size_t axis) const
    {
      if (axis >= Rank)
        throw std::out_of_range("FixedRankTensor axis out of range");
      return shape()[axis];
    }

    [[nodiscard]] size_t size() const noexcept
    {
      const auto logicalShape = shape();
      return std::accumulate(
        logicalShape.begin(), logicalShape.end(), size_t{1},
        std::multiplies<size_t>());
    }

    [[nodiscard]] size_t storage_size() const noexcept
    {
      return m_layout.storage_size();
    }

    [[nodiscard]] const Layout& layout() const noexcept { return m_layout; }
    [[nodiscard]] const T* storage_data() const noexcept { return m_storage.get(); }
    [[nodiscard]] T* storage_data() noexcept { return m_storage.get(); }

    [[nodiscard]] const T& flat(size_t index) const
    {
      return (*this)(indices_from_flat(index));
    }

    [[nodiscard]] T& flat(size_t index)
    {
      return (*this)(indices_from_flat(index));
    }

    /// Borrow a read-only row-major range, advancing coordinates without
    /// converting a flat index on every access. The tensor must outlive it.
    struct FlatView
    {
      const FixedRankTensor* tensor;
      struct Iterator
      {
        using value_type = T;
        using difference_type = ptrdiff_t;
        using reference = const T&;
        using pointer = const T*;
        using iterator_concept = std::random_access_iterator_tag;
        using iterator_category = std::random_access_iterator_tag;

        const FixedRankTensor* tensor = nullptr;
        shape_type shape{};
        shape_type indices{};
        difference_type position = 0;

        reference operator*() const { return (*tensor)(indices); }
        pointer operator->() const { return std::addressof(**this); }
        reference operator[](difference_type offset) const { return *(*this + offset); }
        Iterator& operator++()
        {
          ++position;
          for (size_t axis = Rank; axis-- > 0;)
          {
            if (++indices[axis] < shape[axis])
              break;
            indices[axis] = 0;
          }
          return *this;
        }
        Iterator operator++(int) { auto previous = *this; ++*this; return previous; }
        Iterator& operator--()
        {
          --position;
          for (size_t axis = Rank; axis-- > 0;)
          {
            if (indices[axis] != 0)
            {
              --indices[axis];
              break;
            }
            indices[axis] = shape[axis] - 1;
          }
          return *this;
        }
        Iterator operator--(int) { auto previous = *this; --*this; return previous; }
        Iterator& operator+=(difference_type offset)
        {
          if (offset == 0)
            return *this;
          position += offset;
          size_t flat = static_cast<size_t>(position);
          for (size_t axis = Rank; axis-- > 0;)
          {
            indices[axis] = flat % shape[axis];
            flat /= shape[axis];
          }
          return *this;
        }
        Iterator& operator-=(difference_type offset) { return *this += -offset; }
        friend Iterator operator+(Iterator it, difference_type offset) { return it += offset; }
        friend Iterator operator+(difference_type offset, Iterator it) { return it += offset; }
        friend Iterator operator-(Iterator it, difference_type offset) { return it -= offset; }
        difference_type operator-(const Iterator& other) const { return position - other.position; }
        bool operator==(const Iterator& other) const
        {
          return tensor == other.tensor && position == other.position;
        }
        auto operator<=>(const Iterator& other) const { return position <=> other.position; }
      };
      Iterator begin() const { return {tensor, tensor->shape(), {}, 0}; }
      Iterator end() const
      {
        return {tensor, tensor->shape(), {}, static_cast<ptrdiff_t>(tensor->size())};
      }
    };

    [[nodiscard]] FlatView flat_view() const & { return {this}; }
    auto flat_view() const && = delete;

    template <typename... Indices>
      requires (sizeof...(Indices) == Rank)
        && (std::convertible_to<Indices, size_t> && ...)
    [[nodiscard]] const T& operator()(Indices... indices) const
    {
      return (*this)(shape_type{static_cast<size_t>(indices)...});
    }

    [[nodiscard]] const T& operator()(const shape_type& indices) const
    {
      return m_storage[m_layout.offset(indices)];
    }

    [[nodiscard]] T& operator()(const shape_type& indices)
    {
      return m_storage[m_layout.offset(indices)];
    }

    template <typename... Indices>
      requires (sizeof...(Indices) == Rank)
        && (std::convertible_to<Indices, size_t> && ...)
    [[nodiscard]] T& operator()(Indices... indices)
    {
      return (*this)(shape_type{static_cast<size_t>(indices)...});
    }

    /// Return a shallow view with independent adapter metadata.
    [[nodiscard]] FixedRankTensor with_layout(Layout layout) const
    {
      if (layout.storage_size() != storage_size())
        throw std::invalid_argument(
          "FixedRankTensor view layout requires a different storage size");
      return FixedRankTensor(m_storage, std::move(layout));
    }

    /// Copy the complete physical storage while preserving the logical layout.
    [[nodiscard]] FixedRankTensor copy() const
    {
      FixedRankTensor result(m_layout);
      if (storage_size() != 0)
        std::copy_n(m_storage.get(), storage_size(), result.m_storage.get());
      return result;
    }

    [[nodiscard]] bool operator==(const FixedRankTensor& rhs) const
    {
      if (shape() != rhs.shape())
        return false;
      for (size_t i = 0; i < size(); ++i)
      {
        if (flat(i) != rhs.flat(i))
          return false;
      }
      return true;
    }

  private:
    FixedRankTensor(std::shared_ptr<T[]> storage, Layout layout)
      : m_layout(std::move(layout)), m_storage(std::move(storage))
    { }

    [[nodiscard]] shape_type indices_from_flat(size_t index) const
    {
      if (index >= size())
        throw std::out_of_range("FixedRankTensor index out of range");
      shape_type indices{};
      const auto logicalShape = shape();
      for (size_t axis = Rank; axis-- > 0;)
      {
        indices[axis] = index % logicalShape[axis];
        index /= logicalShape[axis];
      }
      return indices;
    }

    Layout m_layout;
    std::shared_ptr<T[]> m_storage;
  };
}

#endif // STABLEBEAR_FIXED_RANK_TENSOR_H
