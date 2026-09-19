#ifndef STABLEBEAR_NESTED_TENSOR_H
#define STABLEBEAR_NESTED_TENSOR_H

#include "tensor.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

namespace sb
{
  /// A recursively nested tensor with a fixed leaf type.
  ///
  /// The tensor contains either leaf values or nested tensors at each level.
  /// The shared storage breaks the recursive type definition and keeps node
  /// copies cheap; copy() provides an explicit recursive deep copy.
  template <typename LeafT>
  class NestedTensor
  {
  public:
    using leaf_type = LeafT;
    using value_type = std::variant<LeafT, NestedTensor<LeafT>>;
    using nested_tensor_type = Tensor<NestedTensor<LeafT>>;
    using storage_type = std::variant<Tensor<LeafT>, nested_tensor_type>;

    NestedTensor() : m_storage(std::make_shared<storage_type>(Tensor<LeafT>({0}))), m_depth(1) { }

    explicit NestedTensor(const Tensor<LeafT>& leaf)
      : m_storage(std::make_shared<storage_type>(leaf)), m_depth(1) { }

    explicit NestedTensor(Tensor<LeafT>&& leaf)
      : m_storage(std::make_shared<storage_type>(std::move(leaf))), m_depth(1) { }

    explicit NestedTensor(const nested_tensor_type& nested, size_t childDepth = 0)
      : NestedTensor(copy_nested(nested), childDepth) { }

    explicit NestedTensor(nested_tensor_type&& nested, size_t childDepth = 0)
      : m_storage(std::make_shared<storage_type>(std::move(nested)))
    {
      const auto& values = std::get<nested_tensor_type>(*m_storage);
      size_t inferredDepth = childDepth;
      bool found = false;
      walk(values, [&](const std::vector<size_t>& index) {
        const size_t depth = values(index).depth();
        if (!found)
        {
          inferredDepth = depth;
          found = true;
        }
        else if (depth != inferredDepth)
        {
          throw std::invalid_argument("NestedTensor children must have the same nesting depth");
        }
      });
      if (!found && childDepth == 0)
      {
        throw std::invalid_argument("Empty NestedTensor requires an explicit child depth");
      }
      m_depth = inferredDepth + 1;
    }

    /// Wrap outer tensor view metadata without recursively copying its children.
    /// The caller must already own the child values; public value construction
    /// through the const-reference constructor intentionally copies them.
    [[nodiscard]] static NestedTensor from_outer_view(
      nested_tensor_type view, size_t childDepth)
    {
      return NestedTensor(std::move(view), childDepth);
    }

    [[nodiscard]] bool is_leaf() const noexcept
    {
      return std::holds_alternative<Tensor<LeafT>>(*m_storage);
    }

    [[nodiscard]] size_t depth() const noexcept { return m_depth; }

    [[nodiscard]] const std::vector<size_t>& shape() const noexcept
    {
      return is_leaf() ? leaf().shape() : nested().shape();
    }

    [[nodiscard]] size_t shape(size_t dim) const noexcept
    {
      return is_leaf() ? leaf().shape(dim) : nested().shape(dim);
    }

    [[nodiscard]] const std::vector<ptrdiff_t>& strides() const noexcept
    {
      return is_leaf() ? leaf().strides() : nested().strides();
    }

    [[nodiscard]] ptrdiff_t stride(size_t dim) const noexcept
    {
      return is_leaf() ? leaf().stride(dim) : nested().stride(dim);
    }

    [[nodiscard]] size_t rank() const noexcept
    {
      return is_leaf() ? leaf().rank() : nested().rank();
    }

    [[nodiscard]] size_t size() const noexcept
    {
      return is_leaf() ? leaf().size() : nested().size();
    }

    [[nodiscard]] bool is_contiguous() const noexcept
    {
      return is_leaf() ? leaf().is_contiguous() : nested().is_contiguous();
    }

    [[nodiscard]] value_type operator()(const std::vector<size_t>& index) const;
    [[nodiscard]] value_type operator()(size_t index) const { return (*this)(std::vector<size_t>{index}); }
    [[nodiscard]] value_type flat(size_t index) const;

    template <typename SliceVector>
    [[nodiscard]] NestedTensor operator[](SliceVector sliceVector) const
    {
      if (is_leaf())
      {
        return NestedTensor(leaf()[sliceVector]);
      }
      return NestedTensor(nested()[sliceVector], m_depth - 1);
    }

    [[nodiscard]] NestedTensor broadcast_to(const std::vector<size_t>& shape) const
    {
      if (is_leaf())
      {
        return NestedTensor(leaf().broadcast_to(shape));
      }
      return NestedTensor(nested().broadcast_to(shape), m_depth - 1);
    }

    [[nodiscard]] NestedTensor flatten() const
    {
      if (is_leaf())
      {
        return NestedTensor(leaf().flatten());
      }
      return NestedTensor(nested().flatten(), m_depth - 1);
    }

    [[nodiscard]] NestedTensor reshape(const std::vector<ptrdiff_t>& shape) const
    {
      if (is_leaf())
      {
        return NestedTensor(leaf().reshape(shape));
      }
      return NestedTensor(nested().reshape(shape), m_depth - 1);
    }

    [[nodiscard]] NestedTensor transpose(const std::vector<size_t>& axes = {}) const
    {
      if (is_leaf())
      {
        return NestedTensor(leaf().transpose(axes));
      }
      return NestedTensor(nested().transpose(axes), m_depth - 1);
    }

    [[nodiscard]] NestedTensor swapaxes(size_t axis1, size_t axis2) const
    {
      if (is_leaf())
      {
        return NestedTensor(leaf().swapaxes(axis1, axis2));
      }
      return NestedTensor(nested().swapaxes(axis1, axis2), m_depth - 1);
    }

    [[nodiscard]] NestedTensor squeeze() const
    {
      if (is_leaf())
      {
        return NestedTensor(leaf().squeeze());
      }
      return NestedTensor(nested().squeeze(), m_depth - 1);
    }

    [[nodiscard]] NestedTensor squeeze(size_t axis) const
    {
      if (is_leaf())
      {
        return NestedTensor(leaf().squeeze(axis));
      }
      return NestedTensor(nested().squeeze(axis), m_depth - 1);
    }

    [[nodiscard]] NestedTensor expand_dims(ptrdiff_t axis) const
    {
      if (is_leaf())
      {
        return NestedTensor(leaf().expand_dims(axis));
      }
      return NestedTensor(nested().expand_dims(axis), m_depth - 1);
    }

    [[nodiscard]] const Tensor<LeafT>& leaf() const
    {
      if (!is_leaf())
      {
        throw std::logic_error("NestedTensor node is not a leaf tensor");
      }
      return std::get<Tensor<LeafT>>(*m_storage);
    }

    [[nodiscard]] const nested_tensor_type& nested() const
    {
      if (is_leaf())
      {
        throw std::logic_error("NestedTensor node is not a nested tensor");
      }
      return std::get<nested_tensor_type>(*m_storage);
    }

    [[nodiscard]] NestedTensor copy() const
    {
      if (is_leaf())
      {
        return NestedTensor(leaf().copy());
      }

      nested_tensor_type result(nested().shape());
      walk(nested(), [&](const std::vector<size_t>& index) {
        result(index) = nested()(index).copy();
      });
      return NestedTensor(std::move(result), m_depth - 1);
    }

    [[nodiscard]] bool operator==(const NestedTensor& rhs) const
    {
      if (m_depth != rhs.m_depth || is_leaf() != rhs.is_leaf())
      {
        return false;
      }
      return is_leaf() ? leaf() == rhs.leaf() : nested() == rhs.nested();
    }

    [[nodiscard]] bool operator!=(const NestedTensor& rhs) const { return !(*this == rhs); }

  private:
    [[nodiscard]] static nested_tensor_type copy_nested(const nested_tensor_type& source)
    {
      nested_tensor_type result(source.shape());
      walk(source, [&](const std::vector<size_t>& index) {
        result(index) = source(index).copy();
      });
      return result;
    }

    std::shared_ptr<storage_type> m_storage;
    size_t m_depth;
  };

  template <typename LeafT>
  typename NestedTensor<LeafT>::value_type NestedTensor<LeafT>::operator()(
      const std::vector<size_t>& index) const
  {
    if (is_leaf())
    {
      return leaf()(index);
    }
    return nested()(index);
  }

  template <typename LeafT>
  typename NestedTensor<LeafT>::value_type NestedTensor<LeafT>::flat(size_t index) const
  {
    if (is_leaf())
    {
      return leaf().flat(index);
    }
    return nested().flat(index);
  }

  template <typename T>
  struct is_nested_tensor : std::false_type {};

  template <typename T>
  struct is_nested_tensor<NestedTensor<T>> : std::true_type { using leaf_type = T; };

  template <typename T>
  inline constexpr bool is_nested_tensor_v = is_nested_tensor<T>::value;

  namespace detail
  {
    template <typename T>
    struct TensorNesting
    {
      using leaf_type = T;
      static constexpr size_t depth = 0;
    };

    template <typename T>
    struct TensorNesting<Tensor<T>>
    {
      using leaf_type = typename TensorNesting<T>::leaf_type;
      static constexpr size_t depth = TensorNesting<T>::depth + 1;
    };
  }

  /// Convert a statically nested Tensor type into the equivalent runtime
  /// NestedTensor. Leaf tensors retain their storage; outer tensor levels are
  /// converted recursively. The source type supplies the depth even when an
  /// outer tensor is empty.
  template <typename T>
  [[nodiscard]] auto to_nested_tensor(const Tensor<T>& tensor)
  {
    using Source = detail::TensorNesting<T>;
    using LeafT = typename Source::leaf_type;

    if constexpr (Source::depth == 0)
    {
      return NestedTensor<LeafT>(tensor);
    }
    else
    {
      Tensor<NestedTensor<LeafT>> children(tensor.shape());
      walk(tensor, [&](const std::vector<size_t>& index) {
        children(index) = to_nested_tensor(tensor(index));
      });
      return NestedTensor<LeafT>(std::move(children), Source::depth);
    }
  }

  static_assert(IsTensor<NestedTensor<uint64_t>>);

} // namespace sb

#endif // STABLEBEAR_NESTED_TENSOR_H
