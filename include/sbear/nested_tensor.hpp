#ifndef STABLEBEAR_NESTED_TENSOR_H
#define STABLEBEAR_NESTED_TENSOR_H

#include "tensor.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

namespace sb
{
  /// One element in a recursively nested tensor with a fixed leaf type.
  ///
  /// A node contains either a leaf Tensor<LeafT> or another tensor of nodes.
  /// The shared storage breaks the recursive type definition and keeps node
  /// copies cheap; copy() provides an explicit recursive deep copy.
  template <typename LeafT>
  class NestedTensor
  {
  public:
    using leaf_type = LeafT;
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

    [[nodiscard]] bool is_leaf() const noexcept
    {
      return std::holds_alternative<Tensor<LeafT>>(*m_storage);
    }

    [[nodiscard]] size_t depth() const noexcept { return m_depth; }

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

  template <typename T>
  struct is_nested_tensor : std::false_type {};

  template <typename T>
  struct is_nested_tensor<NestedTensor<T>> : std::true_type { using leaf_type = T; };

  template <typename T>
  inline constexpr bool is_nested_tensor_v = is_nested_tensor<T>::value;

} // namespace sb

#endif // STABLEBEAR_NESTED_TENSOR_H
