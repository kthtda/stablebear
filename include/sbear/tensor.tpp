#include <cmath>
#include <functional>
#include <type_traits>

#include <taskflow/algorithm/for_each.hpp>

#include "executor.hpp"
#include "walk.hpp"

namespace sb
{
  // Forward declarations -- definitions below walk()
  template <IsTensor TTensor, typename UnaryFunc>
#ifndef __CUDACC__
  requires std::invocable<UnaryFunc, std::vector<size_t>>
#endif
  tf::Future<void> parallel_walk_async(const TTensor& tensor, UnaryFunc&& f, Executor& exec);

  template <IsTensor TTensor, typename UnaryFunc>
#ifndef __CUDACC__
  requires std::invocable<UnaryFunc, std::vector<size_t>>
#endif
  void parallel_walk(const TTensor& tensor, UnaryFunc&& f, Executor& exec);

  template <typename T>
  Tensor<T>::Tensor(const std::vector<size_t>& shape, const T& init)
    : m_shape(shape)
  {
    auto sz = get_total_size();
    m_data = std::make_shared<T[]>(sz);
    std::fill_n(m_data.get(), sz, init);

    // Compute strides
    if (!m_shape.empty())
    {
      m_strides.resize(m_shape.size());
      m_strides.back() = 1;
      for (auto i = static_cast<ptrdiff_t>(m_shape.size()) - 2; i >= 0; --i)
        m_strides[i] = m_strides[i + 1] * static_cast<ptrdiff_t>(m_shape[i + 1]);
    }
  }

  template <typename T>
  Tensor<T>& Tensor<T>::operator=(const T& val)
  {
    apply([&val](T& element){ element = val; });
    return *this;
  }

  template <typename T>
  template <typename U> requires std::equality_comparable_with<U, T>
  bool Tensor<T>::operator==(const Tensor<U>& rhs) const
  {
    if (m_shape != rhs.shape())
    {
      return false;
    }

    bool equal = true;
    sb::walk(*this, [&equal, this, &rhs](const std::vector<size_t>& idx) {
      return (equal &= ( (*this)(idx) == rhs(idx) ) );
    });

    return equal;
  }

  template <typename T>
  template <typename U> requires std::equality_comparable_with<U, T>
  bool Tensor<T>::operator!=(const Tensor<U>& rhs) const
  {
    if (m_shape != rhs.shape())
    {
      return true;
    }

    return any_of_idx([this, &rhs](const std::vector<size_t>& idx) {
      return (*this)(idx) != rhs(idx);
    });
  }

  template <typename T>
  template <typename SliceVector>
  Tensor<T> Tensor<T>::operator[](SliceVector sliceVector) const
  {
    return extract(sliceVector);
  }

  template <typename T>
  const T& Tensor<T>::operator()(const std::vector<size_t>& index) const
  {
    return index_to_ref(index);
  }

  template <typename T>
  T& Tensor<T>::operator()(const std::vector<size_t>& index)
  {
    return index_to_ref(index);
  }

  template <typename T>
  const T& Tensor<T>::operator()(size_t index) const
  {
    return index_to_ref({ index });
  }

  template <typename T>
  T& Tensor<T>::operator()(size_t index)
  {
    return index_to_ref({ index });
  }

  template <typename T>
  const T& Tensor<T>::flat(size_t index) const
  {
    return index_to_ref(flat_to_multi_index(index, {m_shape.begin(), m_shape.end()}));
  }

  template <typename T>
  T& Tensor<T>::flat(size_t index)
  {
    return index_to_ref(flat_to_multi_index(index, {m_shape.begin(), m_shape.end()}));
  }

  template <typename T>
  template <typename U>
  requires std::is_constructible_v<T, U>
  void Tensor<T>::assign_from(const Tensor<U>& rhs)
  {
    auto target = shape();
    auto out_shape = broadcast_shapes(target, rhs.shape());
    if (out_shape != target)
    {
      throw std::invalid_argument("Cannot broadcast RHS of shape " + shape_to_string(rhs.shape()) +
          " into target of shape " + shape_to_string(target) +
          " (broadcast result would have shape " + shape_to_string(out_shape) + ")");
    }

    auto rhs_view = rhs.broadcast_to(target);

    // If the RHS aliases this tensor's storage (e.g. a[:] = a[::-1] or
    // a[1:] = a[:-1], where both sides are views of the same buffer), an
    // element-wise walk would read positions it has already overwritten.
    // Materialize an independent copy of the RHS first, matching NumPy's
    // handling of overlapping assignment. Aliasing is only possible when the
    // element types match and the two views share the same base buffer; data()
    // includes the view offset, so compare the buffer base (data() - offset()).
    if constexpr (std::is_same_v<T, U>)
    {
      if (this->data() - this->offset() == rhs.data() - rhs.offset())
      {
        Tensor<T> rhs_copy = rhs_view.copy();
        sb::walk(*this, [this, &rhs_copy](const std::vector<size_t>& idx){
          // store_copy as well: rhs_copy.copy() duplicates only the outer
          // buffer, so element types that share a buffer (point clouds, the
          // matrix types) would still alias the source until copied per cell.
          (*this)(idx) = detail::store_copy(rhs_copy(idx));
        });
        return;
      }
    }

    sb::walk(*this, [this, &rhs_view](const std::vector<size_t>& idx){
      // store_copy deep-copies element types that share a buffer (point clouds
      // and the matrix types), so a slice assignment never aliases the source.
      (*this)(idx) = detail::store_copy(T(rhs_view(idx)));
    });
  }

  template <typename T>
  template <typename U>
  requires CanDivideTo<T, T, U>
  Tensor<T>& Tensor<T>::operator/=(const U& u)
  {
    apply([&u](T& val){
      val /= u;
    });

    return *this;
  }

  template <typename T>
  template <typename U>
  requires CanDivideTo<T, T, U>
  Tensor<T> Tensor<T>::operator/(const U& u) const
  {
    Tensor<T> ret = copy();
    ret /= u;
    return ret;
  }

  template <typename T>
  template <typename U>
  requires CanMultiplyTo<T, T, U>
  Tensor<T>& Tensor<T>::operator*=(const U& u)
  {
    apply([&u](T& val){
      if constexpr (std::is_same_v<T, bool>)
        val = val && u;
      else
        val *= u;
    });

    return *this;
  }

  template <typename T>
  template <typename U>
  requires CanMultiplyTo<T, T, U>
  Tensor<T> Tensor<T>::operator*(const U& u) const
  {
    Tensor<T> ret = copy();
    ret *= u;
    return ret;
  }

  template <typename T>
  template <typename U>
  requires CanAddTo<T, T, U>
  Tensor<T>& Tensor<T>::operator+=(const U& u)
  {
    apply([&u](T& val){
      val += u;
    });

    return *this;
  }

  template <typename T>
  template <typename U>
  requires CanAddTo<T, T, U>
  Tensor<T> Tensor<T>::operator+(const U& u) const
  {
    Tensor<T> ret = copy();
    ret += u;
    return ret;
  }

  template <typename T>
  template <typename U>
  requires CanSubtractTo<T, T, U>
  Tensor<T>& Tensor<T>::operator-=(const U& u)
  {
    apply([&u](T& val){
      val -= u;
    });

    return *this;
  }

  template <typename T>
  template <typename U>
  requires CanSubtractTo<T, T, U>
  Tensor<T> Tensor<T>::operator-(const U& u) const
  {
    Tensor<T> ret = copy();
    ret -= u;
    return ret;
  }

  template <typename T>
  Tensor<T> Tensor<T>::operator-() const requires CanNegate<T>
  {
    Tensor<T> ret = copy();
    ret.apply([](T& val){
      val = -val;
    });
    return ret;
  }

  template <typename U, typename T>
  requires CanMultiplyTo<T, U, T>
  Tensor<T> operator*(const U& u, const Tensor<T>& t)
  {
    Tensor<T> ret = t.copy();
    ret.apply([&u](T& val){
      if constexpr (std::is_same_v<T, bool>)
        val = u && val;
      else
        val = u * val;
    });
    return ret;
  }

  template <typename U, typename T>
  requires CanAddTo<T, U, T>
  Tensor<T> operator+(const U& u, const Tensor<T>& t)
  {
    Tensor<T> ret = t.copy();
    ret.apply([&u](T& val){
      val = u + val;
    });
    return ret;
  }

  template <typename U, typename T>
  requires CanSubtractTo<T, U, T>
  Tensor<T> operator-(const U& u, const Tensor<T>& t)
  {
    Tensor<T> ret = t.copy();
    ret.apply([&u](T& val){
      val = u - val;
    });
    return ret;
  }

  template <typename U, typename T>
  requires CanDivideTo<T, U, T>
  Tensor<T> operator/(const U& u, const Tensor<T>& t)
  {
    Tensor<T> ret = t.copy();
    ret.apply([&u](T& val){
      val = u / val;
    });
    return ret;
  }

  /**
   * Elementwise power for tensors.
   *
   * Returns a new tensor whose elements are each raised to `exponent`
   * via `sb::pow`. Works for scalar element types (float, double)
   * and any type that provides a `.pow()` member (e.g. Pcf).
   *
   * @param t        the input tensor
   * @param exponent the exponent to raise each element to
   * @return a new tensor with transformed elements
   */
  template <typename T, typename U>
  requires CanPow<T, U>
  Tensor<T> pow(const Tensor<T>& t, U exponent)
  {
    Tensor<T> ret = t.copy();
    ret.apply([&exponent](T& val){
      val = static_cast<T>(sb::pow(val, exponent));
    });
    return ret;
  }

  /**
   * In-place elementwise power for tensors.
   *
   * Raises each element of `t` to `exponent` in place via `sb::pow`.
   *
   * @param t        the tensor to modify
   * @param exponent the exponent to raise each element to
   */
  template <typename T, typename U>
  requires CanPow<T, U>
  void ipow(Tensor<T>& t, U exponent)
  {
    t.apply([&exponent](T& val){
      val = static_cast<T>(sb::pow(val, exponent));
    });
  }

  // ============================================================================
  // Broadcasting
  // ============================================================================

  template <typename T>
  Tensor<T> Tensor<T>::broadcast_to(const std::vector<size_t>& target_shape) const
  {
    size_t ndim = target_shape.size();
    size_t src_ndim = m_shape.size();

    if (ndim < src_ndim)
    {
      throw std::invalid_argument("Cannot broadcast shape " +
          shape_to_string(m_shape) + " to " + shape_to_string(target_shape) +
          ": target has fewer dimensions");
    }

    Tensor ret;
    ret.m_data = m_data;
    ret.m_offset = m_offset;
    ret.m_shape = target_shape;
    ret.m_strides.resize(ndim);
    ret.m_isContiguous = false;

    size_t prepended = ndim - src_ndim;
    for (size_t i = 0; i < ndim; ++i)
    {
      if (i < prepended)
      {
        ret.m_strides[i] = 0;
      }
      else
      {
        size_t si = i - prepended;
        if (m_shape[si] == target_shape[i])
        {
          ret.m_strides[i] = m_strides[si];
        }
        else if (m_shape[si] == 1)
        {
          ret.m_strides[i] = 0;
        }
        else
        {
          throw std::invalid_argument("Cannot broadcast shape " +
              shape_to_string(m_shape) + " to " + shape_to_string(target_shape));
        }
      }
    }

    return ret;
  }

  // Tensor-Tensor arithmetic with broadcasting

  namespace detail
  {
    /// Apply a binary operation elementwise to two broadcast-compatible tensors.
    /// The result element type R is deduced from the return type of op.
    template <typename T, typename BinaryOp,
              typename R = std::invoke_result_t<BinaryOp, const T&, const T&>>
    Tensor<R> broadcast_binop(const Tensor<T>& lhs, const Tensor<T>& rhs, BinaryOp op)
    {
      auto out_shape = broadcast_shapes(lhs.shape(), rhs.shape());
      auto lhs_view = lhs.broadcast_to(out_shape);
      auto rhs_view = rhs.broadcast_to(out_shape);
      Tensor<R> result(out_shape);
      sb::walk(result, [&](const std::vector<size_t>& idx) {
        result(idx) = op(lhs_view(idx), rhs_view(idx));
      });
      return result;
    }

    template <typename T, typename BinaryOp>
    Tensor<T>& broadcast_binop_inplace(Tensor<T>& lhs, const Tensor<T>& rhs, BinaryOp op)
    {
      auto out_shape = broadcast_shapes(lhs.shape(), rhs.shape());
      if (out_shape != lhs.shape())
        throw std::invalid_argument("Cannot broadcast in-place: output shape " +
            shape_to_string(out_shape) + " differs from LHS shape " + shape_to_string(lhs.shape()));
      auto rhs_view = rhs.broadcast_to(out_shape);

      // If the RHS aliases the LHS storage (e.g. a[1:] += a[:-1], where both
      // sides are views of the same buffer), the walk below would read
      // elements it has already updated. Materialize an independent copy of
      // the RHS first, matching NumPy and assign_from; data() includes the
      // view offset, so compare the buffer base (data() - offset()).
      if (lhs.data() - lhs.offset() == rhs.data() - rhs.offset())
      {
        Tensor<T> rhs_copy = rhs_view.copy();
        sb::walk(lhs, [&](const std::vector<size_t>& idx) {
          lhs(idx) = op(lhs(idx), rhs_copy(idx));
        });
        return lhs;
      }

      sb::walk(lhs, [&](const std::vector<size_t>& idx) {
        lhs(idx) = op(lhs(idx), rhs_view(idx));
      });
      return lhs;
    }
  }

  template <typename T>
  Tensor<T> Tensor<T>::operator+(const Tensor& rhs) const
  { return detail::broadcast_binop(*this, rhs, [](const T& a, const T& b) -> T { return a + b; }); }

  template <typename T>
  Tensor<T> Tensor<T>::operator-(const Tensor& rhs) const
  { return detail::broadcast_binop(*this, rhs, [](const T& a, const T& b) -> T { return a - b; }); }

  template <typename T>
  Tensor<T> Tensor<T>::operator*(const Tensor& rhs) const
  {
    if constexpr (std::is_same_v<T, bool>)
      return detail::broadcast_binop(*this, rhs, [](const T& a, const T& b) -> T { return a && b; });
    else
      return detail::broadcast_binop(*this, rhs, [](const T& a, const T& b) -> T { return a * b; });
  }

  template <typename T>
  Tensor<T> Tensor<T>::operator/(const Tensor& rhs) const
  { return detail::broadcast_binop(*this, rhs, [](const T& a, const T& b) -> T { return a / b; }); }

  template <typename T>
  Tensor<T>& Tensor<T>::operator+=(const Tensor& rhs)
  { return detail::broadcast_binop_inplace(*this, rhs, [](const T& a, const T& b){ return a + b; }); }

  template <typename T>
  Tensor<T>& Tensor<T>::operator-=(const Tensor& rhs)
  { return detail::broadcast_binop_inplace(*this, rhs, [](const T& a, const T& b){ return a - b; }); }

  template <typename T>
  Tensor<T>& Tensor<T>::operator*=(const Tensor& rhs)
  { return detail::broadcast_binop_inplace(*this, rhs, [](const T& a, const T& b){ return a * b; }); }

  template <typename T>
  Tensor<T>& Tensor<T>::operator/=(const Tensor& rhs)
  { return detail::broadcast_binop_inplace(*this, rhs, [](const T& a, const T& b){ return a / b; }); }

  // Elementwise comparison with broadcasting (returns Tensor<bool>)

  template <typename T>
  requires std::equality_comparable<T>
  Tensor<bool> elementwise_eq(const Tensor<T>& lhs, const Tensor<T>& rhs)
  { return detail::broadcast_binop(lhs, rhs, [](const T& a, const T& b){ return a == b; }); }

  template <typename T>
  requires std::equality_comparable<T>
  Tensor<bool> elementwise_ne(const Tensor<T>& lhs, const Tensor<T>& rhs)
  { return detail::broadcast_binop(lhs, rhs, [](const T& a, const T& b){ return a != b; }); }

  template <typename T>
  requires CanOrder<T>
  Tensor<bool> elementwise_lt(const Tensor<T>& lhs, const Tensor<T>& rhs)
  { return detail::broadcast_binop(lhs, rhs, [](const T& a, const T& b){ return a < b; }); }

  template <typename T>
  requires CanOrder<T>
  Tensor<bool> elementwise_le(const Tensor<T>& lhs, const Tensor<T>& rhs)
  { return detail::broadcast_binop(lhs, rhs, [](const T& a, const T& b){ return a <= b; }); }

  template <typename T>
  requires CanOrder<T>
  Tensor<bool> elementwise_gt(const Tensor<T>& lhs, const Tensor<T>& rhs)
  { return detail::broadcast_binop(lhs, rhs, [](const T& a, const T& b){ return a > b; }); }

  template <typename T>
  requires CanOrder<T>
  Tensor<bool> elementwise_ge(const Tensor<T>& lhs, const Tensor<T>& rhs)
  { return detail::broadcast_binop(lhs, rhs, [](const T& a, const T& b){ return a >= b; }); }

  // ============================================================================
  // Masked operations
  // ============================================================================

  template <typename T>
  Tensor<T> masked_select(const Tensor<T>& src, const Tensor<bool>& mask)
  {
    if (mask.shape() != src.shape())
    {
      throw std::invalid_argument(
        "masked_select: mask shape does not match tensor shape");
    }

    // Single pass: collect matching elements into a buffer
    std::vector<T> buf;
    walk(src, [&](const std::vector<size_t>& idx) {
      if (mask(idx))
        buf.push_back(src(idx));
    });

    if (buf.empty())
    {
      return Tensor<T>({0});
    }

    Tensor<T> result({buf.size()});
    std::move(buf.begin(), buf.end(), result.data());

    return result;
  }

  template <typename T>
  void masked_assign(Tensor<T>& dst, const Tensor<bool>& mask, const Tensor<T>& values)
  {
    if (mask.shape() != dst.shape())
    {
      throw std::invalid_argument(
        "masked_assign: mask shape does not match tensor shape");
    }

    size_t pos = 0;
    walk(dst, [&](const std::vector<size_t>& idx) {
      if (mask(idx))
      {
        if (pos >= values.size())
        {
          throw std::invalid_argument(
            "masked_assign: more true values in mask than elements in values");
        }
        dst(idx) = values({pos++});
      }
    });

    if (pos != values.size())
    {
      throw std::invalid_argument(
        "masked_assign: values length (" + std::to_string(values.size()) +
        ") does not match number of true values in mask (" + std::to_string(pos) + ")");
    }
  }

  template <typename T>
  void masked_fill(Tensor<T>& dst, const Tensor<bool>& mask, const T& value)
  {
    if (mask.shape() != dst.shape())
    {
      throw std::invalid_argument(
        "masked_fill: mask shape does not match tensor shape");
    }

    walk(dst, [&](const std::vector<size_t>& idx) {
      if (mask(idx))
        dst(idx) = value;
    });
  }

  namespace detail
  {
    inline void validate_axis_mask(size_t axis, size_t ndim, const Tensor<bool>& mask, size_t axis_size)
    {
      if (axis >= ndim)
      {
        throw std::invalid_argument(
          "axis " + std::to_string(axis) +
          " out of range for tensor with " + std::to_string(ndim) + " dimensions");
      }
      if (mask.shape().size() != 1 || mask.shape()[0] != axis_size)
      {
        throw std::invalid_argument(
          "mask must be 1D with length matching dimension " + std::to_string(axis));
      }
    }

    inline std::vector<size_t> collect_true_indices(const Tensor<bool>& mask)
    {
      std::vector<size_t> indices;
      for (size_t i = 0; i < mask.shape()[0]; ++i)
      {
        if (mask(i))
          indices.push_back(i);
      }
      return indices;
    }
  }

  template <typename T>
  Tensor<T> axis_select(const Tensor<T>& src, size_t axis, const Tensor<bool>& mask)
  {
    detail::validate_axis_mask(axis, src.shape().size(), mask, src.shape()[axis]);
    auto true_indices = detail::collect_true_indices(mask);

    // Build output shape
    auto out_shape = src.shape();
    out_shape[axis] = true_indices.size();

    if (true_indices.empty())
    {
      return Tensor<T>(out_shape);
    }

    Tensor<T> result(out_shape);
    walk(result, [&](const std::vector<size_t>& out_idx) {
      auto src_idx = out_idx;
      src_idx[axis] = true_indices[out_idx[axis]];
      result(out_idx) = src(src_idx);
    });

    return result;
  }

  template <typename T>
  void axis_assign(Tensor<T>& dst, size_t axis, const Tensor<bool>& mask, const Tensor<T>& values)
  {
    detail::validate_axis_mask(axis, dst.shape().size(), mask, dst.shape()[axis]);
    auto true_indices = detail::collect_true_indices(mask);

    // Validate values shape
    auto expected_shape = dst.shape();
    expected_shape[axis] = true_indices.size();
    if (values.shape() != expected_shape)
    {
      throw std::invalid_argument(
        "axis_assign: values shape does not match expected shape");
    }

    // Walk over values and write into dst at mapped positions
    walk(values, [&](const std::vector<size_t>& val_idx) {
      auto dst_idx = val_idx;
      dst_idx[axis] = true_indices[val_idx[axis]];
      dst(dst_idx) = values(val_idx);
    });
  }

  template <typename T>
  void axis_fill(Tensor<T>& dst, size_t axis, const Tensor<bool>& mask, const T& value)
  {
    detail::validate_axis_mask(axis, dst.shape().size(), mask, dst.shape()[axis]);

    walk(dst, [&](const std::vector<size_t>& idx) {
      if (mask(idx[axis]))
        dst(idx) = value;
    });
  }

  namespace detail
  {
    struct AxisMaskInfo
    {
      size_t axis;
      std::vector<size_t> true_indices;
    };

    inline std::vector<AxisMaskInfo> prepare_axis_masks(
      const std::vector<std::pair<size_t, Tensor<bool>>>& axis_masks,
      const std::vector<size_t>& shape)
    {
      std::vector<AxisMaskInfo> infos;
      infos.reserve(axis_masks.size());
      for (const auto& [axis, mask] : axis_masks)
      {
        validate_axis_mask(axis, shape.size(), mask, shape[axis]);
        infos.push_back({axis, collect_true_indices(mask)});
      }
      return infos;
    }
  }

  template <typename T>
  Tensor<T> multi_axis_select(const Tensor<T>& src,
    const std::vector<std::pair<size_t, Tensor<bool>>>& axis_masks)
  {
    auto infos = detail::prepare_axis_masks(axis_masks, src.shape());

    auto out_shape = src.shape();
    for (const auto& info : infos)
      out_shape[info.axis] = info.true_indices.size();

    Tensor<T> result(out_shape);
    walk(result, [&](const std::vector<size_t>& out_idx) {
      auto src_idx = out_idx;
      for (const auto& info : infos)
        src_idx[info.axis] = info.true_indices[out_idx[info.axis]];
      result(out_idx) = src(src_idx);
    });

    return result;
  }

  template <typename T>
  void multi_axis_fill(Tensor<T>& dst,
    const std::vector<std::pair<size_t, Tensor<bool>>>& axis_masks,
    const T& value)
  {
    for (const auto& [axis, mask] : axis_masks)
      detail::validate_axis_mask(axis, dst.shape().size(), mask, dst.shape()[axis]);

    walk(dst, [&](const std::vector<size_t>& idx) {
      for (const auto& [axis, mask] : axis_masks)
      {
        if (!mask(idx[axis]))
          return;
      }
      dst(idx) = value;
    });
  }

  template <typename T>
  void multi_axis_assign(Tensor<T>& dst,
    const std::vector<std::pair<size_t, Tensor<bool>>>& axis_masks,
    const Tensor<T>& values)
  {
    auto infos = detail::prepare_axis_masks(axis_masks, dst.shape());

    // Validate the values shape against the selected (outer) shape before the
    // assignment walk, so a mismatch raises rather than writing out of bounds.
    auto expected_shape = dst.shape();
    for (const auto& info : infos)
      expected_shape[info.axis] = info.true_indices.size();
    if (values.shape() != expected_shape)
    {
      throw std::invalid_argument(
        "multi_axis_assign: values shape " + shape_to_string(values.shape()) +
        " does not match expected shape " + shape_to_string(expected_shape));
    }

    walk(values, [&](const std::vector<size_t>& val_idx) {
      auto dst_idx = val_idx;
      for (const auto& info : infos)
        dst_idx[info.axis] = info.true_indices[val_idx[info.axis]];
      dst(dst_idx) = values(val_idx);
    });
  }

  // ============================================================================
  // Generalized outer indexing (bool masks and/or int indices)
  // ============================================================================

  namespace detail
  {
    struct ResolvedSelector
    {
      size_t axis;
      std::vector<size_t> indices;  // selected positions along this axis
    };

    inline std::vector<ResolvedSelector> resolve_selectors(
      const std::vector<std::pair<size_t, AxisSelector>>& selectors,
      const std::vector<size_t>& shape)
    {
      std::vector<ResolvedSelector> result;
      result.reserve(selectors.size());
      for (const auto& [axis, sel] : selectors)
      {
        if (axis >= shape.size())
          throw std::invalid_argument("outer indexing: axis " + std::to_string(axis) +
            " out of range for tensor with " + std::to_string(shape.size()) + " dimensions");

        ResolvedSelector rs;
        rs.axis = axis;
        std::visit([&](const auto& idx_tensor) {
          using IdxT = std::decay_t<decltype(idx_tensor)>;
          if (idx_tensor.shape().size() != 1)
            throw std::invalid_argument("outer indexing: selector must be 1D");

          if constexpr (std::is_same_v<IdxT, Tensor<bool>>)
          {
            if (idx_tensor.shape()[0] != shape[axis])
              throw std::invalid_argument("outer indexing: bool mask length must match axis size");
            rs.indices = collect_true_indices(idx_tensor);
          }
          else
          {
            const auto axis_size = static_cast<ptrdiff_t>(shape[axis]);
            rs.indices.reserve(idx_tensor.shape()[0]);
            for (size_t i = 0; i < idx_tensor.shape()[0]; ++i)
            {
              // Resolve negatives against the axis size and bounds-check so
              // the C++ entry point is memory-safe regardless of caller.
              auto v = static_cast<ptrdiff_t>(idx_tensor({i}));
              if (v < 0) v += axis_size;
              if (v < 0 || v >= axis_size)
                throw std::out_of_range(
                  "index " + std::to_string(static_cast<ptrdiff_t>(idx_tensor({i}))) +
                  " is out of bounds for axis with size " + std::to_string(axis_size));
              rs.indices.push_back(static_cast<size_t>(v));
            }
          }
        }, sel);
        result.push_back(std::move(rs));
      }
      return result;
    }
  }

  template <typename T>
  Tensor<T> outer_select(const Tensor<T>& src,
    const std::vector<std::pair<size_t, AxisSelector>>& selectors)
  {
    auto resolved = detail::resolve_selectors(selectors, src.shape());

    auto out_shape = src.shape();
    for (const auto& rs : resolved)
      out_shape[rs.axis] = rs.indices.size();

    Tensor<T> result(out_shape);
    walk(result, [&](const std::vector<size_t>& out_idx) {
      auto src_idx = out_idx;
      for (const auto& rs : resolved)
        src_idx[rs.axis] = rs.indices[out_idx[rs.axis]];
      result(out_idx) = src(src_idx);
    });

    return result;
  }

  template <typename T>
  void outer_fill(Tensor<T>& dst,
    const std::vector<std::pair<size_t, AxisSelector>>& selectors,
    const T& value)
  {
    auto resolved = detail::resolve_selectors(selectors, dst.shape());

    // Build a fast lookup: for each axis with a selector, a set of selected indices
    std::vector<std::vector<bool>> axis_selected(dst.shape().size());
    for (const auto& rs : resolved)
    {
      axis_selected[rs.axis].assign(dst.shape()[rs.axis], false);
      for (size_t idx : rs.indices)
        axis_selected[rs.axis][idx] = true;
    }

    walk(dst, [&](const std::vector<size_t>& idx) {
      for (const auto& rs : resolved)
      {
        if (!axis_selected[rs.axis][idx[rs.axis]])
          return;
      }
      dst(idx) = value;
    });
  }

  template <typename T>
  void outer_assign(Tensor<T>& dst,
    const std::vector<std::pair<size_t, AxisSelector>>& selectors,
    const Tensor<T>& values)
  {
    auto resolved = detail::resolve_selectors(selectors, dst.shape());

    // Validate the values shape against the selected (outer) shape before the
    // assignment walk, so a mismatch raises rather than writing out of bounds.
    auto expected_shape = dst.shape();
    for (const auto& rs : resolved)
      expected_shape[rs.axis] = rs.indices.size();
    if (values.shape() != expected_shape)
    {
      throw std::invalid_argument(
        "outer_assign: values shape " + shape_to_string(values.shape()) +
        " does not match expected shape " + shape_to_string(expected_shape));
    }

    walk(values, [&](const std::vector<size_t>& val_idx) {
      auto dst_idx = val_idx;
      for (const auto& rs : resolved)
        dst_idx[rs.axis] = rs.indices[val_idx[rs.axis]];
      dst(dst_idx) = values(val_idx);
    });
  }

  // ============================================================================
  // Type casting
  // ============================================================================

  template <typename T, typename U>
  requires std::is_constructible_v<T, U>
  Tensor<T> tensor_cast(const Tensor<U>& src)
  {
    Tensor<T> result(src.shape());
    walk(src, [&](const std::vector<size_t>& idx) {
      result(idx) = T(src(idx));
    });
    return result;
  }

  template <typename T, typename U>
  requires std::is_constructible_v<T, U>
  Tensor<Tensor<T>> pcloud_cast(const Tensor<Tensor<U>>& src)
  {
    Tensor<Tensor<T>> result(src.shape());
    walk(src, [&](const std::vector<size_t>& idx) {
      result(idx) = tensor_cast<T>(src(idx));
    });
    return result;
  }

  // ============================================================================
  // Joining operations
  // ============================================================================

  template <typename T>
  Tensor<T> concatenate(const std::vector<Tensor<T>>& tensors, size_t axis)
  {
    if (tensors.empty())
      throw std::invalid_argument("concatenate: need at least one tensor");

    auto ndim = tensors[0].shape().size();
    if (axis >= ndim)
      throw std::invalid_argument("concatenate: axis " + std::to_string(axis) +
        " out of range for " + std::to_string(ndim) + "-D tensor");

    // Validate shapes match on all axes except the join axis
    for (size_t t = 1; t < tensors.size(); ++t)
    {
      if (tensors[t].shape().size() != ndim)
        throw std::invalid_argument("concatenate: all tensors must have the same number of dimensions");
      for (size_t d = 0; d < ndim; ++d)
      {
        if (d != axis && tensors[t].shape()[d] != tensors[0].shape()[d])
          throw std::invalid_argument("concatenate: shape mismatch on dimension " + std::to_string(d));
      }
    }

    // Compute output shape
    auto out_shape = tensors[0].shape();
    out_shape[axis] = 0;
    for (const auto& t : tensors)
      out_shape[axis] += t.shape()[axis];

    Tensor<T> result(out_shape);

    // Copy each tensor's data into the result
    size_t offset = 0;
    for (const auto& src : tensors)
    {
      auto src_axis_size = src.shape()[axis];
      walk(src, [&](const std::vector<size_t>& src_idx) {
        auto dst_idx = src_idx;
        dst_idx[axis] += offset;
        result(dst_idx) = src(src_idx);
      });
      offset += src_axis_size;
    }

    return result;
  }

  template <typename T>
  Tensor<T> stack(const std::vector<Tensor<T>>& tensors, ptrdiff_t axis)
  {
    if (tensors.empty())
      throw std::invalid_argument("stack: need at least one tensor");

    auto ndim = static_cast<ptrdiff_t>(tensors[0].shape().size());

    // Resolve negative axis; valid range for new axis is [-(ndim+1), ndim]
    if (axis < 0)
      axis += ndim + 1;
    if (axis < 0 || axis > ndim)
      throw std::invalid_argument("stack: axis out of range");

    // Validate all shapes match
    for (size_t t = 1; t < tensors.size(); ++t)
    {
      if (tensors[t].shape() != tensors[0].shape())
        throw std::invalid_argument("stack: all tensors must have the same shape");
    }

    // expand_dims each tensor, then concatenate along the new axis
    std::vector<Tensor<T>> expanded;
    expanded.reserve(tensors.size());
    for (const auto& t : tensors)
      expanded.push_back(t.expand_dims(axis));

    return concatenate(expanded, static_cast<size_t>(axis));
  }

  template <typename T>
  std::vector<Tensor<T>> split(const Tensor<T>& tensor,
    const std::vector<size_t>& split_points, size_t axis)
  {
    auto ndim = tensor.shape().size();
    if (axis >= ndim)
      throw std::invalid_argument("split: axis " + std::to_string(axis) +
        " out of range for " + std::to_string(ndim) + "-D tensor");

    auto axis_size = tensor.shape()[axis];

    // Build slice boundaries: [0, sp0, sp1, ..., axis_size]
    std::vector<size_t> boundaries;
    boundaries.reserve(split_points.size() + 2);
    boundaries.push_back(0);
    for (auto sp : split_points)
    {
      if (sp > axis_size)
        sp = axis_size;
      boundaries.push_back(sp);
    }
    boundaries.push_back(axis_size);

    std::vector<Tensor<T>> result;
    result.reserve(boundaries.size() - 1);
    for (size_t i = 0; i + 1 < boundaries.size(); ++i)
    {
      std::vector<Slice> slices(ndim, all());
      slices[axis] = range(
        static_cast<ptrdiff_t>(boundaries[i]),
        static_cast<ptrdiff_t>(boundaries[i + 1]),
        std::nullopt);
      result.push_back(tensor[slices]);
    }
    return result;
  }

  template <typename T>
  std::vector<Tensor<T>> split(const Tensor<T>& tensor,
    size_t n_sections, size_t axis)
  {
    auto ndim = tensor.shape().size();
    if (axis >= ndim)
      throw std::invalid_argument("split: axis " + std::to_string(axis) +
        " out of range for " + std::to_string(ndim) + "-D tensor");

    if (n_sections == 0)
      throw std::invalid_argument("split: number of sections must be > 0");

    auto axis_size = tensor.shape()[axis];
    if (axis_size % n_sections != 0)
      throw std::invalid_argument("split: tensor of size " +
        std::to_string(axis_size) + " along axis " + std::to_string(axis) +
        " cannot be evenly split into " + std::to_string(n_sections) + " sections");

    auto section_size = axis_size / n_sections;
    std::vector<size_t> split_points;
    split_points.reserve(n_sections - 1);
    for (size_t i = 1; i < n_sections; ++i)
      split_points.push_back(i * section_size);

    return split(tensor, split_points, axis);
  }

  template <typename T>
  std::vector<Tensor<T>> array_split(const Tensor<T>& tensor,
    size_t n_sections, size_t axis)
  {
    auto ndim = tensor.shape().size();
    if (axis >= ndim)
      throw std::invalid_argument("array_split: axis " + std::to_string(axis) +
        " out of range for " + std::to_string(ndim) + "-D tensor");
    if (n_sections == 0)
      throw std::invalid_argument("array_split: number of sections must be > 0");

    auto axis_size = tensor.shape()[axis];
    auto base_size = axis_size / n_sections;
    auto remainder = axis_size % n_sections;

    std::vector<size_t> split_points;
    split_points.reserve(n_sections - 1);
    size_t pos = 0;
    for (size_t i = 0; i + 1 < n_sections; ++i)
    {
      pos += base_size + (i < remainder ? 1 : 0);
      split_points.push_back(pos);
    }

    return split(tensor, split_points, axis);
  }

  namespace detail
  {
    template <typename I>
    void validate_axis_indices(size_t axis, size_t ndim, const Tensor<I>& indices, size_t axis_size)
    {
      if (axis >= ndim)
      {
        throw std::invalid_argument(
          "axis " + std::to_string(axis) +
          " out of range for tensor with " + std::to_string(ndim) + " dimensions");
      }
      if (indices.shape().size() != 1)
      {
        throw std::invalid_argument("indices must be 1D");
      }
      for (size_t i = 0; i < indices.shape()[0]; ++i)
      {
        auto idx = static_cast<size_t>(indices({i}));
        if (idx >= axis_size)
        {
          throw std::out_of_range(
            "index " + std::to_string(indices({i})) +
            " is out of bounds for axis with size " + std::to_string(axis_size));
        }
      }
    }
  }

  template <typename T, typename I>
  Tensor<T> index_select(const Tensor<T>& src, size_t axis, const Tensor<I>& indices)
  {
    detail::validate_axis_indices(axis, src.shape().size(), indices, src.shape()[axis]);

    auto out_shape = src.shape();
    out_shape[axis] = indices.shape()[0];

    if (indices.shape()[0] == 0)
    {
      return Tensor<T>(out_shape);
    }

    Tensor<T> result(out_shape);
    walk(result, [&](const std::vector<size_t>& out_idx) {
      auto src_idx = out_idx;
      src_idx[axis] = static_cast<size_t>(indices({out_idx[axis]}));
      result(out_idx) = src(src_idx);
    });

    return result;
  }

  template <typename T, typename I>
  void index_assign(Tensor<T>& dst, size_t axis, const Tensor<I>& indices, const Tensor<T>& values)
  {
    detail::validate_axis_indices(axis, dst.shape().size(), indices, dst.shape()[axis]);

    auto expected_shape = dst.shape();
    expected_shape[axis] = indices.shape()[0];
    if (values.shape() != expected_shape)
    {
      throw std::invalid_argument(
        "index_assign: values shape does not match expected shape");
    }

    walk(values, [&](const std::vector<size_t>& val_idx) {
      auto dst_idx = val_idx;
      dst_idx[axis] = static_cast<size_t>(indices({val_idx[axis]}));
      dst(dst_idx) = values(val_idx);
    });
  }

  template <typename T, typename I>
  void index_fill(Tensor<T>& dst, size_t axis, const Tensor<I>& indices, const T& value)
  {
    detail::validate_axis_indices(axis, dst.shape().size(), indices, dst.shape()[axis]);

    for (size_t i = 0; i < indices.shape()[0]; ++i)
    {
      auto target = static_cast<size_t>(indices({i}));
      // Walk over all positions along the other axes
      walk(dst, [&](const std::vector<size_t>& idx) {
        if (idx[axis] == target)
          dst(idx) = value;
      });
    }
  }

  template <typename T>
  [[nodiscard]] size_t Tensor<T>::size() const noexcept
  {
    if (m_shape.empty())
    {
      return 0_uz;
    }
    return std::accumulate(m_shape.begin(), m_shape.end(), 1_uz, std::multiplies<size_t>());
  }

  template <typename T>
  Tensor<T> Tensor<T>::copy() const
  {
    Tensor<T> ret(shape());

    sb::walk(*this, [&ret, this](const std::vector<size_t>& idx){
      ret(idx) = (*this)(idx);
    });

    return ret;
  }

  template <typename T>
  Tensor<T> Tensor<T>::flatten() const
  {
    Tensor ret;
    if (m_isContiguous)
    {
      ret = *this;
    }
    else
    {
      ret = copy();
    }

    ret.m_viewType = ViewType::Flattened;
    ret.m_shape = { get_total_size() };
    ret.m_strides = { ptrdiff_t{1} };
    return ret;
  }

  template <typename T>
  Tensor<T> Tensor<T>::reshape(const std::vector<ptrdiff_t>& new_shape) const
  {
    auto total = get_total_size();

    // Resolve -1 dimension
    std::vector<size_t> resolved(new_shape.size());
    ptrdiff_t infer_idx = -1;
    size_t known_product = 1;
    for (size_t i = 0; i < new_shape.size(); ++i)
    {
      if (new_shape[i] == -1)
      {
        if (infer_idx >= 0)
          throw std::invalid_argument("Only one dimension can be inferred (-1)");
        infer_idx = static_cast<ptrdiff_t>(i);
      }
      else if (new_shape[i] < 0)
      {
        throw std::invalid_argument("Invalid dimension size: " + std::to_string(new_shape[i]));
      }
      else
      {
        resolved[i] = static_cast<size_t>(new_shape[i]);
        known_product *= resolved[i];
      }
    }

    if (infer_idx >= 0)
    {
      if (known_product == 0 || total % known_product != 0)
        throw std::invalid_argument("Cannot infer dimension: total size " +
          std::to_string(total) + " is not divisible by " + std::to_string(known_product));
      resolved[infer_idx] = total / known_product;
    }

    // Validate total size
    size_t new_total = 1;
    for (auto d : resolved) new_total *= d;
    if (new_total != total)
      throw std::invalid_argument("Cannot reshape tensor of size " +
        std::to_string(total) + " into shape " + shape_to_string(resolved));

    // Use contiguous source (copy if needed)
    Tensor<T> src = m_isContiguous ? *this : copy();

    Tensor<T> ret;
    ret.m_data = src.m_data;
    ret.m_offset = src.m_offset;
    ret.m_shape = resolved;
    ret.m_isContiguous = true;
    ret.m_viewType = ViewType::Base;

    // Compute row-major strides
    ret.m_strides.resize(resolved.size());
    if (!resolved.empty())
    {
      ret.m_strides.back() = 1;
      for (auto i = static_cast<ptrdiff_t>(resolved.size()) - 2; i >= 0; --i)
        ret.m_strides[i] = ret.m_strides[i + 1] * static_cast<ptrdiff_t>(resolved[i + 1]);
    }

    return ret;
  }

  template <typename T>
  Tensor<T> Tensor<T>::squeeze() const
  {
    Tensor<T> ret;
    ret.m_data = m_data;
    ret.m_offset = m_offset;
    ret.m_isContiguous = m_isContiguous;
    ret.m_viewType = ViewType::Base;

    for (size_t i = 0; i < m_shape.size(); ++i)
    {
      if (m_shape[i] != 1)
      {
        ret.m_shape.push_back(m_shape[i]);
        ret.m_strides.push_back(m_strides[i]);
      }
    }

    return ret;
  }

  template <typename T>
  Tensor<T> Tensor<T>::squeeze(size_t axis) const
  {
    if (axis >= m_shape.size())
      throw std::invalid_argument("squeeze: axis " + std::to_string(axis) +
        " out of range for tensor with " + std::to_string(m_shape.size()) + " dimensions");
    if (m_shape[axis] != 1)
      throw std::invalid_argument("squeeze: cannot squeeze axis " + std::to_string(axis) +
        " with size " + std::to_string(m_shape[axis]));

    Tensor<T> ret;
    ret.m_data = m_data;
    ret.m_offset = m_offset;
    ret.m_isContiguous = m_isContiguous;
    ret.m_viewType = ViewType::Base;

    for (size_t i = 0; i < m_shape.size(); ++i)
    {
      if (i != axis)
      {
        ret.m_shape.push_back(m_shape[i]);
        ret.m_strides.push_back(m_strides[i]);
      }
    }

    return ret;
  }

  template <typename T>
  Tensor<T> Tensor<T>::expand_dims(ptrdiff_t axis) const
  {
    auto ndim = static_cast<ptrdiff_t>(m_shape.size());
    // Resolve negative axis; valid range is [-(ndim+1), ndim]
    if (axis < 0)
      axis += ndim + 1;
    if (axis < 0 || axis > ndim)
      throw std::invalid_argument("expand_dims: axis " + std::to_string(axis) +
        " out of range for tensor with " + std::to_string(ndim) + " dimensions");

    auto pos = static_cast<size_t>(axis);

    Tensor<T> ret;
    ret.m_data = m_data;
    ret.m_offset = m_offset;
    ret.m_isContiguous = m_isContiguous;
    ret.m_viewType = ViewType::Base;
    ret.m_shape = m_shape;
    ret.m_strides = m_strides;
    ret.m_shape.insert(ret.m_shape.begin() + pos, 1);
    ret.m_strides.insert(ret.m_strides.begin() + pos, pos < m_strides.size() ? m_strides[pos] : 1);

    return ret;
  }

  template <typename T>
  Tensor<T> Tensor<T>::transpose(const std::vector<size_t>& axes) const
  {
    auto ndim = m_shape.size();

    std::vector<size_t> perm;
    if (axes.empty())
    {
      perm.resize(ndim);
      std::iota(perm.rbegin(), perm.rend(), 0_uz);
    }
    else
    {
      if (axes.size() != ndim)
        throw std::invalid_argument("transpose: axes must have length " +
          std::to_string(ndim) + ", got " + std::to_string(axes.size()));

      std::vector<bool> seen(ndim, false);
      for (auto a : axes)
      {
        if (a >= ndim)
          throw std::invalid_argument("transpose: axis " + std::to_string(a) + " out of range");
        if (seen[a])
          throw std::invalid_argument("transpose: repeated axis " + std::to_string(a));
        seen[a] = true;
      }
      perm = axes;
    }

    Tensor<T> ret;
    ret.m_data = m_data;
    ret.m_offset = m_offset;
    ret.m_shape.resize(ndim);
    ret.m_strides.resize(ndim);
    for (size_t i = 0; i < ndim; ++i)
    {
      ret.m_shape[i] = m_shape[perm[i]];
      ret.m_strides[i] = m_strides[perm[i]];
    }
    ret.m_isContiguous = false;
    ret.m_viewType = ViewType::Base;

    return ret;
  }

  template <typename T>
  Tensor<T> Tensor<T>::swapaxes(size_t axis1, size_t axis2) const
  {
    auto ndim = m_shape.size();
    if (axis1 >= ndim || axis2 >= ndim)
      throw std::invalid_argument("swapaxes: axis out of range for tensor with " +
        std::to_string(ndim) + " dimensions");

    std::vector<size_t> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0_uz);
    std::swap(perm[axis1], perm[axis2]);
    return transpose(perm);
  }

  template <typename T>
  template <typename UnaryFunc>
#ifndef __CUDACC__
  requires std::invocable<UnaryFunc, const T&>
#endif
  bool Tensor<T>::any_of(UnaryFunc&& f) const
  {
    return sb::any_of(*this, std::forward<UnaryFunc>(f));
  }

  template <typename T>
  template <typename UnaryFunc>
#ifndef __CUDACC__
  requires std::invocable<UnaryFunc, std::vector<size_t>>
#endif
  bool Tensor<T>::any_of_idx(UnaryFunc&& f) const
  {
    bool match = false;

    sb::walk(*this, [&match, &f](const std::vector<size_t>& idx) {

      if (f(idx))
      {
        match = true;
        return false;
      }

      return true;
    });

    return match;

  }

  template <typename T>
  bool Tensor<T>::allclose(const Tensor& rhs, T atol, T rtol) const requires FloatType<T>
  {
    return sb::allclose(*this, rhs, atol, rtol);
  }

  template <typename T>
  template <typename UnaryFunc> requires std::invocable<UnaryFunc, T&>
  void Tensor<T>::apply(UnaryFunc&& f)
  {
    sb::walk(*this, [&f, this](const std::vector<size_t>& idx){ f((*this)(idx)); });
  }

  template <typename T>
  template <typename SliceVector>
  Tensor<T> Tensor<T>::extract(SliceVector sliceVector) const
  {
    Tensor ret;

    ret.m_data = m_data; // view

    // temporary just to get sizes
    ret.m_shape = m_shape;
    ret.m_strides = m_strides;

    ret.m_offset = m_offset;

    std::vector<size_t> dimsToDrop;

    // For now, we'll drop the assumption that the tensor is contiguous in memory
    // as soon as we extract a subtensor. There are, however, some cases where the resulting subtensor would be
    // contiguous that we can try to optimize for in the future (e.g., extracting the top n rows of a matrix).
    // Things will work fine with this assumption dropped but certain operations could be a little slower (probably
    // unlikely to matter for the type of things we're targeting).
    ret.m_isContiguous = false;

    size_t i = 0;
    for (auto & slice : sliceVector)
    {
      std::visit([i, &ret, &dimsToDrop](auto&& arg) {
        using argT = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<argT, SliceIndex>)
        {
          // Resolve a (possibly negative) single integer index against the
          // dimension size and bounds-check it, mirroring NumPy semantics.
          // Throwing here (rather than computing an out-of-buffer offset)
          // turns silent OOB reads / SIGSEGV-on-write into a clean IndexError.
          auto dim_size = static_cast<ptrdiff_t>(ret.m_shape[i]);
          auto ix = arg.index;
          if (ix < 0) ix += dim_size;
          if (ix < 0 || ix >= dim_size)
          {
            throw std::out_of_range(
              "index " + std::to_string(arg.index) +
              " is out of bounds for axis " + std::to_string(i) +
              " with size " + std::to_string(dim_size));
          }
          ret.m_shape[i] = 1;
          ret.m_offset += ix * ret.m_strides[i];
          dimsToDrop.emplace_back(i);
        }
        else if constexpr (std::is_same_v<argT, SliceRange>)
        {
          if (!arg.step)
          {
            arg.step = 1_z;
          }

          auto step = *arg.step;
          if (step == 0)
          {
            throw std::invalid_argument("slice step cannot be zero");
          }

          auto dim_size = static_cast<ptrdiff_t>(ret.m_shape[i]);

          // Mirror Python's slice.indices(n): resolve negative bounds by
          // adding dim_size, then clamp into the valid range. For a positive
          // step the range is [0, n]; for a negative step it is [-1, n-1].
          const ptrdiff_t lower = (step < 0) ? -1_z : 0_z;
          const ptrdiff_t upper = (step < 0) ? dim_size - 1_z : dim_size;

          ptrdiff_t start;
          if (!arg.start)
          {
            start = (step < 0) ? upper : lower;
          }
          else
          {
            start = *arg.start;
            if (start < 0)
            {
              start += dim_size;
              if (start < lower) start = lower;
            }
            else if (start > upper)
            {
              start = upper;
            }
          }

          ptrdiff_t stop;
          if (!arg.stop)
          {
            stop = (step < 0) ? lower : upper;
          }
          else
          {
            stop = *arg.stop;
            if (stop < 0)
            {
              stop += dim_size;
              if (stop < lower) stop = lower;
            }
            else if (stop > upper)
            {
              stop = upper;
            }
          }

          ptrdiff_t len;
          if (step > 0)
          {
            len = (start < stop) ? (stop - start - 1_z) / step + 1_z : 0_z;
          }
          else
          {
            len = (start > stop) ? (start - stop - 1_z) / (-step) + 1_z : 0_z;
          }

          ret.m_shape[i] = static_cast<size_t>(len);
          ret.m_offset += start * ret.m_strides[i];
          ret.m_strides[i] *= step;
        }
        // For SliceAll, don't modify shape
      }, slice);
      ++i;
    }

    size_t nDroppedDims = 0_uz;
    for (auto dim : dimsToDrop)
    {
      ret.m_shape.erase(ret.m_shape.begin() + dim - nDroppedDims);
      ret.m_strides.erase(ret.m_strides.begin() + dim - nDroppedDims);
      ++nDroppedDims;
    }

    return ret;
  }


  template <typename T>
  size_t Tensor<T>::get_total_size() const
  {
    return std::accumulate(m_shape.begin(), m_shape.end(), 1_uz, std::multiplies<>());
  }

  template <typename T>
  ptrdiff_t Tensor<T>::index_to_data_index(const std::vector<size_t>& index) const
  {
    ptrdiff_t ret = 0;
    switch (m_viewType)
    {
    case ViewType::Base:
      ret = std::inner_product(index.begin(), index.end(), m_strides.begin(), ptrdiff_t{0},
        std::plus<>(), [](size_t idx, ptrdiff_t stride) -> ptrdiff_t {
          return static_cast<ptrdiff_t>(idx) * stride;
        });
      ret += m_offset;
      return ret;
    case ViewType::Flattened:
      if (index.size() != 1_uz)
      {
        throw std::runtime_error("Index into flat tensor should be 1d.");
      }

      if (m_isContiguous)
      {
        return m_offset + static_cast<ptrdiff_t>(index[0]);
      }

      return ret;
    }

    throw std::runtime_error("Unhandled view type!");
  }

  template <typename T>
  const T& Tensor<T>::index_to_ref(const std::vector<size_t>& index) const
  {
    return m_data[index_to_data_index(index)];
  }

  template <typename T>
  T& Tensor<T>::index_to_ref(const std::vector<size_t>& index)
  {
    return m_data[index_to_data_index(index)];
  }

  template <typename T, typename UnaryPred>
#ifndef __CUDACC__
  requires std::invocable<UnaryPred, const T&>
#endif
  bool any_of(const Tensor<T>& tensor, UnaryPred&& pred)
  {
    bool found = false;
    walk(tensor, [&found, &tensor, &pred](const std::vector<size_t>& idx) {
      found = pred(tensor(idx));
      return !found;
    });
    return found;
  }

  template <typename T, typename UnaryPred>
#ifndef __CUDACC__
  requires std::invocable<UnaryPred, const T&>
#endif
  bool all_of(const Tensor<T>& tensor, UnaryPred&& pred)
  {
    bool result = true;
    walk(tensor, [&result, &tensor, &pred](const std::vector<size_t>& idx) {
      return (result = pred(tensor(idx)));
    });
    return result;
  }

  // Follows the numpy convention: |a - b| <= atol + rtol * |b|
  template <FloatType T>
  bool allclose(const Tensor<T>& a, const Tensor<T>& b, T atol, T rtol)
  {
    if (a.shape() != b.shape())
      return false;

    bool close = true;
    walk(a, [&close, &a, &b, atol, rtol](const std::vector<size_t>& idx) {
      auto aval = a(idx);
      auto bval = b(idx);
      return (close = (std::abs(aval - bval) <= atol + rtol * std::abs(bval)));
    });
    return close;
  }

  template <typename MatT>
  requires is_compressed_matrix_v<MatT>
  bool allclose(const MatT& a, const MatT& b,
                typename MatT::value_type atol, typename MatT::value_type rtol)
  {
    if (a.storage_count() != b.storage_count())
      return false;

    for (size_t i = 0; i < a.storage_count(); ++i)
    {
      auto aval = a.data()[i];
      auto bval = b.data()[i];
      if (std::abs(aval - bval) > atol + rtol * std::abs(bval))
        return false;
    }
    return true;
  }

  inline std::vector<size_t> flat_to_multi_index(size_t flat, const std::vector<size_t>& shape)
  {
    auto ndim = shape.size();
    std::vector<size_t> idx(ndim);
    for (ptrdiff_t i = ndim - 1; i >= 0; --i)
    {
      idx[i] = flat % shape[i];
      flat /= shape[i];
    }
    return idx;
  }
}
