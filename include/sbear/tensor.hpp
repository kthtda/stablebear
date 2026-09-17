#ifndef STABLEBEAR_TENSOR_H
#define STABLEBEAR_TENSOR_H

#include <memory>
#include <vector>
#include <variant>
#include <numeric>
#include <algorithm>
#include <concepts>
#include <optional>
#include <stdexcept>
#include <sstream>

#include <iostream>

#include <taskflow/core/taskflow.hpp>

#include "config.hpp"

namespace sb
{
  class Executor;

  struct SliceAll { };

  struct SliceIndex
  {
    ptrdiff_t index;
  };

  struct SliceRange
  {
    std::optional<ptrdiff_t> start;
    std::optional<ptrdiff_t> stop;
    std::optional<ptrdiff_t> step;
  };

  using Slice = std::variant<SliceIndex, SliceRange>;

  [[nodiscard]] inline Slice all() { return Slice{SliceRange{}}; }

  [[nodiscard]] inline Slice index(ptrdiff_t index)
  {
    return Slice{SliceIndex{ .index = index }};
  }

  [[nodiscard]] inline Slice range(std::optional<ptrdiff_t> start, std::optional<ptrdiff_t> stop, std::optional<ptrdiff_t> step)
  {
    return Slice{SliceRange{ .start = start, .stop = stop, .step = step }};
  }

  template <typename T>
  class Tensor
  {
  public:
    using value_type = T;

    enum class ViewType
    {
      Base,           // Normal indexing (no reshaping, etc.)
      Flattened       // Flattened view (1-d indexing)
    };

    explicit Tensor(const std::vector<size_t>& shape, const T& init = {});
    Tensor() : Tensor({}, {}) { }

    /// Assign val to every element of the Tensor
    Tensor& operator=(const T& val);

    /**
     * Elementwise equality comparison
     * @tparam U value type of the compared with tensor (must be `equality_comparable_with` `value_type`)
     * @param rhs tensor to compare against
     * @return `true` if all elements are equal
     */
    template <typename U> requires std::equality_comparable_with<U, T>
    bool operator==(const Tensor<U>& rhs) const;

    /**
     * Elementwise (non)equality comparison
     * @tparam U value type of the compared with tensor (must be `equality_comparable_with` `value_type`)
     * @param rhs tensor to compare against
     * @return `true` if at least one element differs
     */
    template <typename U> requires std::equality_comparable_with<U, T>
    bool operator!=(const Tensor<U>& rhs) const;

    bool allclose(const Tensor& rhs, T atol = T(1e-8), T rtol = T(1e-5)) const requires FloatType<T>;

    template <typename U>
    requires std::is_constructible_v<T, U>
    void assign_from(const Tensor<U>& rhs);

    template <typename U>
    requires CanDivideTo<T, T, U>
    Tensor& operator/=(const U& u);

    template <typename U>
    requires CanDivideTo<T, T, U>
    [[nodiscard]] Tensor operator/(const U& u) const;

    template <typename U>
    requires CanMultiplyTo<T, T, U>
    Tensor& operator*=(const U& u);

    template <typename U>
    requires CanMultiplyTo<T, T, U>
    [[nodiscard]] Tensor operator*(const U& u) const;

    template <typename U>
    requires CanAddTo<T, T, U>
    Tensor& operator+=(const U& u);

    template <typename U>
    requires CanAddTo<T, T, U>
    [[nodiscard]] Tensor operator+(const U& u) const;

    template <typename U>
    requires CanSubtractTo<T, T, U>
    Tensor& operator-=(const U& u);

    template <typename U>
    requires CanSubtractTo<T, T, U>
    [[nodiscard]] Tensor operator-(const U& u) const;

    [[nodiscard]] Tensor operator-() const requires CanNegate<T>;

    /**
     * Return a broadcast view of this tensor with the given target shape.
     * Dimensions of size 1 are expanded (stride set to 0); prepended dimensions also get stride 0.
     * No data is copied — the result shares the underlying storage.
     */
    [[nodiscard]] Tensor broadcast_to(const std::vector<size_t>& target_shape) const;

    /**
     * Elementwise tensor-tensor arithmetic with NumPy-style broadcasting.
     *
     * The two operands are broadcast to a common shape before the operation.
     * A new tensor with the broadcast output shape is returned; neither operand is modified.
     *
     * @param rhs right-hand operand (must be broadcast-compatible with `*this`)
     * @return new tensor containing the elementwise result
     * @throws std::invalid_argument if shapes are not broadcast-compatible
     */
    [[nodiscard]] Tensor operator+(const Tensor& rhs) const;
    [[nodiscard]] Tensor operator-(const Tensor& rhs) const;
    [[nodiscard]] Tensor operator*(const Tensor& rhs) const;
    [[nodiscard]] Tensor operator/(const Tensor& rhs) const;

    /**
     * In-place elementwise tensor-tensor arithmetic with broadcasting.
     *
     * The right-hand operand is broadcast to the shape of `*this`. The broadcast output shape
     * must equal the shape of `*this` (i.e. `*this` is never expanded), matching NumPy semantics.
     *
     * @param rhs right-hand operand (must be broadcast-compatible without expanding `*this`)
     * @return reference to the modified `*this` tensor
     * @throws std::invalid_argument if the broadcast output shape differs from the LHS shape
     */
    Tensor& operator+=(const Tensor& rhs);
    Tensor& operator-=(const Tensor& rhs);
    Tensor& operator*=(const Tensor& rhs);
    Tensor& operator/=(const Tensor& rhs);

    [[nodiscard]] const std::vector<ptrdiff_t>& strides() const noexcept { return m_strides; }
    [[nodiscard]] ptrdiff_t stride(size_t idx) const noexcept { return m_strides[idx]; }
    [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return m_shape; }
    [[nodiscard]] size_t shape(size_t dim) const noexcept { return m_shape[dim]; }
    [[nodiscard]] size_t rank() const noexcept { return m_shape.size(); }

    /**
     * Compute the total number of elements in the tensor.
     * @return Total number of elements in the tensor.
     */
    [[nodiscard]] size_t size() const noexcept;

    [[nodiscard]] bool is_contiguous() const noexcept { return m_isContiguous; }

    [[nodiscard]] ptrdiff_t offset() const noexcept { return m_offset; }
    [[nodiscard]] value_type* data() const noexcept { return m_data.get() + m_offset; }

    template <typename SliceVector>
    [[nodiscard]] Tensor operator[](SliceVector sliceVector) const;

    /// Direct element access
    [[nodiscard]] const T& operator()(const std::vector<size_t>& index) const;
    [[nodiscard]] T& operator()(const std::vector<size_t>& index);
    // Direct element access (1d)
    [[nodiscard]] const T& operator()(size_t index) const;
    [[nodiscard]] T& operator()(size_t index);

    /// Flat (linear) element access — treats the tensor as if flattened in
    /// row-major order.  Works for any tensor, including sliced views.
    [[nodiscard]] const T& flat(size_t index) const;
    [[nodiscard]] T& flat(size_t index);

    //T& operator()(const std::vector<size_t>& index);
    //const T& operator()(const std::vector<size_t>& index) const;

    Tensor flatten() const;

    /**
     * Return a tensor with the given shape, sharing data when contiguous.
     * One dimension may be -1 to infer its size from the total element count.
     * If the tensor is not contiguous, a contiguous copy is made first.
     */
    [[nodiscard]] Tensor reshape(const std::vector<ptrdiff_t>& new_shape) const;

    /**
     * Return a view with axes permuted. If axes is empty, reverses all axes (like NumPy .T).
     */
    [[nodiscard]] Tensor transpose(const std::vector<size_t>& axes = {}) const;

    /**
     * Return a view with two axes swapped.
     */
    [[nodiscard]] Tensor swapaxes(size_t axis1, size_t axis2) const;

    /**
     * Remove all size-1 dimensions. Returns a view.
     */
    [[nodiscard]] Tensor squeeze() const;

    /**
     * Remove a specific size-1 dimension. Raises if its size is not 1. Returns a view.
     */
    [[nodiscard]] Tensor squeeze(size_t axis) const;

    /**
     * Insert a size-1 dimension at the given axis position. Supports negative
     * indexing. Returns a view.
     */
    [[nodiscard]] Tensor expand_dims(ptrdiff_t axis) const;

    /**
     * Make a deep copy of the tensor. The new tensor will be a contiguous version of the original tensor.
     * @return Deep copy of the tensor
     */
    Tensor copy() const;

    template <typename UnaryPred>
#ifndef __CUDACC__
    requires std::invocable<UnaryPred, const T&>
#endif
    bool any_of(UnaryPred&& f) const;

    template <typename UnaryPred>
#ifndef __CUDACC__
    requires std::invocable<UnaryPred, std::vector<size_t>>
#endif
    bool any_of_idx(UnaryPred&& f) const;

    /**
     * Apply a function at each element of the tensor (uses `walk` internally to visit the elements)
     * @tparam UnaryFunc Function object of type `T&` -> `void` (non-`void` return values get discarded)
     * @param f The function object to invoke at each index
     */
    template <typename UnaryFunc>
    requires std::invocable<UnaryFunc, T&>
    void apply(UnaryFunc&& f);

    class AxisIterator
    {
    public:
      AxisIterator(Tensor* tensor, size_t dim, ptrdiff_t pos)
        : m_tensor(tensor), m_dim(dim)
      {
        m_sliceVector.resize(tensor->shape().size(), SliceAll());
        m_sliceVector[dim] = SliceIndex{ pos };
      }

      Tensor operator*() const
      {
        return (*m_tensor)[m_sliceVector];
      }

      AxisIterator& operator++()
      {
        ++std::get<SliceIndex>(m_sliceVector[m_dim]).index;
        return *this;
      }

    private:
      std::vector<Slice> m_sliceVector;
      Tensor* m_tensor;
      size_t m_dim = 0_uz;
    };

  private:
    template <typename SliceVector>
    [[nodiscard]] Tensor extract(SliceVector sliceVector) const;

    [[nodiscard]] size_t get_total_size() const;

    [[nodiscard]] ptrdiff_t index_to_data_index(const std::vector<size_t>& index) const;
    [[nodiscard]] const T& index_to_ref(const std::vector<size_t>& index) const;
    [[nodiscard]] T& index_to_ref(const std::vector<size_t>& index);

    std::vector<ptrdiff_t> m_strides;
    std::vector<size_t> m_shape;
    std::shared_ptr<value_type[]> m_data;
    ptrdiff_t m_offset = 0;

    ViewType m_viewType = ViewType::Base;
    bool m_isContiguous = true;
  };

  template <typename U, typename T>
  requires CanMultiplyTo<T, U, T>
  [[nodiscard]] Tensor<T> operator*(const U& u, const Tensor<T>& t);

  template <typename U, typename T>
  requires CanAddTo<T, U, T>
  [[nodiscard]] Tensor<T> operator+(const U& u, const Tensor<T>& t);

  template <typename U, typename T>
  requires CanSubtractTo<T, U, T>
  [[nodiscard]] Tensor<T> operator-(const U& u, const Tensor<T>& t);

  template <typename U, typename T>
  requires CanDivideTo<T, U, T>
  [[nodiscard]] Tensor<T> operator/(const U& u, const Tensor<T>& t);

  template <typename IntT>
  inline std::string shape_to_string(const std::vector<IntT>& shape)
  {
    std::stringstream ss;
    ss << "(";
    for (auto i = 0_uz; i < shape.size(); ++i)
    {
      if (i != 0)
      {
        ss << ", ";
      }
      ss << shape[i];
    }
    ss << ")";
    return ss.str();
  }

  template <typename IntT>
  inline std::string index_to_string(const std::vector<IntT>& idx)
  {
    if (idx.size() == 1)
    {
      return std::to_string(idx[0]);
    }
    else
    {
      return shape_to_string(idx);
    }
  }

  /**
   * Compute the broadcast-compatible output shape from two input shapes (NumPy rules).
   *
   * Shapes are compared right-to-left. Dimensions match if they are equal or one of them is 1.
   * Missing leading dimensions are treated as size 1.
   *
   * @param a shape of the first operand
   * @param b shape of the second operand
   * @return the broadcast output shape
   * @throws std::invalid_argument if the shapes are not broadcast-compatible
   */
  inline std::vector<size_t> broadcast_shapes(
    const std::vector<size_t>& a,
    const std::vector<size_t>& b)
  {
    size_t ndim = std::max(a.size(), b.size());
    std::vector<size_t> result(ndim);

    for (size_t i = 0; i < ndim; ++i)
    {
      size_t da = (i < ndim - a.size()) ? 1 : a[i - (ndim - a.size())];
      size_t db = (i < ndim - b.size()) ? 1 : b[i - (ndim - b.size())];

      if (da == db)
        result[i] = da;
      else if (da == 1)
        result[i] = db;
      else if (db == 1)
        result[i] = da;
      else
        throw std::invalid_argument("Shapes are not broadcast-compatible: " +
            shape_to_string(a) + " and " + shape_to_string(b));
    }
    return result;
  }

  namespace detail
  {
    // Deep-copy a value about to be stored in a tensor cell, so the cell never
    // aliases the source. Element types that hold a shared buffer (Tensor /
    // PointCloud, DistanceMatrix, SymmetricMatrix) expose a deep copy() that is
    // preferred; all other element types (float, Pcf, ...) already copy deeply
    // by value, so they are returned unchanged.
    template <typename T>
    [[nodiscard]] T store_copy(const T& v)
    {
      if constexpr (requires { v.copy(); })
        return v.copy();
      else
        return v;
    }
  }

  template <typename T, typename UnaryPred>
#ifndef __CUDACC__
  requires std::invocable<UnaryPred, const T&>
#endif
  bool any_of(const Tensor<T>& tensor, UnaryPred&& pred);

  template <typename T, typename UnaryPred>
#ifndef __CUDACC__
  requires std::invocable<UnaryPred, const T&>
#endif
  bool all_of(const Tensor<T>& tensor, UnaryPred&& pred);

  template <FloatType T>
  bool allclose(const Tensor<T>& a, const Tensor<T>& b, T atol = T(1e-8), T rtol = T(1e-5));

  // Follows the numpy convention: |a - b| <= atol + rtol * |b|
  template <typename MatT>
  requires is_compressed_matrix_v<MatT>
  bool allclose(const MatT& a, const MatT& b,
                typename MatT::value_type atol = typename MatT::value_type(1e-8),
                typename MatT::value_type rtol = typename MatT::value_type(1e-5));

  /// Convert a flat (linear) index to a multi-dimensional index for the given shape.
  std::vector<size_t> flat_to_multi_index(size_t flat, const std::vector<size_t>& shape);

  // ============================================================================
  // Masked operations
  // ============================================================================

  /**
   * Select elements from src where mask is true, returning a 1D tensor.
   * Mask shape must match src.shape(). Elements are collected in row-major order.
   */
  template <typename T>
  [[nodiscard]] Tensor<T> masked_select(const Tensor<T>& src, const Tensor<bool>& mask);

  /**
   * Assign values into dst at positions where mask is true.
   * Mask shape must match dst.shape(). values must be 1D with length == count of true values.
   */
  template <typename T>
  void masked_assign(Tensor<T>& dst, const Tensor<bool>& mask, const Tensor<T>& values);

  /**
   * Fill dst with a scalar value at positions where mask is true.
   * Mask shape must match dst.shape().
   */
  template <typename T>
  void masked_fill(Tensor<T>& dst, const Tensor<bool>& mask, const T& value);

  /**
   * Select along a single axis where mask is true.
   * mask must be 1D with length == src.shape()[axis].
   * Returns a tensor with shape[axis] reduced to count of true values.
   */
  template <typename T>
  [[nodiscard]] Tensor<T> axis_select(const Tensor<T>& src, size_t axis, const Tensor<bool>& mask);

  /**
   * Assign values into dst along a single axis where mask is true.
   * mask must be 1D with length == dst.shape()[axis].
   * values.shape() must match the shape that axis_select(dst, axis, mask) would produce.
   */
  template <typename T>
  void axis_assign(Tensor<T>& dst, size_t axis, const Tensor<bool>& mask, const Tensor<T>& values);

  /**
   * Fill dst with a scalar value along a single axis where mask is true.
   * mask must be 1D with length == dst.shape()[axis].
   */
  template <typename T>
  void axis_fill(Tensor<T>& dst, size_t axis, const Tensor<bool>& mask, const T& value);

  /**
   * Select along multiple axes where each mask is true (outer indexing).
   * Each pair is (axis, mask). Masks are applied independently per axis.
   */
  template <typename T>
  [[nodiscard]] Tensor<T> multi_axis_select(const Tensor<T>& src,
    const std::vector<std::pair<size_t, Tensor<bool>>>& axis_masks);

  /**
   * Fill dst with a scalar value at positions where all masks are true (outer indexing).
   * Each pair is (axis, mask). A position is filled if mask_i[idx[axis_i]] is true for every i.
   */
  template <typename T>
  void multi_axis_fill(Tensor<T>& dst,
    const std::vector<std::pair<size_t, Tensor<bool>>>& axis_masks,
    const T& value);

  /**
   * Assign values into dst at positions where all masks are true (outer indexing).
   * Each pair is (axis, mask). values.shape() must match the multi_axis_select output shape.
   */
  template <typename T>
  void multi_axis_assign(Tensor<T>& dst,
    const std::vector<std::pair<size_t, Tensor<bool>>>& axis_masks,
    const Tensor<T>& values);

  // ============================================================================
  // Generalized multi-axis operations (bool masks and/or int indices)
  // ============================================================================

  /// An axis selector: either a boolean mask or an integer index array.
  using AxisSelector = std::variant<Tensor<bool>, Tensor<int64_t>>;

  /**
   * Select along multiple axes using any mix of bool masks and int index arrays
   * (outer indexing). Each pair is (axis, selector).
   */
  template <typename T>
  [[nodiscard]] Tensor<T> outer_select(const Tensor<T>& src,
    const std::vector<std::pair<size_t, AxisSelector>>& selectors);

  /**
   * Fill dst with a scalar at positions selected by outer indexing.
   */
  template <typename T>
  void outer_fill(Tensor<T>& dst,
    const std::vector<std::pair<size_t, AxisSelector>>& selectors,
    const T& value);

  /**
   * Assign values into dst at positions selected by outer indexing.
   * values.shape() must match the outer_select output shape.
   */
  template <typename T>
  void outer_assign(Tensor<T>& dst,
    const std::vector<std::pair<size_t, AxisSelector>>& selectors,
    const Tensor<T>& values);

  // ============================================================================
  // Type casting
  // ============================================================================

  /**
   * Create a new tensor of type Tensor<T> by converting each element from Tensor<U>.
   * Requires T to be constructible from U.
   */
  template <typename T, typename U>
  requires std::is_constructible_v<T, U>
  [[nodiscard]] Tensor<T> tensor_cast(const Tensor<U>& src);

  // ============================================================================
  // Joining operations
  // ============================================================================

  /**
   * Concatenate tensors along an existing axis.
   * All tensors must have the same shape except along the join axis.
   */
  template <typename T>
  [[nodiscard]] Tensor<T> concatenate(const std::vector<Tensor<T>>& tensors, size_t axis);

  /**
   * Stack tensors along a new axis.
   * All tensors must have the same shape. Supports negative axis.
   */
  template <typename T>
  [[nodiscard]] Tensor<T> stack(const std::vector<Tensor<T>>& tensors, ptrdiff_t axis);

  /**
   * Split a tensor into sub-tensors along an axis.
   * indices_or_sections is either a single count (equal splits) or
   * a list of split points (like NumPy). Returns views.
   */
  template <typename T>
  [[nodiscard]] std::vector<Tensor<T>> split(const Tensor<T>& tensor,
    const std::vector<size_t>& split_points, size_t axis);

  template <typename T>
  [[nodiscard]] std::vector<Tensor<T>> split(const Tensor<T>& tensor,
    size_t n_sections, size_t axis);

  /**
   * Split a tensor into n_sections parts, allowing uneven splits.
   * The first (axis_size % n_sections) parts get one extra element.
   */
  template <typename T>
  [[nodiscard]] std::vector<Tensor<T>> array_split(const Tensor<T>& tensor,
    size_t n_sections, size_t axis);

  // ============================================================================
  // Index-based gather/scatter operations
  // ============================================================================

  /**
   * Gather elements from src along a single axis using integer indices.
   * indices must be 1D. Returns a tensor with shape[axis] == indices.size().
   */
  template <typename T, typename I>
  [[nodiscard]] Tensor<T> index_select(const Tensor<T>& src, size_t axis, const Tensor<I>& indices);

  /**
   * Scatter values into dst along a single axis at integer indices.
   * indices must be 1D. values.shape() must match the shape that
   * index_select(dst, axis, indices) would produce.
   */
  template <typename T, typename I>
  void index_assign(Tensor<T>& dst, size_t axis, const Tensor<I>& indices, const Tensor<T>& values);

  /**
   * Fill dst with a scalar value along a single axis at integer indices.
   * indices must be 1D.
   */
  template <typename T, typename I>
  void index_fill(Tensor<T>& dst, size_t axis, const Tensor<I>& indices, const T& value);

}

#include "tensor.tpp"

#include "detail/tensor_1d_value_iterator.hpp"

#endif //STABLEBEAR_TENSOR_H
