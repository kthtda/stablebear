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

#ifndef SB_ALGORITHMS_MATRIX_REDUCE_H
#define SB_ALGORITHMS_MATRIX_REDUCE_H

#include "../../functional/pcf.hpp"
#include "../../tensor.hpp"
#include "../../executor.hpp"
#include "../../detail/tensor_1d_value_iterator.hpp"
#include "reduce.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace sb
{
  template <typename T>
  inline void printVec(const char* n, const std::vector<T>& v) {
    std::cout << n << " : ";
    for (auto i : v)
    {
      std::cout << ", " << i;
    }
    std::cout << std::endl;
  };

  // Validate a reduction dimension against a tensor shape. An out-of-range dim
  // would otherwise index/erase past the end of the shape vector (UB: wrong
  // result for dim == ndim, heap corruption for dim > ndim). Throwing here
  // surfaces a clean, catchable exception (pybind11 maps std::out_of_range to
  // a Python IndexError).
  inline void check_reduce_dim(size_t dim, size_t ndim)
  {
    if (dim >= ndim)
    {
      std::ostringstream oss;
      oss << "Reduction dimension " << dim << " is out of range for tensor of dimension " << ndim;
      throw std::out_of_range(oss.str());
    }
  }

  // Shared skeleton for every dimension reduction: build the reduced-shape
  // output tensor, then for each output cell reduce the input slice along `dim`
  // in place (via a strided iterator -- no per-slice copy). Callers supply the
  // output element type `OutT` and a `reduction` functor mapping the slice
  // [first, last) to a single OutT value.
  template <typename OutT, typename PcfT, typename ReductionF>
  Tensor<OutT> parallel_dim_reduce(
      const Tensor<PcfT>& in,
      size_t dim,
      ReductionF&& reduction,
      Executor& exec = default_executor())
  {
    auto shape = in.shape();
    check_reduce_dim(dim, shape.size());
    auto inDimSize = shape[dim];

    shape.erase(shape.begin() + dim);
    if (shape.empty())
    {
      shape.resize(1, 1);
    }

    Tensor<OutT> ret(shape);
    auto op = std::forward<ReductionF>(reduction);
    sb::parallel_walk(ret, [&ret, &in, inDimSize, dim, &op](const std::vector<size_t>& idx){
      std::vector<size_t> inIdx(in.shape().size(), 0_uz);

      std::copy(idx.begin(), idx.begin() + dim, inIdx.begin());
      if (inIdx.size() > 1)
      {
        // MSVC debug does not like (inIdx.begin() + dim + 1) even if nothing is written there it seems.
        std::copy(idx.begin() + dim, idx.end(), inIdx.begin() + dim + 1);
      }

      // inIdx[dim] == 0 marks the origin of the slice; the iterator walks `dim`.
      Tensor1dValueIterator<const Tensor<PcfT>> first(in, inIdx, dim, 0_uz);
      Tensor1dValueIterator<const Tensor<PcfT>> last(in, inIdx, dim, inDimSize);

      ret(idx) = op(first, last);
    }, exec);

    return ret;
  }

  template <typename PcfT, typename ReductionF>
  Tensor<PcfT> parallel_tensor_reduce(
      const Tensor<PcfT>& in,
      size_t dim,
      ReductionF&& reduction,
      Executor& exec = default_executor())
  {
    auto op = std::forward<ReductionF>(reduction);
    return parallel_dim_reduce<PcfT>(in, dim, [op](auto first, auto last) {
      return reduce(first, last, op);
    }, exec);
  }

  template <typename PcfT>
  Tensor<PcfT> mean(const Tensor<PcfT>& in, size_t dim, Executor& exec = default_executor())
  {
    auto ret = parallel_tensor_reduce(in, dim, [](const typename PcfT::rectangle_type& rect) {
      return rect.f_value + rect.g_value;
    }, exec);

    const auto inDimSize = in.shape()[dim];
    ret /= inDimSize;

    return ret;
  }

  template <typename PcfT, typename UnaryF, typename MaxOp>
  auto max_element(const Tensor<PcfT>& in, size_t dim, UnaryF&& f, MaxOp&& maxOp, Executor& exec = default_executor())
  {
    using OutValueT = std::decay_t<decltype(f(std::declval<PcfT>()))>;

    static_assert(std::invocable<MaxOp, OutValueT, OutValueT>);

    // max has no identity element, so it cannot reduce an empty range: the
    // body below seeds from *first and reduces [first + 1, last), both UB when
    // the slice is empty (inDimSize == 0). Reject up front with a clean,
    // catchable exception (pybind11 maps std::invalid_argument to a Python
    // ValueError) instead of dereferencing a past-the-end iterator -- mirrors
    // numpy's "zero-size array to reduction operation" error. Validate dim
    // first so shape[dim] is in range.
    check_reduce_dim(dim, in.shape().size());
    if (in.shape()[dim] == 0)
    {
      std::ostringstream oss;
      oss << "Cannot reduce an empty dimension: dimension " << dim << " has size 0";
      throw std::invalid_argument(oss.str());
    }

    auto fn = std::forward<UnaryF>(f);
    auto mop = std::forward<MaxOp>(maxOp);
    return parallel_dim_reduce<OutValueT>(in, dim, [fn, mop](auto first, auto last) {
      auto init = fn(*first);
      return std::transform_reduce(first + 1, last, init, mop, fn);
    }, exec);
  }
}

#endif
