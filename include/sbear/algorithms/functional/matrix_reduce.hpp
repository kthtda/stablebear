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
#include "reduce.hpp"

#include <iostream>
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

  template <typename PcfT, typename ReductionF>
  Tensor<PcfT> parallel_tensor_reduce(
      const Tensor<PcfT>& in,
      size_t dim,
      ReductionF&& reduction,
      Executor& exec = default_executor())
  {
    auto shape = in.shape();
    auto inDimSize = shape[dim];

    shape.erase(shape.begin() + dim);
    if (shape.empty())
    {
      shape.resize(1, 1);
    }

    Tensor<PcfT> ret(shape);
    auto op = std::forward<ReductionF>(reduction);
    sb::parallel_walk(ret, [&ret, &in, inDimSize, dim, &op](const std::vector<size_t>& idx){
      std::vector<size_t> inIdx(in.shape().size(), 0_uz);

      std::copy(idx.begin(), idx.begin() + dim, inIdx.begin());
      if (inIdx.size() > 1)
      {
        // MSVC debug does not like (inIdx.begin() + dim + 1) even if nothing is written there it seems.
        std::copy(idx.begin() + dim, idx.end(), inIdx.begin() + dim + 1);
      }

      std::vector<PcfT> tmp; // TODO: Rewrite without need to copy
      tmp.reserve(inDimSize);
      for (auto i = 0_uz; i < inDimSize; ++i)
      {
        inIdx[dim] = i;
        tmp.emplace_back( in(inIdx) );
      }

      auto & out = ret(idx);

      out = reduce(tmp, op);
    }, exec);

    return ret;
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
    using OutTensorT = Tensor<std::decay_t<
        decltype(f(std::declval<PcfT>()))
    >>;

    using OutValueT = typename OutTensorT::value_type;

    static_assert(std::invocable<MaxOp, OutValueT, OutValueT>);

    auto shape = in.shape();
    auto inDimSize = shape[dim];

    shape.erase(shape.begin() + dim);
    if (shape.empty())
    {
      shape.resize(1, 1);
    }
    OutTensorT ret(shape);

    sb::parallel_walk(ret, [&ret, &in, inDimSize, dim, &f, &maxOp](const std::vector<size_t>& idx){
      std::vector<size_t> inIdx(in.shape().size(), 0_uz);

      std::copy(idx.begin(), idx.begin() + dim, inIdx.begin());
      if (inIdx.size() > 1)
      {
        // MSVC debug does not like (inIdx.begin() + dim + 1) even if nothing is written there it seems.
        std::copy(idx.begin() + dim, idx.end(), inIdx.begin() + dim + 1);
      }

      std::vector<PcfT> tmp; // TODO: Rewrite without need to copy
      tmp.reserve(inDimSize);
      for (auto i = 0_uz; i < inDimSize; ++i)
      {
        inIdx[dim] = i;
        tmp.emplace_back( in(inIdx) );
      }

      auto & out = ret(idx);

      auto init = f(*tmp.begin());
      out =  std::transform_reduce(tmp.begin() + 1, tmp.end(), init, maxOp, f);

    }, exec);

    return ret;
  }
}

#endif
