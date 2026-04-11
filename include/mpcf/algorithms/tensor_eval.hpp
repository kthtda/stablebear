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

#ifndef MPCF_TENSOR_EVAL_H
#define MPCF_TENSOR_EVAL_H

#include "../tensor.hpp"
#include "../concepts.hpp"
#include "../executor.hpp"
#include "../settings.hpp"
#include "../walk.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace mpcf
{

  // Evaluate every element of a tensor at a single point, writing results into out.
  // out must have the same shape as elems.
  template <typename DomainT, typename CodomainT, IsTensor TElemTensor, IsTensor TOutTensor>
  requires Evaluable<typename TElemTensor::value_type, DomainT, CodomainT>
  void tensor_eval(const TElemTensor& elems, DomainT x, TOutTensor& out)
  {
    auto eval = [&](const std::vector<size_t>& idx) {
      out(idx) = elems(idx).evaluate(x);
    };

    if (elems.size() >= settings().parallelEvalThreshold)
      parallel_walk(elems, eval, default_executor());
    else
      walk(elems, eval);
  }

  // Evaluate every element of a tensor at a tensor of domain points, writing results into out.
  // out must have shape elems.shape() + domain.shape().
  // Domain points are sorted once; each element uses the linear-scan evaluate overload.
  template <typename DomainT, typename CodomainT, IsTensor TElemTensor, IsTensor TDomainTensor, IsTensor TOutTensor>
  requires Evaluable<typename TElemTensor::value_type, DomainT, CodomainT>
  void tensor_eval(const TElemTensor& elems, const TDomainTensor& domain, TOutTensor& out)
  {
    // Collect domain points and their multi-dim indices in walk (row-major) order
    std::vector<DomainT> domain_values;
    std::vector<std::vector<size_t>> domain_indices;
    auto n = domain.size();
    domain_values.reserve(n);
    domain_indices.reserve(n);
    walk(domain, [&](const std::vector<size_t>& idx) {
      domain_values.push_back(domain(idx));
      domain_indices.push_back(idx);
    });

    // Sort once by domain value
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return domain_values[a] < domain_values[b];
    });

    Tensor<DomainT> sorted_domain(std::vector<size_t>{n});
    for (size_t i = 0; i < n; ++i)
      sorted_domain(std::vector<size_t>{i}) = domain_values[order[i]];

    // Evaluate each element with the linear-scan overload, then unsort.
    // Each element gets its own sorted_result buffer for thread safety.
    auto eval_elem = [&](const std::vector<size_t>& elem_idx) {
      Tensor<CodomainT> sorted_result(std::vector<size_t>{n});
      elems(elem_idx).evaluate(sorted_domain, sorted_result, n);

      auto combined_idx = elem_idx;
      combined_idx.resize(elem_idx.size() + domain_indices[0].size());

      for (size_t i = 0; i < n; ++i) {
        const auto& d_idx = domain_indices[order[i]];
        std::copy(d_idx.begin(), d_idx.end(), combined_idx.begin() + elem_idx.size());
        out(combined_idx) = sorted_result(std::vector<size_t>{i});
      }
    };

    if (elems.size() * n >= settings().parallelEvalThreshold)
      parallel_walk(elems, eval_elem, default_executor());
    else
      walk(elems, eval_elem);
  }

} // namespace mpcf

#endif // MPCF_TENSOR_EVAL_H
