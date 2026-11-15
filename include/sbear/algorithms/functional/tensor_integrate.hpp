#ifndef SB_TENSOR_INTEGRATE_H
#define SB_TENSOR_INTEGRATE_H

#include "../../functional/plf.hpp"
#include "../../tensor.hpp"
#include "../../executor.hpp"
#include "../../walk.hpp"

#include <vector>

namespace sb
{
  /// Integrate every PCF in a tensor, giving a tensor of PLFs of the same
  /// shape. Throws std::invalid_argument if a PCF's times are not strictly
  /// increasing.
  template <FloatType Tt, FloatType Tv>
  Tensor<Plf<Tt, Tv>> integrate(const Tensor<Pcf<Tt, Tv>>& pcfs, Executor& exec = default_executor())
  {
    Tensor<Plf<Tt, Tv>> out(pcfs.shape());
    parallel_walk(pcfs, [&pcfs, &out](const std::vector<size_t>& idx) {
      out(idx) = integrate(pcfs(idx));
    }, exec);
    return out;
  }
}

#endif
