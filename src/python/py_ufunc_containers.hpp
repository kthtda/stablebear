#ifndef STABLEBEAR_PY_UFUNC_CONTAINERS_H
#define STABLEBEAR_PY_UFUNC_CONTAINERS_H

#include <sbear/tensor.hpp>
#include <sbear/timeseries.hpp>

#include <string>

namespace sb_py
{
  /// Instantiate a binding callback for every container an operation over
  /// elements of type ElemT accepts: a tensor, an indexed tensor when ElemT
  /// supports selections, and a time series. The callback gets the container
  /// as a template argument and a name suffix for the bound task class.
  template <typename ElemT, typename Bind>
  void bind_ufunc_containers(Bind&& bind)
  {
    bind.template operator()<sb::Tensor<ElemT>>(std::string());
    if constexpr (sb::IndexableTensorElement<ElemT>)
    {
      bind.template operator()<sb::Tensor<ElemT, sb::TensorProperty::Indexed>>(std::string("Indexed"));
    }
    bind.template operator()<sb::TimeSeries<ElemT>>(std::string("Series"));
  }
}

#endif // STABLEBEAR_PY_UFUNC_CONTAINERS_H
