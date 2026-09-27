#ifndef STABLEBEAR_PY_TENSOR_PROPERTY_VARIANTS_H
#define STABLEBEAR_PY_TENSOR_PROPERTY_VARIANTS_H

#include <sbear/tensor.hpp>

namespace sb_py
{
  /// Instantiate a binding callback for ordinary and indexed tensor storage.
  template <typename Bind>
  void bind_tensor_property_variants(Bind&& bind)
  {
    bind.template operator()<sb::TensorProperty::None>();
    bind.template operator()<sb::TensorProperty::Indexed>();
  }

  /// Instantiate a binary binding callback for every ordinary/indexed pair.
  template <typename Bind>
  void bind_tensor_property_pairs(Bind&& bind)
  {
    bind.template operator()<
      sb::TensorProperty::None, sb::TensorProperty::None>();
    bind.template operator()<
      sb::TensorProperty::None, sb::TensorProperty::Indexed>();
    bind.template operator()<
      sb::TensorProperty::Indexed, sb::TensorProperty::None>();
    bind.template operator()<
      sb::TensorProperty::Indexed, sb::TensorProperty::Indexed>();
  }
}

#endif // STABLEBEAR_PY_TENSOR_PROPERTY_VARIANTS_H
