#ifndef STABLEBEAR_PY_NP_TENSOR_CONVERT_H
#define STABLEBEAR_PY_NP_TENSOR_CONVERT_H

#include "pybind.hpp"

namespace sb_py
{
  void register_np_conversions(pybind11::module_& m);
}


#endif //STABLEBEAR_PY_NP_TENSOR_CONVERT_H