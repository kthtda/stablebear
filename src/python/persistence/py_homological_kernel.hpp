#ifndef STABLEBEAR_PY_HOMOLOGICAL_KERNEL_H
#define STABLEBEAR_PY_HOMOLOGICAL_KERNEL_H

#include "../pybind.hpp"

namespace sb_py
{
  void register_persistence_homological_kernel(pybind11::module_ &m);
}

#endif // STABLEBEAR_PY_HOMOLOGICAL_KERNEL_H
