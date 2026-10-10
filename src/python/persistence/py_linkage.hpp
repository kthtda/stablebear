#ifndef STABLEBEAR_PY_LINKAGE_H
#define STABLEBEAR_PY_LINKAGE_H

#include "../pybind.hpp"

namespace sb_py
{
  void register_persistence_linkage(pybind11::module_& m);
}

#endif // STABLEBEAR_PY_LINKAGE_H
