#ifndef STABLEBEAR_PY_RIPSER_H
#define STABLEBEAR_PY_RIPSER_H

#include "../pybind.hpp"

namespace sb_py
{
  void register_persistence_ripser(pybind11::module_& m);
}

#endif //STABLEBEAR_PY_RIPSER_H