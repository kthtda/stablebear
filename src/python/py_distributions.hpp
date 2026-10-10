#ifndef STABLEBEAR_PY_DISTRIBUTIONS_H
#define STABLEBEAR_PY_DISTRIBUTIONS_H

#include <pybind11/pybind11.h>

namespace sb_py
{
  void register_distributions(pybind11::module_& m);
}

#endif
