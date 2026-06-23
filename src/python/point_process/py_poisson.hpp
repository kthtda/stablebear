#ifndef STABLEBEAR_PY_POISSON_H
#define STABLEBEAR_PY_POISSON_H

#include <pybind11/pybind11.h>

namespace sb_py
{
  void register_point_process_poisson(pybind11::module_& m);
}

#endif
