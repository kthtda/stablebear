#ifndef STABLEBEAR_PY_POINT_CLOUD_H
#define STABLEBEAR_PY_POINT_CLOUD_H

#include "pybind.hpp"

namespace sb_py
{
  void register_point_cloud_bindings(pybind11::module_& m);
}

#endif // STABLEBEAR_PY_POINT_CLOUD_H
