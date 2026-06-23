#ifndef STABLEBEAR_PY_SUBSAMPLE_H
#define STABLEBEAR_PY_SUBSAMPLE_H

#include <pybind11/pybind11.h>

namespace sb_py
{
  void register_sampling_subsample(pybind11::module_& m);
}

#endif
