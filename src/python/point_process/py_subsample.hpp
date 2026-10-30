#ifndef STABLEBEAR_PY_POINT_PROCESS_SUBSAMPLE_H
#define STABLEBEAR_PY_POINT_PROCESS_SUBSAMPLE_H

#include "../nb.hpp"

namespace sb_py
{
  void register_point_process_subsample(nb::module_& m);
  void register_relative_subsample(nb::module_& m);
}

#endif
