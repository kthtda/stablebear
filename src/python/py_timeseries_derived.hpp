#ifndef STABLEBEAR_PY_TIMESERIES_DERIVED_H
#define STABLEBEAR_PY_TIMESERIES_DERIVED_H

#include "nb.hpp"

namespace sb_py
{
  /// Bind the functions that derive one series from another, such as delay
  /// embeddings, as module functions overloaded on the series type.
  void register_timeseries_derived(nb::module_& m);
}

#endif // STABLEBEAR_PY_TIMESERIES_DERIVED_H
