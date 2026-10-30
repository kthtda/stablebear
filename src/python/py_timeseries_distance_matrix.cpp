#include "py_timeseries_bind.hpp"

#include <sbear/distance_matrix.hpp>

namespace sb_py
{
  void register_timeseries_distance_matrix(nb::module_& m)
  {
    register_typed_timeseries<sb::DistanceMatrix<sb::float32_t>>(m, "DistanceMatrix32");
    register_typed_timeseries<sb::DistanceMatrix<sb::float64_t>>(m, "DistanceMatrix64");
  }
}
