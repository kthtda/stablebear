#ifndef SB_ALGORITHMS_FUNCTIONAL_LP_DISTANCE_H
#define SB_ALGORITHMS_FUNCTIONAL_LP_DISTANCE_H

#include <sbear/functional/pcf.hpp>
#include <sbear/functional/operations.cuh>
#include <sbear/algorithms/functional/matrix_integrate.hpp>

namespace sb
{
  template <typename Tt, typename Tv>
  Tv lp_distance(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g, Tv p = Tv(1))
  {
    if (p == Tv(1))
    {
      return integrate(f, g, OperationL1Dist<Tt, Tv>{});
    }

    return integrate(f, g, OperationLpDist<Tt, Tv>(p));
  }

}

#endif
