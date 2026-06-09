/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

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
      OperationL1Dist<Tt, Tv> op;
      return op(integrate(f, g, op));
    }

    OperationLpDist<Tt, Tv> op(p);
    return op(integrate(f, g, op));
  }

}

#endif
