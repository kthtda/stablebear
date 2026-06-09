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

#ifndef SB_ALGORITHMS_SUBDIVIDE_H
#define SB_ALGORITHMS_SUBDIVIDE_H

#include <vector>
#include <utility>
#include <algorithm>

namespace sb
{
  inline std::vector<std::pair<size_t, size_t>> 
  subdivide(size_t blockSize, size_t nItems)
  {
    std::vector<std::pair<size_t, size_t>> boundaries;
    for (size_t i = 0ul;; i += blockSize)
    {
      if (!boundaries.empty() && boundaries.back().second == nItems - 1)
      {
        return boundaries;
      }
      
      boundaries.emplace_back(i, std::min<std::size_t>(i + blockSize - std::size_t(1), nItems));
      if (boundaries.back().second >= nItems)
      {
        boundaries.back().second = nItems - 1;
        return boundaries;
      }
    }
  }
}

#endif
