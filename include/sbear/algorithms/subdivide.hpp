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
