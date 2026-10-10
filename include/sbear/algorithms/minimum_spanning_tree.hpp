#ifndef SB_ALGORITHMS_MINIMUM_SPANNING_TREE_H
#define SB_ALGORITHMS_MINIMUM_SPANNING_TREE_H

#include "../concepts.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace sb
{
  template <typename T>
  struct MergeEdge
  {
    size_t a; // endpoints of the MST edge = representatives of the two merging components
    size_t b;
    T mergeDist;
  };

  // Fills merges with the n-1 single-linkage merges of the metric on the
  // n = dist.size() points, sorted ascending by merge distance. Array-based
  // Prim: O(n^2) distance queries, O(n) memory.
  template <typename DistT, typename T>
  requires DistanceOracle<DistT, T>
  void mst_merge_order(const DistT &dist, std::vector<MergeEdge<T>> &merges)
  {
    const auto n = dist.size();
    merges.clear();
    if (n <= 1)
    {
      return;
    }
    merges.reserve(n - 1);

    std::vector<T> minDist(n, std::numeric_limits<T>::infinity());
    std::vector<size_t> parent(n, 0);
    std::vector<bool> visited(n, false);

    minDist[0] = T{0};

    for (size_t iter = 0; iter < n; ++iter)
    {
      // Pick the unvisited node closest to the tree.
      size_t nearest = 0;
      bool found = false;
      for (size_t j = 0; j < n; ++j)
      {
        if (!visited[j] && (!found || minDist[j] < minDist[nearest]))
        {
          nearest = j;
          found = true;
        }
      }

      visited[nearest] = true;
      if (iter > 0)
      {
        merges.push_back({parent[nearest], nearest, minDist[nearest]});
      }

      for (size_t j = 0; j < n; ++j)
      {
        if (!visited[j])
        {
          auto candidateDist = dist(nearest, j);
          if (candidateDist < minDist[j])
          {
            minDist[j] = candidateDist;
            parent[j] = nearest;
          }
        }
      }
    }

    std::sort(merges.begin(), merges.end(), [](const MergeEdge<T> &lhs, const MergeEdge<T> &rhs) {
      return lhs.mergeDist < rhs.mergeDist;
    });
  }
} // namespace sb

#endif
