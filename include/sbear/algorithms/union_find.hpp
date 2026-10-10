#ifndef SB_ALGORITHMS_UNION_FIND_H
#define SB_ALGORITHMS_UNION_FIND_H

#include <cstddef>
#include <numeric>
#include <vector>

namespace sb
{
  /// Disjoint-set forest over the elements 0..n-1 with path halving (the
  /// standard one-pass variant of path compression).
  ///
  /// unite() is directed: the caller passes the two roots and picks which one
  /// survives. This is load-bearing for callers that cache roots across
  /// unions (e.g. the homological-kernel cross filtration): a cached root can
  /// only stop being a root when a union writes over its slot, so knowing
  /// which root was absorbed lets the caller patch its caches exactly.
  ///
  /// Copy assignment reuses the target's capacity (plain vector copy
  /// assignment), so repeatedly re-seeding a scratch instance from a fixed
  /// seed allocates nothing in the steady state.
  class UnionFind
  {
  public:
    /// n singleton components.
    explicit UnionFind(size_t n) : m_parent(n)
    {
      std::iota(m_parent.begin(), m_parent.end(), size_t{0});
    }

    /// Root of x's component, compressing the path along the way (standard
    /// "path halving": every visited node is re-pointed to its grandparent).
    [[nodiscard]] size_t find(size_t x)
    {
      while (m_parent[x] != x)
      {
        m_parent[x] = m_parent[m_parent[x]];
        x = m_parent[x];
      }
      return x;
    }

    /// Merge two components given their roots: survivor stays a root, absorbed
    /// is attached beneath it. unite(r, r) is a harmless self-assignment, so
    /// unconditional unions of already-connected roots are fine.
    void unite(size_t survivor, size_t absorbed)
    {
      m_parent[absorbed] = survivor;
    }

  private:
    std::vector<size_t> m_parent;
  };
} // namespace sb

#endif
