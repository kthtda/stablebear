#ifndef STABLEBEAR_HOMOLOGICAL_KERNEL_H
#define STABLEBEAR_HOMOLOGICAL_KERNEL_H

#include "../algorithms/minimum_spanning_tree.hpp"
#include "../algorithms/union_find.hpp"
#include "../concepts.hpp"
#include "../distance_matrix.hpp"
#include "../distances.hpp"
#include "../executor.hpp"
#include "../task.hpp"
#include "../tensor.hpp"
#include "../walk.hpp"
#include "barcode.hpp"

#include "taskflow/core/taskflow.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sb::ph
{
  namespace detail
  {
    // Performs the cross filtration: replays the d' merges (the births w_i) and
    // computes each death v_i against d.
    //
    // The death of merge i is the scale at which [a_i] and [b_i] connect in the
    // quotient of (X, d) by the i-1 earlier contractions — the subdominant ultra
    // pseudo-metric of the quotient space. A contraction can create a shortcut
    // through the contracted component that lowers the connection scale of
    // other pairs, so each death depends on the whole set of earlier
    // contractions, not just the merging pair.
    //
    // Connectivity at scale t of (X, d) equals connectivity of the d-MST
    // restricted to edges <= t, and contractions only add identifications on
    // top. So each death is answered by a Kruskal sweep: seed a union-find with
    // the current d'-components, add d-MST edges in ascending order, and stop
    // as soon as a_i ~ b_i; the weight of the connecting edge is the death.
    //
    // The sweep for merge i stops after rank(v_i) edges, and the deaths are a
    // permutation of the d-MST edge ranks, so all sweeps together perform
    // exactly n(n-1)/2 bounded union-find operations plus an O(n) seed copy per
    // merge: O(n^2 alpha(n)) time, O(n) memory. Throws if some death lands
    // below its birth (beyond roundoff), which means d does not dominate d'.
    template <typename T>
    void compute_cross_filtration(
        const std::vector<MergeEdge<T>> &primeMerges, const std::vector<MergeEdge<T>> &dMerges,
        std::vector<PersistencePair<T>> &bars)
    {
      bars.clear();
      if (primeMerges.empty())
      {
        return;
      }
      // Exactly n-1 merges for n points is an invariant the callers establish
      // (both merge vectors come from mst_merge_order), so the point count is
      // recoverable from the merge count.
      const size_t n = primeMerges.size() + 1;
      bars.reserve(primeMerges.size());

      // primeComponents holds the components induced by the contractions
      // performed so far (= the current d'-components); dComponents is the
      // working copy each death query runs on, adding d-MST edges on top of
      // those contractions. The copy assignment reuses capacity, so the
      // steady state allocates nothing.
      UnionFind primeComponents(n);
      UnionFind dComponents(n);

      for (auto const &merge : primeMerges)
      {
        dComponents = primeComponents;

        // The d'-components alone can never connect merge.a to merge.b (a and
        // b are in different ones by definition of this merge), and the full
        // d-MST is spanning — so the loop always runs at least once and exits
        // with edgeIdx <= n-1, having set death.
        //
        // The connectivity check is find-free: rootA/rootB are cached, and a
        // cached root can only stop being one when a union absorbs it
        // (absorbedRoot below) — so patching on that case keeps both caches
        // exact.
        auto rootA = dComponents.find(merge.a);
        auto rootB = dComponents.find(merge.b);
        auto death = std::numeric_limits<T>::quiet_NaN();
        size_t edgeIdx = 0;
        while (rootA != rootB)
        {
          auto const &edge = dMerges[edgeIdx++];
          // Unconditional union: an edge whose endpoints are already connected
          // (through the d'-contractions or earlier edges) is a harmless
          // self-union.
          auto survivingRoot = dComponents.find(edge.a);
          auto absorbedRoot = dComponents.find(edge.b);
          dComponents.unite(survivingRoot, absorbedRoot);
          if (absorbedRoot == rootA)
          {
            rootA = survivingRoot;
          }
          else if (absorbedRoot == rootB)
          {
            rootB = survivingRoot;
          }
          death = edge.mergeDist;
        }

        // Births (from d') and deaths (from d) can reach mathematically equal
        // values through different arithmetic, and inputs materialized at large
        // coordinate magnitude (e.g. projected point clouds) carry absolute
        // error ~eps * |coordinate|, which can dwarf a few ULPs of the
        // distances. sqrt(eps) relative forgives any plausible roundoff, while
        // genuine domination failures — which violate at data scale — still
        // throw.
        const auto tolerance =
            std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(std::abs(merge.mergeDist), std::abs(death));
        if (death < merge.mergeDist - tolerance)
        {
          throw std::runtime_error("homological kernel: d does not dominate d' (death below birth)");
        }
        bars.emplace_back(merge.mergeDist, std::max(death, merge.mergeDist));

        // Only now does the contraction enter primeComponents: merge i must
        // see the quotient by merges 1..i-1, not by itself or later ones.
        primeComponents.unite(primeComponents.find(merge.a), primeComponents.find(merge.b));
      }
    }

    // Full kernel computation for one instance: d' merge order (the births),
    // d-MST (the death timeline), cross filtration. Requires d' <= d pointwise;
    // compute_cross_filtration throws otherwise. The instance size comes from
    // dDist.size(); callers validate that both oracles agree on it. For
    // sizes <= 1 the kernel barcode is empty.
    // mergeDistTransform maps each stored merge distance from the scale the
    // DistT functor computes in to the scale of the output bars (sqrt for the
    // squared-Euclidean oracle, identity for oracles already in bar scale); it
    // must be monotone so the merge order is unchanged.
    template <typename DistT, typename T, typename PostF = std::identity>
    requires DistanceOracle<DistT, T>
    void homological_kernel_single_impl(
        const DistT &dDist, const DistT &dPrimeDist, Barcode<T> &ret, PostF mergeDistTransform = {})
    {
      std::vector<MergeEdge<T>> primeMerges; // d' merge order: the births
      std::vector<MergeEdge<T>> dMerges;     // d-MST edges: the sweep timeline that answers the deaths
      mst_merge_order(dPrimeDist, primeMerges);
      mst_merge_order(dDist, dMerges);
      for (auto &m : primeMerges)
      {
        m.mergeDist = mergeDistTransform(m.mergeDist);
      }
      for (auto &m : dMerges)
      {
        m.mergeDist = mergeDistTransform(m.mergeDist);
      }

      std::vector<PersistencePair<T>> bars;
      compute_cross_filtration(primeMerges, dMerges, bars);
      ret = std::move(bars);
    }

    template <typename T>
    void homological_kernel_distmat_single_impl(
        const Tensor<DistanceMatrix<T>> &distmat, const Tensor<DistanceMatrix<T>> &distmatPrime,
        Tensor<Barcode<T>> &retBarcodes, const std::vector<size_t> &index)
    {
      auto const &dm = distmat(index);           // the Distmat<T> for this instance
      auto const &dmPrime = distmatPrime(index); // its aligned d′ counterpart
      if (dm.size() != dmPrime.size())
      {
        throw std::runtime_error(
            "homological kernel: distance matrices at index " + index_to_string(index) + " have mismatched sizes (" +
            std::to_string(dm.size()) + " and " + std::to_string(dmPrime.size()) + ")");
      }

      detail::homological_kernel_single_impl(dm, dmPrime, retBarcodes(index));
    }

    template <typename T, TensorProperties Properties, TensorProperties PrimeProperties>
    void homological_kernel_pcloud_single_impl(
        const Tensor<PointCloud<T>, Properties> &pclouds,
        const Tensor<PointCloud<T>, PrimeProperties> &pcloudsPrime,
        Tensor<Barcode<T>> &retBarcodes, const std::vector<size_t> &index)
    {
      auto const &pc = pclouds(index);
      auto const &pcPrime = pcloudsPrime(index);

      if (pc.coords().rank() != 2 || pcPrime.coords().rank() != 2)
      {
        throw std::runtime_error(
            "homological kernel: point cloud at index " + index_to_string(index) +
            " must have shape (m, n)");
      }

      if (pc.n_points() != pcPrime.n_points() || pc.dim() != pcPrime.dim())
      {
        throw std::runtime_error(
            "homological kernel: point clouds at index " + index_to_string(index) + " have mismatched shapes " +
            shape_to_string(std::vector<size_t>{pc.n_points(), pc.dim()}) + " and " +
            shape_to_string(std::vector<size_t>{pcPrime.n_points(), pcPrime.dim()}));
      }

      // Every rank-2 cloud flows through: 0 or 1 points give an empty barcode
      // via the n <= 1 early-outs, and a zero-dimensional cloud (n, 0) induces
      // the all-zero metric, yielding n-1 zero-length bars exactly like the
      // distance-matrix route does for the equivalent all-zero matrix.
      SquaredEuclideanDistance<T> dDist(pc); // uses logical coordinates, computes d^2 on demand
      SquaredEuclideanDistance<T> dPrimeDist(pcPrime);

      detail::homological_kernel_single_impl(
          dDist, dPrimeDist, retBarcodes(index), [](T v) { return std::sqrt(v); });
    }

  } // namespace detail

  template <
      typename ElemT, typename T, TensorProperties Properties = TensorProperty::None,
      TensorProperties PrimeProperties = Properties>
  class HomologicalKernelImpl : public StoppableTask<void>
  {
  public:
    HomologicalKernelImpl(
        Tensor<ElemT, Properties> input, Tensor<ElemT, PrimeProperties> inputPrime, Tensor<Barcode<T>> &ret)
        : m_input(std::move(input)), m_inputPrime(std::move(inputPrime)), m_ret(ret)
    {
    }

  private:
    tf::Future<void> run_async(Executor &exec) override
    {
      if (m_input.shape() != m_inputPrime.shape())
      {
        throw std::runtime_error(
            "homological kernel: input tensors must have the same shape (got " + shape_to_string(m_input.shape()) +
            " and " + shape_to_string(m_inputPrime.shape()) + ")");
      }

      // Validate before touching m_ret so a failed spawn leaves the caller's
      // out tensor intact.
      m_ret = Tensor<Barcode<T>>(m_input.shape());

      next_step(
          m_input.size(), "Computing H0 kernel",
          std::is_same_v<ElemT, PointCloud<T>> ? "pointcloud" : "distance matrix");
      return sb::parallel_walk_async(
          m_input,
          [this](const std::vector<size_t> &index) {
            if (stop_requested())
            {
              return;
            }

            if constexpr (std::is_same_v<ElemT, PointCloud<T>>)
            {
              detail::homological_kernel_pcloud_single_impl(m_input, m_inputPrime, m_ret, index);
            }
            else
            {
              detail::homological_kernel_distmat_single_impl(m_input, m_inputPrime, m_ret, index);
            }

            add_progress(1);
          },
          exec);
    }
    Tensor<ElemT, Properties> m_input;
    Tensor<ElemT, PrimeProperties> m_inputPrime;
    Tensor<Barcode<T>> &m_ret;
  };

} // namespace sb::ph

#endif
