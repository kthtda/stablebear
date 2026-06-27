#ifndef SB_WALK_H
#define SB_WALK_H

#include "config.hpp"
#include "concepts.hpp"
#include "executor.hpp"
#include "random_generator.hpp"

#include <taskflow/algorithm/for_each.hpp>

namespace sb
{

  namespace detail
  {
    /**
     * Core sequential walk: visits every index in row-major order, providing
     * both the multi-index and the flat (row-major) counter to the callback.
     * If f returns bool, walking stops when f returns false.
     */
    template <IsTensor TTensor, typename Func>
    void walk_impl(const TTensor& tensor, Func&& f)
    {
      auto shape_range = tensor.shape();
      std::vector<size_t> shape(std::begin(shape_range), std::end(shape_range));

      // A rank-0 (shape.empty()) tensor holds exactly one element and is visited
      // once; only a genuinely empty extent (some dim == 0) has zero elements
      // and is skipped.
      if (std::any_of(shape.begin(), shape.end(), [](size_t n){ return n == 0; }))
      {
        return;
      }

      auto ndim = shape.size();
      std::vector<size_t> cur(ndim, 0_uz);
      size_t flat = 0;

      while (true)
      {
        if constexpr (std::is_same_v<decltype(f(cur, flat)), bool>)
        {
          if (!f(cur, flat))
          {
            return;
          }
        }
        else
        {
          f(cur, flat);
        }

        ++flat;

        // Rank-0 has a single element; the carry loop below assumes ndim >= 1.
        if (ndim == 0)
        {
          return;
        }

        for (ptrdiff_t i = ndim - 1; i >= 0; --i)
        {
          ++cur[i];

          if (cur[i] < shape[i])
          {
            break;
          }

          if (i == 0)
          {
            return;
          }

          cur[i] = 0;
        }
      }
    }

    /**
     * Core parallel walk: distributes flat indices across threads, converting
     * each to a multi-index. The callback receives both the multi-index and
     * the flat index. Returns a tf::Future.
     */
    template <IsTensor TTensor, typename Func>
    tf::Future<void> parallel_walk_impl(const TTensor& tensor, Func&& f, Executor& exec)
    {
      auto shape_range = tensor.shape();
      std::vector<size_t> shape(std::begin(shape_range), std::end(shape_range));

      // A rank-0 tensor has exactly one element (total == 1 below) and is walked
      // once; only a zero-size extent (some dim == 0) has no elements and is
      // skipped.
      if (std::any_of(shape.begin(), shape.end(), [](size_t n){ return n == 0; }))
      {
        tf::Taskflow flow;
        return exec.cpu()->run(std::move(flow));
      }

      auto ndim = shape.size();
      size_t total = 1;
      for (auto s : shape)
        total *= s;

      tf::Taskflow flow;
      flow.for_each_index<size_t, size_t, size_t>(0ul, total, 1ul,
          [f, shape = std::move(shape), ndim](size_t flat) {
        thread_local std::vector<size_t> idx;
        idx.resize(ndim);

        size_t rem = flat;
        for (ptrdiff_t i = ndim - 1; i >= 0; --i)
        {
          idx[i] = rem % shape[i];
          rem /= shape[i];
        }

        f(idx, flat);
      });

      return exec.cpu()->run(std::move(flow));
    }
  }

  // ============================================================================
  // Standard walk overloads
  // ============================================================================

  /**
   * Visit every index of any IsTensor in row-major order, invoking f(idx) at each.
   * If f returns bool, walking stops when f returns false.
   */
  template <IsTensor TTensor, typename UnaryFunc>
#ifndef __CUDACC__
  requires std::invocable<UnaryFunc, std::vector<size_t>>
#endif
  void walk(const TTensor& tensor, UnaryFunc&& f)
  {
    detail::walk_impl(tensor, [&f](const std::vector<size_t>& idx, size_t) -> decltype(auto) {
      return f(idx);
    });
  }

  /**
   * Visit every index in parallel via the given Executor, invoking f(idx) at each.
   * Does not support early termination. Returns a tf::Future.
   */
  template <IsTensor TTensor, typename UnaryFunc>
#ifndef __CUDACC__
  requires std::invocable<UnaryFunc, std::vector<size_t>>
#endif
  tf::Future<void> parallel_walk_async(const TTensor& tensor, UnaryFunc&& f, Executor& exec)
  {
    return detail::parallel_walk_impl(tensor, [f = std::forward<UnaryFunc>(f)](const std::vector<size_t>& idx, size_t) {
      f(idx);
    }, exec);
  }

  /**
   * Like parallel_walk_async(), but blocks until complete.
   */
  template <IsTensor TTensor, typename UnaryFunc>
#ifndef __CUDACC__
  requires std::invocable<UnaryFunc, std::vector<size_t>>
#endif
  void parallel_walk(const TTensor& tensor, UnaryFunc&& f, Executor& exec)
  {
    parallel_walk_async(tensor, std::forward<UnaryFunc>(f), exec).wait();
  }

  // ============================================================================
  // Walk with random: deterministically-seeded engine at each element
  // ============================================================================

  /**
   * Walk every index of a tensor, providing a deterministically-seeded random
   * engine at each element. The lambda receives (idx, engine&) where the engine
   * is seeded from the generator and the element's flat (row-major) index.
   */
  template <IsTensor TTensor, typename EngineT, typename BinaryFunc>
#ifndef __CUDACC__
  requires std::invocable<BinaryFunc, std::vector<size_t>, EngineT&>
#endif
  void walk(const TTensor& tensor, RandomGenerator<EngineT>& gen, BinaryFunc&& f)
  {
    auto block = gen.reserve(tensor.size());
    detail::walk_impl(tensor, [block, &f](const std::vector<size_t>& idx, size_t flat) {
      auto engine = block.sub_generator(flat);
      f(idx, engine);
    });
  }

  /**
   * Like walk() with random, but distributes work across threads.
   * Deterministic regardless of thread count or execution order. Returns a tf::Future.
   */
  template <IsTensor TTensor, typename EngineT, typename BinaryFunc>
#ifndef __CUDACC__
  requires std::invocable<BinaryFunc, std::vector<size_t>, EngineT&>
#endif
  tf::Future<void> parallel_walk_async(const TTensor& tensor, RandomGenerator<EngineT>& gen,
                                       BinaryFunc&& f, Executor& exec)
  {
    auto block = gen.reserve(tensor.size());
    return detail::parallel_walk_impl(tensor, [block, f = std::forward<BinaryFunc>(f)](const std::vector<size_t>& idx, size_t flat) {
      auto engine = block.sub_generator(flat);
      f(idx, engine);
    }, exec);
  }

  /**
   * Like parallel_walk_async() with random, but blocks until complete.
   */
  template <IsTensor TTensor, typename EngineT, typename BinaryFunc>
#ifndef __CUDACC__
  requires std::invocable<BinaryFunc, std::vector<size_t>, EngineT&>
#endif
  void parallel_walk(const TTensor& tensor, RandomGenerator<EngineT>& gen,
                     BinaryFunc&& f, Executor& exec)
  {
    parallel_walk_async(tensor, gen, std::forward<BinaryFunc>(f), exec).wait();
  }

}

#endif
