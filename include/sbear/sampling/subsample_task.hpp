#ifndef STABLEBEAR_SAMPLING_SUBSAMPLE_TASK_H
#define STABLEBEAR_SAMPLING_SUBSAMPLE_TASK_H

#include "weighted_draw.hpp"

#include "../executor.hpp"
#include "../random_generator.hpp"
#include "../task.hpp"
#include "../tensor.hpp"
#include "../walk.hpp"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace sb::sampling
{

  /// A launched subsampling run: the in-flight draw @p task plus the
  /// (n_query, n_instances) @p samples tensor it fills. @p ElemT is the
  /// per-cell subsample type, an indexed PointCloud or DistanceMatrix view.
  /// Read @p samples only once @p task reports complete.
  template <typename ElemT>
  struct SubsampleHandle
  {
    std::unique_ptr<StoppableTask<void>> task;
    Tensor<ElemT> samples;
  };

  namespace detail
  {

    /// Stoppable, progress-reporting draw: one parallel walk over the
    /// (n_query, n_instances) output grid, where each cell draws one
    /// subsample from its query's prepared weight row. Subsamples store only
    /// the drawn indices, sharing the reference's data buffer.
    template <typename ElemT>
    class SubsampleTask : public StoppableTask<void>
    {
      using T = typename ElemT::value_type;
      using SeedBlock = DefaultRandomGenerator::Block;

    public:
      SubsampleTask(ElemT source, Tensor<T> weights, Tensor<size_t> nEligible,
                    Tensor<ElemT> out, size_t sampleSize, bool replace,
                    SeedBlock seedBlock)
        : m_source(std::move(source)), m_weights(std::move(weights)),
          m_nEligible(std::move(nEligible)), m_out(std::move(out)),
          m_sampleSize(sampleSize), m_replace(replace), m_seedBlock(seedBlock)
      { }

    private:
      tf::Future<void> run_async(Executor& exec) override
      {
        next_step(m_out.size(), "Drawing subsamples.", "subsample");

        const size_t nInstances = m_out.shape(1);
        return parallel_walk_async(m_out,
            [this, nInstances](const std::vector<size_t>& cellIdx) {
          if (stop_requested())
            return;

          // cellIdx = (query, instance); the engine is seeded per cell from
          // the reserved block, so draws are independent of thread count.
          auto engine = m_seedBlock.sub_generator(cellIdx[0] * nInstances + cellIdx[1]);
          const size_t queryIdx = cellIdx[0];
          const std::span<const T> row = weight_row(m_weights, queryIdx);
          const size_t nEligible = m_nEligible({queryIdx});

          Tensor<uint64_t> indices = draw_indices(row, nEligible, m_sampleSize, m_replace, engine);
          m_out(cellIdx) = ElemT(m_source, std::move(indices));

          add_progress(1);
        }, exec);
      }

      ElemT m_source;
      Tensor<T> m_weights;         ///< (n_query, n_reference), rows prepared for drawing
      Tensor<size_t> m_nEligible;  ///< per-query eligible count (prepare_weight_matrix)
      Tensor<ElemT> m_out;         ///< (n_query, n_instances)
      size_t m_sampleSize;
      bool m_replace;
      SeedBlock m_seedBlock;  ///< one seed slot per output cell, reserved by the caller
    };

    /// Allocate the (n_query, n_instances) output and launch a stoppable
    /// SubsampleTask filling it from the prepared weight matrix. @p source is
    /// the reference whose buffer the indexed subsamples share. The seed block
    /// is reserved from @p gen here, synchronously — the caller's generator
    /// advances past this draw before the async task runs, so a subsequent
    /// draw can never reuse its seeds.
    template <typename ElemT>
    SubsampleHandle<ElemT> draw_subsets_from_weights(ElemT source,
        Tensor<typename ElemT::value_type> weights, Tensor<size_t> nEligible,
        size_t sampleSize, size_t nInstances, bool replace, DefaultRandomGenerator& gen,
        Executor& exec)
    {
      Tensor<ElemT> samples({weights.shape(0), nInstances});
      auto task = std::make_unique<SubsampleTask<ElemT>>(
          std::move(source), std::move(weights), std::move(nEligible), samples,
          sampleSize, replace, gen.reserve(samples.size()));

      task->start_async(exec);
      return {std::move(task), std::move(samples)};
    }

  } // namespace detail

}

#endif
