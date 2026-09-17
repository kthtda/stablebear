#ifndef STABLEBEAR_SAMPLING_SUBSAMPLE_H
#define STABLEBEAR_SAMPLING_SUBSAMPLE_H

// Per-query subsampling of a reference point cloud or distance matrix.
//
// Structured like the persistence pipeline (compute_persistence.hpp): a
// `detail` per-query core, one templated StoppableTask that walks the query
// points in parallel, and per-element-type aliases. Each query point computes
// its own weight row on the fly (weighting.hpp functors) and draws all of its
// instances from it (weighted_draw.hpp primitives), so no dense
// (n_query x n_reference) weight matrix is ever materialized — only one
// n_reference row per worker thread lives at a time.
//
// Each subsample is an indexed view: it stores just the drawn indices and
// shares the reference's coordinate/distance buffer. The output tensor is
// filled in place through the @p out reference the caller passes, mirroring the
// Ripser tasks.

#include "weighted_draw.hpp"
#include "weighting.hpp"

#include "../distance_matrix.hpp"
#include "../executor.hpp"
#include "../point_cloud.hpp"
#include "../random_generator.hpp"
#include "../task.hpp"
#include "../tensor.hpp"
#include "../walk.hpp"

#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace sb::sampling
{

  /// Distance-matrix subsampling has no filter — the stored distance *is* the
  /// filter value. This placeholder lets the point-cloud and distance-matrix
  /// paths share one task template; the filter is used only in the point-cloud
  /// branch (selected by `if constexpr`), so the distance-matrix instantiation
  /// carries this empty type instead.
  struct NoFilter
  {
  };

  namespace detail
  {

    /// Fill query @p q's weight row @p row (length n_reference) from a point
    /// cloud: @p distribution of @p filter(query point, reference point) per
    /// reference point, log-space max-shifted when the distribution offers the
    /// channel, then prepared for drawing (validated; converted in place to a
    /// CDF when @p toCdf). Returns the eligible (strictly-positive) count. The
    /// point-cloud analogue of run_euclidean_ripser.
    template <typename T, typename FilterF, typename DistF>
    size_t weight_query_row_pcloud(
        const PointCloud<T> &reference, const PointCloud<T> &query, size_t q, const FilterF &filter,
        const DistF &distribution, bool toCdf, std::span<T> row)
    {
      for (size_t r = 0; r < row.size(); ++r)
      {
        row[r] = weight_entry(distribution, filter(query, q, reference, r));
      }
      if constexpr (has_log_weight<DistF, T>)
      {
        log_row_to_weights(row);
      }
      return prepare_weight_row(row, toCdf);
    }

    /// As weight_query_row_pcloud, with filter values read from the precomputed
    /// distance matrix @p reference; @p query holds reference row indices. The
    /// distance-matrix analogue.
    template <typename T, typename DistF>
    size_t weight_query_row_distmat(
        const DistanceMatrix<T> &reference, const Tensor<uint64_t> &query, size_t q, const DistF &distribution,
        bool toCdf, std::span<T> row)
    {
      const auto queryRow = static_cast<size_t>(query(q));
      for (size_t r = 0; r < row.size(); ++r)
      {
        row[r] = weight_entry(distribution, reference(queryRow, r));
      }
      if constexpr (has_log_weight<DistF, T>)
      {
        log_row_to_weights(row);
      }
      return prepare_weight_row(row, toCdf);
    }

  } // namespace detail

  /// Parallel per-query subsampling task, templated on the subsample element
  /// type (PointCloud<T> or DistanceMatrix<T>), the query representation
  /// (PointCloud<T> coordinates, or a Tensor<uint64_t> of reference indices),
  /// and the weighting functors. Mirrors RipserTaskImpl: one class, one
  /// parallel walk, `if constexpr` dispatch between the two input kinds.
  ///
  /// The filter/distribution are compile-time template parameters (unlike the
  /// Ripser task, which has none) so the built-in weighting inlines into a
  /// fully fused draw path — the feature's core performance property. The
  /// binding selects the concrete instantiation via std::visit over the functor
  /// variants.
  template <typename ElemT, typename QueryT, typename FilterF, typename DistF>
  class SubsampleTaskImpl : public StoppableTask<void>
  {
    using T = typename ElemT::value_type;
    static constexpr bool is_pcloud = std::is_same_v<ElemT, PointCloud<T>>;
    static constexpr bool query_is_indices = std::is_same_v<QueryT, Tensor<uint64_t>>;

  public:
    /// @p out is filled in place (allocated to (n_query, n_instances) when the
    /// task runs), like the Ripser tasks. @p gen is read only in the
    /// synchronous prologue of run_async, to reserve the draw's seed block
    /// before the caller's next draw.
    SubsampleTaskImpl(
        ElemT reference, QueryT query, FilterF filter, DistF distribution, Tensor<ElemT> &out, size_t sampleSize,
        size_t nInstances, bool replace, DefaultRandomGenerator &gen)
        : m_reference(std::move(reference)), m_query(std::move(query)), m_filter(std::move(filter)),
          m_distribution(std::move(distribution)), m_out(out), m_sampleSize(sampleSize), m_nInstances(nInstances),
          m_replace(replace), m_gen(gen)
    {
    }

  private:
    tf::Future<void> run_async(Executor &exec) override
    {
      // Synchronous prologue (runs on the caller's thread before the walk is
      // launched): validate, size the output, and reserve the seed block.
      // Reserving here, before the async draws, is what makes consecutive calls
      // draw fresh samples. The block is sized to the *output* cells, not the
      // walked queries: the parallel_walk(gen) overload would seed one engine
      // per query cell, but each query fans out into n_instances draws that
      // must each get their own engine, seeded by flat output index so the
      // result is independent of thread count. Hence the block is reserved and
      // sub-seeded by hand rather than via the walk overload.
      validate();

      const size_t nQuery = query_count();
      const size_t nReference = reference_count();
      m_out = Tensor<ElemT>({nQuery, m_nInstances});
      m_queryIndices = Tensor<size_t>({nQuery}); // drives the per-query walk
      const auto seedBlock = m_gen.reserve(m_out.size());

      next_step(m_out.size(), "Drawing subsamples.", "subsample");

      const size_t nInstances = m_nInstances;
      const size_t sampleSize = m_sampleSize;
      const bool replace = m_replace;
      return parallel_walk_async(
          m_queryIndices,
          [this, nReference, nInstances, sampleSize, replace, seedBlock](const std::vector<size_t> &idx) {
            if (stop_requested())
            {
              return;
            }

            const size_t q = idx[0];

            // One n_reference weight row per worker, reused across queries: this is
            // what replaces the old dense (n_query x n_reference) matrix. The
            // weighting overwrites every entry, so no clearing is needed.
            thread_local std::vector<T> rowBuffer;
            rowBuffer.resize(nReference);
            const std::span<T> row(rowBuffer.data(), nReference);

            // With replacement the row becomes a CDF, built once and drawn from by
            // every instance; without replacement it stays raw weights.
            const size_t nEligible = weight_query_row(q, replace, row);
            const std::span<const T> drawRow(row.data(), row.size());

            for (size_t i = 0; i < nInstances; ++i)
            {
              // Seed per output cell from its flat index, so the draw is identical
              // regardless of how the walk is scheduled across threads.
              auto engine = seedBlock.sub_generator((q * nInstances) + i);
              Tensor<uint64_t> indices = detail::draw_indices(drawRow, nEligible, sampleSize, replace, engine);
              m_out({q, i}) = ElemT(m_reference, std::move(indices));
            }
            add_progress(nInstances);
          },
          exec);
    }

    /// Number of query points to walk.
    size_t query_count() const
    {
      if constexpr (query_is_indices)
      {
        return m_query.shape(0);
      }
      else
      {
        return m_query.n_points();
      }
    }

    /// Length of a weight row (points in the reference).
    size_t reference_count() const
    {
      if constexpr (is_pcloud)
      {
        return m_reference.n_points();
      }
      else
      {
        return m_reference.size();
      }
    }

    /// Compute + prepare query @p q's weight row, dispatching on the input kind.
    size_t weight_query_row(size_t q, bool toCdf, std::span<T> row) const
    {
      if constexpr (!is_pcloud)
      {
        return detail::weight_query_row_distmat(m_reference, m_query, q, m_distribution, toCdf, row);
      }
      else if constexpr (query_is_indices)
      {
        // Query points selected from the reference itself: the query point is
        // reference row m_query(q), so the reference doubles as the query cloud.
        return detail::weight_query_row_pcloud(
            m_reference, m_reference, static_cast<size_t>(m_query(q)), m_filter, m_distribution, toCdf, row);
      }
      else
      {
        return detail::weight_query_row_pcloud(m_reference, m_query, q, m_filter, m_distribution, toCdf, row);
      }
    }

    /// Reject malformed inputs on the caller's thread, before any draws. Folds
    /// in the checks the free-function entry points used to make.
    void validate() const
    {
      if constexpr (is_pcloud)
      {
        if (m_reference.coords().rank() != 2)
        {
          throw std::invalid_argument("reference must be a 2-D (n_points, dim) point cloud");
        }
        if constexpr (!query_is_indices)
        {
          if (m_query.coords().rank() != 2)
          {
            throw std::invalid_argument("query must be a 2-D (n_points, dim) point cloud");
          }
          if (m_query.dim() != m_reference.dim())
          {
            throw std::invalid_argument("reference and query must have the same dimension");
          }
        }
      }
      else
      {
        if (m_reference.size() == 0)
        {
          throw std::invalid_argument("reference distance matrix must be nonempty");
        }
      }
      if (m_sampleSize == 0)
      {
        throw std::invalid_argument("sample_size must be positive");
      }
    }

    ElemT m_reference; ///< reference whose buffer the indexed subsamples share
    QueryT m_query;   ///< query points (coordinates) or reference indices
    FilterF m_filter; ///< NoFilter for the distance-matrix path
    DistF m_distribution;
    Tensor<ElemT> &m_out; ///< (n_query, n_instances), filled in place
    size_t m_sampleSize;
    size_t m_nInstances;
    bool m_replace;
    DefaultRandomGenerator &m_gen; ///< used only in run_async's synchronous prologue
    Tensor<size_t> m_queryIndices; ///< walk driver, kept alive for the async walk
  };

  /// Point-cloud subsampling: weight each reference point by @p distribution of
  /// @p filter(query point, reference point).
  template <typename T, typename FilterF, typename DistF>
  using SubsampleTask = SubsampleTaskImpl<PointCloud<T>, PointCloud<T>, FilterF, DistF>;

  /// Point-cloud subsampling with the query points given as reference row
  /// indices instead of coordinates (query point q is reference row query(q)).
  template <typename T, typename FilterF, typename DistF>
  using SubsampleIndexQueryTask = SubsampleTaskImpl<PointCloud<T>, Tensor<uint64_t>, FilterF, DistF>;

  /// Distance-matrix subsampling: distances come from the precomputed reference
  /// matrix, the query holds reference row indices, and there is no filter.
  template <typename T, typename DistF>
  using SubsampleDistMatTask = SubsampleTaskImpl<DistanceMatrix<T>, Tensor<uint64_t>, NoFilter, DistF>;

} // namespace sb::sampling

#endif
