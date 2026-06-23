#ifndef SB_ALGORITHMS_MATRIX_INTEGRATE_H
#define SB_ALGORITHMS_MATRIX_INTEGRATE_H

#include "iterate_rectangles.hpp"
#include "../../functional/pcf.hpp"
#include "../../distance_matrix.hpp"
#include "../../symmetric_matrix.hpp"
#include "../../tensor.hpp"
#include "../../executor.hpp"
#include "../../task.hpp"

#include <vector>

#include <taskflow/algorithm/for_each.hpp>

namespace sb
{
  template <typename Tt, typename Tv, typename PairOp>
  Tv integrate(const Pcf<Tt, Tv>& f, const Pcf<Tt, Tv>& g, PairOp op, Tt a = Tt(0), Tt b = std::numeric_limits<Tt>::max())
  {
    using rect_t = typename Pcf<Tt, Tv>::rectangle_type;

    Tv val = Tv(0);
    iterate_rectangles(f.points(), g.points(), [&val, &op](const rect_t& rect) -> void {
      val += (rect.right - rect.left) * op(rect.f_value, rect.g_value);
    }, a, b);

    return val;
  }

  /// CPU pairwise integration task (pdist / l2_kernel).
  /// OutputT must support operator()(i, j) = val (DistanceMatrix or SymmetricMatrix).
  /// When includeDiagonal is false, computes i > j only (DistanceMatrix).
  /// When includeDiagonal is true, computes i >= j (SymmetricMatrix).
  template <typename TOperation, typename PcfFwdIt, typename OutputT, bool includeDiagonal>
  class CpuPairwiseIntegrationTask : public sb::StoppableTask<void>
  {
  public:
    using pcf_type = typename PcfFwdIt::value_type;
    using value_type = typename pcf_type::value_type;
    using time_type = typename pcf_type::time_type;

    CpuPairwiseIntegrationTask(OutputT out, PcfFwdIt beginPcfs, PcfFwdIt endPcfs, TOperation op)
      : m_fs(beginPcfs, endPcfs)
      , m_out(std::move(out))
      , m_op(op)
    { }

    OutputT& output() { return m_out; }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      auto sz = m_fs.size();
      size_t totalWork;
      if constexpr (includeDiagonal)
        totalWork = (sz * (sz + 1)) / 2;
      else
        totalWork = (sz * (sz - 1)) / 2;

      next_step(totalWork, includeDiagonal ? "Computing kernel matrix." : "Computing distance matrix.", "integral");

      tf::Taskflow flow;
      if (m_fs.empty())
      {
        return exec.cpu()->run(std::move(flow));
      }

      flow.for_each_index<size_t, size_t, size_t>(0ul, sz, 1ul, [this](size_t i) {
        if (stop_requested())
        {
          return;
        }
        compute_row(i);
      });

      return exec.cpu()->run(std::move(flow));
    }

    void compute_row(size_t i)
    {
      auto sz = m_fs.size();
      size_t jStart = includeDiagonal ? i : i + 1;
      for (size_t j = jStart; j < sz; ++j)
      {
        auto val = m_op(integrate<time_type, value_type>(m_fs[i], m_fs[j], m_op, 0, std::numeric_limits<time_type>::max()));
        m_out(i, j) = val;
      }
      add_progress(sz - jStart);
    }

    std::vector<pcf_type> m_fs;
    OutputT m_out;
    TOperation m_op;
  };

  /// CPU cross-distance integration task (cdist).
  /// Computes all (i,j) pairs between two separate function sets.
  template <typename TOperation, typename PcfFwdIt>
  class CpuCrossIntegrationTask : public sb::StoppableTask<void>
  {
  public:
    using pcf_type = typename PcfFwdIt::value_type;
    using value_type = typename pcf_type::value_type;
    using time_type = typename pcf_type::time_type;

    CpuCrossIntegrationTask(Tensor<value_type> out,
                 PcfFwdIt beginRows, PcfFwdIt endRows,
                 PcfFwdIt beginCols, PcfFwdIt endCols,
                 TOperation op)
      : m_rowFs(beginRows, endRows)
      , m_colFs(beginCols, endCols)
      , m_out(std::move(out))
      , m_op(op)
    { }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      auto nRows = m_rowFs.size();
      auto nCols = m_colFs.size();

      next_step(nRows * nCols, "Computing cross-distances.", "integral");

      tf::Taskflow flow;
      if (nRows == 0 || nCols == 0)
      {
        return exec.cpu()->run(std::move(flow));
      }

      value_type* data = m_out.data();
      flow.for_each_index<size_t, size_t, size_t>(0ul, nRows, 1ul, [this, data, nCols](size_t i) {
        if (stop_requested())
        {
          return;
        }
        for (size_t j = 0; j < nCols; ++j)
        {
          data[i * nCols + j] = m_op(integrate<time_type, value_type>(
              m_rowFs[i], m_colFs[j], m_op, 0, std::numeric_limits<time_type>::max()));
        }
        add_progress(nCols);
      });

      return exec.cpu()->run(std::move(flow));
    }

    std::vector<pcf_type> m_rowFs;
    std::vector<pcf_type> m_colFs;
    Tensor<value_type> m_out;
    TOperation m_op;
  };

}

#endif
