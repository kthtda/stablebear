#ifndef SB_ALGORITHM_APPLY_FUNCTIONAL_H
#define SB_ALGORITHM_APPLY_FUNCTIONAL_H

#include "../../functional/pcf.hpp"
#include "../../executor.hpp"
#include "../../tensor.hpp"
#include "../../task.hpp"
#include "../../dual.hpp"

#include <vector>
#include <functional>
#include <iterator>
#include <limits>

#include <iostream>

#include <taskflow/algorithm/transform.hpp>

namespace sb
{

  template <IsTensor InTensor, IsTensor OutTensor, typename F>
  class ApplyFunctional : public StoppableTask<void>
  {
  public:
    ApplyFunctional(const InTensor& in, OutTensor& out, F functional)
      : m_in(in), m_out(out), m_functional(functional)
    { }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      return sb::parallel_walk_async(m_in, [this](const std::vector<std::size_t>& idx) {
        m_out(idx) = m_functional(m_in(idx));
      }, exec);
    }

    // Handles that share the callers' storage: the walk outlives the call
    // that started the task.
    InTensor m_in;
    OutTensor m_out;
    F m_functional;
  };

  template <typename ForwardPcfIt, typename OutputIt, typename F>
  void apply_functional(ForwardPcfIt begin, ForwardPcfIt end, OutputIt beginOut, F functional, sb::Executor& exec = sb::default_executor())
  {
    using PcfT = decltype(*begin);

    tf::Taskflow flow;
    flow.transform(begin, end, beginOut, [&functional](const PcfT& pcf) {
      return functional(pcf);
      });

    exec.cpu()->run(std::move(flow)).wait();
  }

  // --- PCF norms ---

  template <PcfLike F>
  typename F::value_type l1_norm(const F& f)
  {
    typename F::value_type out{ 0 };

    iterate_segments(f.points().cbegin(), f.points().cend(), typename F::time_type{ 0 }, infinite_time<F>(), [&out](const typename F::segment_type& seg) {
      out += (seg.right - seg.left) * std::abs(seg.value);
      });

    return out;
  }

  template <PcfLike F>
  typename F::value_type l2_norm(const F& f)
  {
    typename F::value_type out{ 0 };

    iterate_segments(f.points().cbegin(), f.points().cend(), typename F::time_type{ 0 }, infinite_time<F>(), [&out](const typename F::segment_type& seg) {
      out += (seg.right - seg.left) * seg.value * seg.value;
      });

    return std::sqrt(out);
  }

  /// p is the value type, or a Dual of it to also get the derivative with
  /// respect to p.
  template <PcfLike F, typename P>
  P lp_norm(const F& f, P p)
  {
    using std::pow;
    using Tv = typename F::value_type;
    P out{ 0 };

    iterate_segments(f.points().cbegin(), f.points().cend(), typename F::time_type{ 0 }, infinite_time<F>(), [&out, p](const typename F::segment_type& seg) {
      out += static_cast<Tv>(seg.right - seg.left) * pow(static_cast<Tv>(std::abs(seg.value)), p);
      });

    return pow(out, P{ 1 } / p);
  }

  /// Gradient of lp_norm(f, p) with respect to the value at each breakpoint
  /// of f: (|v_k| / ||f||_p)^(p-1) sign(v_k) dt_k. The last breakpoint, whose
  /// segment is infinite, gets 0.
  template <PcfLike F>
  std::vector<typename F::value_type> lp_norm_grad_values(const F& f, typename F::value_type p)
  {
    using Tv = typename F::value_type;
    const auto& pts = f.points();
    std::vector<Tv> grad(pts.size(), Tv{ 0 });

    Tv norm = lp_norm(f, p);
    if (norm == 0)
    {
      return grad;
    }
    for (std::size_t k = 0; k + 1 < pts.size(); ++k)
    {
      if (pts[k].v == 0)
      {
        continue;
      }
      Tv dt = static_cast<Tv>(pts[k + 1].t - pts[k].t);
      Tv sign = pts[k].v > 0 ? Tv{ 1 } : Tv{ -1 };
      grad[k] = static_cast<Tv>(dt * std::pow(std::abs(pts[k].v) / norm, p - 1) * sign);
    }
    return grad;
  }

  template <PiecewiseFunctionLike F>
  typename F::value_type linfinity_norm(const F& f)
  {
    using value_type = typename F::value_type;
    value_type out{ 0 };
    for (auto const& pt : f.points())
    {
      out = std::max(out, static_cast<value_type>(std::abs(pt.v)));
    }
    return out;
  }

}

#endif
