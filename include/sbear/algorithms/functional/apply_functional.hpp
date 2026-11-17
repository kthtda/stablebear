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

  /// Gradient of lp_norm(f, p) with respect to each breakpoint of f, as
  /// points whose t holds the derivative with respect to the time t_k and
  /// whose v holds the derivative with respect to the value v_k:
  ///
  ///   d/dv_k = (|v_k| / ||f||_p)^(p-1) sign(v_k) (t_{k+1} - t_k)
  ///   d/dt_k = ||f||_p^(1-p) (|v_{k-1}|^p - |v_k|^p) / p
  ///
  /// The first time, which is always 0, and the value of the last, infinite
  /// segment get 0.
  template <PcfLike F>
  std::vector<typename F::point_type> lp_norm_grad(const F& f, typename F::value_type p)
  {
    using Tv = typename F::value_type;
    using Tt = typename F::time_type;
    const auto& pts = f.points();
    std::vector<typename F::point_type> grad(pts.size(), typename F::point_type(Tt{ 0 }, Tv{ 0 }));

    // For p = 1 the formulas need no powers: d/dv_k = sign(v_k) (t_{k+1} - t_k)
    // and d/dt_k = |v_{k-1}| - |v_k|.
    const bool isL1 = p == Tv{ 1 };
    Tv norm = isL1 ? l1_norm(f) : lp_norm(f, p);
    if (norm == 0)
    {
      return grad;
    }
    const Tv timeScale = isL1 ? Tv{ 1 } : std::pow(norm, Tv{ 1 } - p) / p;
    auto absPow = [isL1, p](Tv v) {
      return isL1 ? std::abs(v) : std::pow(std::abs(v), p);
    };
    for (std::size_t k = 0; k < pts.size(); ++k)
    {
      if (k > 0)
      {
        grad[k].t = static_cast<Tt>(timeScale * (absPow(pts[k - 1].v) - absPow(pts[k].v)));
      }
      if (k + 1 < pts.size() && pts[k].v != 0)
      {
        Tv dt = static_cast<Tv>(pts[k + 1].t - pts[k].t);
        Tv sign = pts[k].v > 0 ? Tv{ 1 } : Tv{ -1 };
        Tv scale = isL1 ? Tv{ 1 } : std::pow(std::abs(pts[k].v) / norm, p - 1);
        grad[k].v = static_cast<Tv>(dt * scale * sign);
      }
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
