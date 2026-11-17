#ifndef SB_ALGORITHM_APPLY_FUNCTIONAL_H
#define SB_ALGORITHM_APPLY_FUNCTIONAL_H

#include "../../functional/pcf.hpp"
#include "../../executor.hpp"
#include "../../tensor.hpp"
#include "../../task.hpp"

#include <vector>
#include <functional>
#include <iterator>
#include <limits>
#include <cmath>
#include <algorithm>

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
      tf::Taskflow flow;
      std::vector<tf::Task> tasks;

      sb::walk(m_in, [this](const std::vector<std::size_t>& idx) {
        m_out(idx) = m_functional(m_in(idx));
      });

      flow.linearize(tasks);
      return exec.cpu()->run(std::move(flow));
    }

    const InTensor& m_in;
    OutTensor& m_out;
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

  template <PiecewiseFunctionLike F>
  typename F::value_type linfinity_norm(const F& f)
  {
    using value_type = typename F::value_type;
    value_type out{ 0 };
    for (auto const& pt : f.points())
    {
      auto absValue = static_cast<value_type>(std::abs(pt.v));
      // std::max never picks NaN, so NaN must be passed on explicitly
      if (std::isnan(absValue))
      {
        return absValue;
      }
      out = std::max(out, absValue);
    }
    return out;
  }

  template <PcfLike F>
  typename F::value_type lp_norm(const F& f, typename F::value_type p)
  {
    using value_type = typename F::value_type;

    if (std::isinf(p))
    {
      return linfinity_norm(f);
    }

    // Find the largest |f| on [0, inf), passing on NaN, which std::max never picks
    value_type scale{ 0 };
    bool hasNan = false;
    iterate_segments(f.points().cbegin(), f.points().cend(), typename F::time_type{ 0 }, infinite_time<F>(), [&scale, &hasNan](const typename F::segment_type& seg) {
      auto absValue = std::abs(seg.value);
      hasNan = hasNan || std::isnan(absValue);
      scale = std::max(scale, absValue);
      });

    if (hasNan)
    {
      return std::numeric_limits<value_type>::quiet_NaN();
    }
    if (scale == value_type{ 0 } || std::isinf(scale))
    {
      return scale;
    }

    // ||f||_p = scale * ||f / scale||_p. Since |f / scale| <= 1, raising it to
    // the power p cannot overflow, as |f|^p does for large p.
    value_type out{ 0 };
    iterate_segments(f.points().cbegin(), f.points().cend(), typename F::time_type{ 0 }, infinite_time<F>(), [&out, scale, p](const typename F::segment_type& seg) {
      out += (seg.right - seg.left) * std::pow(std::abs(seg.value) / scale, p);
      });

    return scale * std::pow(out, value_type{ 1 } / p);
  }

}

#endif
