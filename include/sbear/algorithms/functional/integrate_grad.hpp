#ifndef SB_INTEGRATE_GRAD_H
#define SB_INTEGRATE_GRAD_H

#include "../../functional/plf.hpp"

#include <algorithm>
#include <cstddef>

namespace sb
{
  /// The number of breakpoints of integrate(f): those of f, and one more
  /// when the last value of f differs from the value before it.
  template <FloatType Tt, FloatType Tv>
  size_t integral_size(const Pcf<Tt, Tv>& f)
  {
    const auto& pts = f.points();
    const size_t n = pts.size();
    const Tv slope = n >= 2 ? pts[n - 2].v : Tv{0};
    return n + (pts.back().v != slope ? 1 : 0);
  }

  /// The gradient of a number L with respect to the breakpoints of f, from
  /// its gradient with respect to the breakpoints of integrate(f).
  ///
  /// g holds, for each of the integral_size(f) breakpoints (T_i, F_i) of the
  /// integral, the pair (dL/dT_i, dL/dF_i), and out receives, for each
  /// breakpoint (t_j, v_j) of f, the pair (dL/dt_j, dL/dv_j).
  ///
  /// With F_i the sum of v_k (t_{k+1} - t_k) over k < i, a time t_j or a
  /// value v_j moves every F_i after it, so its gradient takes the sum of
  /// dL/dF_i over those i. The breakpoint that integrate adds after the
  /// last time t_n is at t_n + s with value F_n + v_n s, for the step
  /// s = max(1, t_n), which moves with t_n when t_n > 1.
  template <FloatType Tt, FloatType Tv, typename G>
  void integrate_grad(const Pcf<Tt, Tv>& f, const G* g, G* out)
  {
    const auto& pts = f.points();
    const size_t n = pts.size();
    std::fill(out, out + 2 * n, G{0});

    // The sum of dL/dF_i over the breakpoints i after j.
    G after = 0;
    if (integral_size(f) > n)
    {
      const G step = static_cast<G>(std::max(static_cast<Tt>(1), pts.back().t));
      out[2 * (n - 1)] += g[2 * n];
      out[2 * (n - 1) + 1] += step * g[2 * n + 1];
      if (pts.back().t > static_cast<Tt>(1))
      {
        // The step is t_n, which moves the added breakpoint once more.
        out[2 * (n - 1)] += g[2 * n] + static_cast<G>(pts.back().v) * g[2 * n + 1];
      }
      after = g[2 * n + 1];
    }

    for (size_t j = n; j-- > 0;)
    {
      if (j + 1 < n)
      {
        // The piece from t_j to t_{j+1}, of value v_j.
        out[2 * j] -= static_cast<G>(pts[j].v) * after;
        out[2 * j + 1] += static_cast<G>(pts[j + 1].t - pts[j].t) * after;
      }
      after += g[2 * j + 1];
      if (j >= 1)
      {
        // T_j is t_j, which also ends the piece of value v_{j-1}. T_0 is
        // the constant 0.
        out[2 * j] += g[2 * j] + static_cast<G>(pts[j - 1].v) * after;
      }
    }
  }
}

#endif
