#ifndef STABLEBEAR_TRANSFORM_BARS_H
#define STABLEBEAR_TRANSFORM_BARS_H

#include "barcode.hpp"
#include "barcode_summary.hpp"
#include "../functional/plf.hpp"

#include <limits>
#include <utility>
#include <vector>

namespace sb::ph
{
  /**
   * The value of @p f at @p t, where an infinite @p t gives the limit of f:
   * +inf or -inf as the last segment rises or falls, or its last value if it
   * is flat.
   */
  template <typename T>
  T evaluate_extended(const Plf<T, T>& f, T t)
  {
    if (!Barcode<T>::is_infinite(t))
    {
      return f.evaluate(t);
    }
    const auto& pts = f.points();
    const auto n = pts.size();
    if (n == 1 || pts[n - 1].v == pts[n - 2].v)
    {
      return pts[n - 1].v;
    }
    return pts[n - 1].v > pts[n - 2].v ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
  }

  /// The barcode with each bar (b, d) replaced by (f(b), f(d)). Unless
  /// @p keepEmpty, bars with f(b) >= f(d) are removed.
  template <typename T>
  Barcode<T> transform_bars(const Barcode<T>& barcode, const Plf<T, T>& f, bool keepEmpty = false)
  {
    std::vector<PersistencePair<T>> bars;
    bars.reserve(barcode.bars().size());
    for (const auto& bar : barcode.bars())
    {
      auto birth = evaluate_extended(f, bar.birth);
      auto death = evaluate_extended(f, bar.death);
      if (keepEmpty || birth < death)
      {
        bars.emplace_back(birth, death);
      }
    }
    return Barcode<T>(std::move(bars));
  }

  /// Apply transform_bars with @p f to every barcode in @p barcodes, a tensor
  /// or a time series, in parallel.
  template <typename Container>
  auto make_transform_bars_task(Container barcodes, Plf<barcode_scalar_t<Container>> f, bool keepEmpty = false)
  {
    using T = barcode_scalar_t<Container>;
    return make_barcode_summary_task(
        std::move(barcodes),
        [f = std::move(f), keepEmpty](const Barcode<T>& bc) { return transform_bars(bc, f, keepEmpty); },
        "Transforming bars");
  }
}

#endif // STABLEBEAR_TRANSFORM_BARS_H
