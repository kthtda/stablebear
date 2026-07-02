#ifndef STABLEBEAR_ACCUMULATED_PERSISTENCE_H
#define STABLEBEAR_ACCUMULATED_PERSISTENCE_H

#include "../functional/pcf.hpp"
#include "barcode.hpp"
#include "barcode_summary.hpp"

#include <algorithm>

namespace sb::ph
{
  /**
   * Converts a single barcode to an accumulated persistence function (APF).
   *
   * See: Biscio, C. A. N. and Moller, J. (2019). The accumulated persistence
   * function, a new useful functional summary statistic for topological data
   * analysis, with a view to brain artery trees and spatial point process
   * applications. Journal of Computational and Graphical Statistics, 28(3),
   * 671-681.
   *
   * @tparam T Data type of the bar birth/death values.
   * @param barcode The barcode to convert.
   * @return The accumulated persistence function as a PCF.
   */
  /**
   * @param max_death  If finite, only bars with death <= max_death are
   *                   included (Equation 2 in the paper, where T = max_death).
   */
  template <typename T>
  Pcf<T, T> barcode_to_accumulated_persistence(const Barcode<T>& barcode,
      T max_death = std::numeric_limits<T>::infinity())
  {
    if (barcode.bars().empty())
    {
      return Pcf<T, T>();
    }

    // For each qualifying bar, compute lifetime and midpoint
    struct Entry
    {
      T midpoint;
      T lifetime;
    };

    std::vector<Entry> entries;
    entries.reserve(barcode.bars().size());

    for (auto const& bar : barcode.bars())
    {
      if (Barcode<T>::is_infinite(bar.death))
        continue;

      if (bar.death > max_death)
        continue;

      T lifetime = bar.death - bar.birth;
      T midpoint = (bar.birth + bar.death) / T{2};
      entries.push_back({midpoint, lifetime});
    }

    if (entries.empty())
    {
      return Pcf<T, T>();
    }

    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
      return a.midpoint < b.midpoint;
    });

    using PcfT = Pcf<T, T>;
    using PcfPointT = typename PcfT::point_type;

    std::vector<PcfPointT> points;

    T cumSum = T{0};
    // Start at the earliest midpoint when it is negative (sublevel-set
    // filtrations), mirroring barcode_to_betti_curve -- starting at 0 would
    // emit an unsorted PCF.
    T lastMidpoint = std::min(T{0}, entries.front().midpoint);

    for (auto const& entry : entries)
    {
      if (entry.midpoint != lastMidpoint)
      {
        points.emplace_back(lastMidpoint, cumSum);
        lastMidpoint = entry.midpoint;
      }
      cumSum += entry.lifetime;
    }

    points.emplace_back(lastMidpoint, cumSum);

    return PcfT(std::move(points));
  }

  template <typename T>
  auto make_accumulated_persistence_task(const Tensor<Barcode<T>>& barcodes, Tensor<Pcf<T, T>>& out,
      T max_death = std::numeric_limits<T>::infinity())
  {
    auto func = [max_death](const Barcode<T>& bc) {
      return barcode_to_accumulated_persistence(bc, max_death);
    };
    return std::make_unique<BarcodeSummaryTask<T, decltype(func)>>(
        barcodes, out, std::move(func),
        "Converting barcodes to accumulated persistence functions");
  }
}

#endif // STABLEBEAR_ACCUMULATED_PERSISTENCE_H
