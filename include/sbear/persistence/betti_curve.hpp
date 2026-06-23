#ifndef STABLEBEAR_BETTI_CURVE_H
#define STABLEBEAR_BETTI_CURVE_H

#include "../functional/pcf.hpp"
#include "barcode.hpp"
#include "barcode_summary.hpp"

#include <algorithm>

namespace sb::ph
{
  /**
   * Converts a single barcode to a Betti curve PCF. The Betti curve counts,
   * for each filtration value t, the number of bars alive at t (i.e., bars
   * with birth <= t < death).
   *
   * @tparam T Data type of the bar birth/death values.
   * @param barcode The barcode to convert.
   * @return The Betti curve as a PCF.
   */
  template <typename T>
  Pcf<T, T> barcode_to_betti_curve(const Barcode<T>& barcode)
  {
    if (barcode.bars().empty())
    {
      return Pcf<T, T>();
    }

    // Collect all events: +1 at birth, -1 at death
    struct Event
    {
      T time;
      int delta; // +1 for birth, -1 for death
    };

    std::vector<Event> events;
    events.reserve(barcode.bars().size() * 2);

    for (auto const& bar : barcode.bars())
    {
      events.push_back({bar.birth, +1});
      if (!Barcode<T>::is_infinite(bar.death))
      {
        events.push_back({bar.death, -1});
      }
    }

    std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
      return a.time < b.time || (a.time == b.time && a.delta > b.delta);
    });

    using PcfT = Pcf<T, T>;
    using PcfPointT = typename PcfT::point_type;

    std::vector<PcfPointT> points;

    T count = 0;
    T lastTime = T{0};

    for (auto const& event : events)
    {
      if (event.time != lastTime)
      {
        points.emplace_back(lastTime, count);
        lastTime = event.time;
      }
      count += event.delta;
    }

    points.emplace_back(lastTime, count);

    return PcfT(std::move(points));
  }

  template <typename T>
  auto make_betti_curve_task(const Tensor<Barcode<T>>& barcodes, Tensor<Pcf<T, T>>& out)
  {
    return std::make_unique<BarcodeSummaryTask<T, decltype(&barcode_to_betti_curve<T>)>>(
        barcodes, out, barcode_to_betti_curve<T>,
        "Converting barcodes to Betti curves");
  }
}

#endif // STABLEBEAR_BETTI_CURVE_H
