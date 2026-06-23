#ifndef STABLEBEAR_STABLE_RANK_H
#define STABLEBEAR_STABLE_RANK_H

#include "../functional/pcf.hpp"
#include "barcode.hpp"
#include "barcode_summary.hpp"

namespace sb::ph
{
  /**
   * Converts a single barcode to a stable rank PCF. See Chachólski, W., & Riihimäki, H. (2020). Metrics and stabilization in one parameter persistence. SIAM Journal on Applied Algebra and Geometry, 4(1), 69-98.
   * @tparam T Data type of the bar birth/death values. The same type will be used for the PCF time and value types.
   * @param barcode The barcode to convert to a stable rank function.
   * @return The stable rank function corresponding to the supplied barcode.
   */
  template <typename T>
  Pcf<T, T> barcode_to_stable_rank(const Barcode<T>& barcode)
  {
    if (barcode.bars().empty())
    {
      return Pcf<T, T>();
    }

    std::vector<T> lifetimes;
    lifetimes.reserve(barcode.bars().size());

    std::transform(barcode.bars().begin(), barcode.bars().end(), std::back_inserter(lifetimes), [](const PersistencePair<T>& pp) -> T {
      if (pp.death == std::numeric_limits<T>::max())
      {
        return std::numeric_limits<T>::max();
      }

      return pp.death - pp.birth;
    });

    std::sort(lifetimes.begin(), lifetimes.end());

    using PcfT = Pcf<T, T>;
    using PcfPointT = typename PcfT::point_type;

    std::vector<PcfPointT> pcfPoints;

    T nAlive = static_cast<T>(lifetimes.size());

    pcfPoints.emplace_back(static_cast<T>(0.), nAlive);

    T lastLifetime = lifetimes.front();
    for (auto const & lifetime : lifetimes)
    {
      if (Barcode<T>::is_infinite(lastLifetime))
      {
        break;
      }

      if (lifetime != lastLifetime)
      {
        pcfPoints.emplace_back(lastLifetime, nAlive);
        lastLifetime = lifetime;
      }
      --nAlive;
    }

    if (!Barcode<T>::is_infinite(lastLifetime))
    {
      pcfPoints.emplace_back(lastLifetime, nAlive);
    }

    return Pcf<T, T>(std::move(pcfPoints));
  }

  template <typename T>
  auto make_stable_rank_task(const Tensor<Barcode<T>>& barcodes, Tensor<Pcf<T, T>>& out)
  {
    return std::make_unique<BarcodeSummaryTask<T, decltype(&barcode_to_stable_rank<T>)>>(
        barcodes, out, barcode_to_stable_rank<T>,
        "Converting barcodes to stable rank functions");
  }
}

#endif //STABLEBEAR_STABLE_RANK_H
