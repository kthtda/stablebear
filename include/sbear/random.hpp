#ifndef SB_RANDOM_H
#define SB_RANDOM_H

#include "functional/pcf.hpp"
#include "tensor.hpp"
#include "walk.hpp"

#include <algorithm>
#include <vector>
#include <random>

namespace sb
{

  template <typename Tt, typename Tv, typename F>
  void noisy_function(Tensor<Pcf<Tt, Tv>>& out, size_t nPoints, F func,
                      Tv noise = 0.1, DefaultRandomGenerator& gen = default_generator())
  {
    using PcfT = Pcf<Tt, Tv>;
    using PointT = typename PcfT::point_type;

    sb::walk(out, gen, [nPoints, noise, &func, &out](const std::vector<size_t>& idx, auto& engine) {

      std::uniform_real_distribution<Tt> tDist(static_cast<Tt>(0.), static_cast<Tt>(1.));
      std::normal_distribution<Tv> vDist(static_cast<Tv>(0.), noise);

      std::vector<Tt> randomTs(nPoints);
      std::vector<Tv> randomNoises(nPoints);

      std::generate(randomTs.begin(), randomTs.end(), [&engine, &tDist]{ return tDist(engine); });
      std::generate(randomNoises.begin(), randomNoises.end(), [&engine, &vDist]{ return vDist(engine); });

      std::sort(randomTs.begin(), randomTs.end());
      randomTs.front() = 0.;

      std::vector<PointT> pts;
      pts.resize(randomTs.size());
      for (auto i = 0_uz; i < randomTs.size(); ++i)
      {
        pts[i].t = randomTs[i];
        pts[i].v = func(randomTs[i]) + randomNoises[i];
      }

      pts.back().v = 0.;

      out(idx) = PcfT(std::move(pts));
    });
  }

}

#endif
