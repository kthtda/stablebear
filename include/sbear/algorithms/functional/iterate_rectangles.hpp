/*
* Copyright 2024-2026 Bjorn Wehlin
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef SB_ALGORITHM_ITERATE_RECTANGLES_H
#define SB_ALGORITHM_ITERATE_RECTANGLES_H

#include <algorithm>
#include <limits>
#include <vector>
#include "../../functional/rectangle.hpp"

namespace sb
{
  template <typename PcfT>
  constexpr typename PcfT::value_type infinite_value()
  {
    return infinity<typename PcfT::value_type>();
  }

  template <typename PcfT>
  constexpr typename PcfT::time_type infinite_time()
  {
    return infinity<typename PcfT::time_type>();
  }

  template <typename PointBiDirIterator>
  PointBiDirIterator max_time_iterator_prior_to(PointBiDirIterator begin, PointBiDirIterator end,
    typename std::iterator_traits<PointBiDirIterator>::value_type::time_type t)
  {
    using TPoint = typename std::iterator_traits<PointBiDirIterator>::value_type;
    using TTime = typename TPoint::time_type;
    auto it = std::upper_bound(begin, end, t, [](TTime val, const TPoint& p) { return val < p.t; });
    if (it == begin)
    {
      return begin;
    }
    return std::prev(it);
  }

  template <typename TimePointT, typename FCb>
  void iterate_rectangles(const std::vector<TimePointT>& fpts, const std::vector<TimePointT>& gpts, FCb cb,
    typename TimePointT::time_type a = TimePointT::zero_time(), typename TimePointT::time_type b = TimePointT::infinite_time())
  {
    using TTime = typename TimePointT::time_type;
    using TVal = typename TimePointT::value_type;

    auto fi = max_time_iterator_prior_to(fpts.begin(), fpts.end(), a);
    auto gi = max_time_iterator_prior_to(gpts.begin(), gpts.end(), a);

    TTime t = a;
    TTime tprev = a;
    TVal fv = 0;
    TVal gv = 0;

    Rectangle<TTime, TVal> rect;

    while (t < b)
    {
      tprev = t;
      fv = fi->v;
      gv = gi->v;

      auto fi_next = std::next(fi);
      auto gi_next = std::next(gi);

      if (fi_next != fpts.end() && gi_next != gpts.end())
      {
        auto delta = fi_next->t - gi_next->t;
        if (delta <= 0)
        {
          fi = fi_next;
        }
        if (delta >= 0)
        {
          gi = gi_next;
        }
      }
      else if (fi_next != fpts.end())
      {
        fi = fi_next;
      }
      else if (gi_next != gpts.end())
      {
        gi = gi_next;
      }
      else
      {
        rect.left = tprev;
        rect.right = b;
        rect.f_value = fv;
        rect.g_value = gv;

        cb(rect);

        return;
      }

      t = std::min(std::max(fi->t, gi->t), b);

      rect.left = tprev;
      rect.right = t;
      rect.f_value = fv;
      rect.g_value = gv;

      cb(rect);
    }
  }

  template <typename PointBiDirIterator, typename FCb>
  inline void iterate_segments(PointBiDirIterator beginPoints, PointBiDirIterator endPoints,
    typename std::iterator_traits<PointBiDirIterator>::value_type::time_type a,
    typename std::iterator_traits<PointBiDirIterator>::value_type::time_type b,
    FCb cb)
  {
    using TPoint = typename std::iterator_traits<PointBiDirIterator>::value_type;
    using TTime = typename TPoint::time_type;
    using TVal = typename TPoint::value_type;

    if (beginPoints == endPoints)
    {
      return;
    }

    // float32_t (0 * inf) does not produce 0, so cap b at max instead of using infinity
    b = std::min(b, (std::numeric_limits<TTime>::max)());

    // Find k = max{i : ti <= a}
    auto prev = max_time_iterator_prior_to(beginPoints, endPoints, a);
    auto it = std::next(prev);

    TTime tprev = a;
    TVal vprev = prev->v;

    Segment<TTime, TVal> seg;
    for (; it != endPoints; ++it)
    {
      TTime tnext = std::min(it->t, b);
      seg.left = tprev;
      seg.right = tnext;
      seg.value = vprev;

      cb(seg);

      if (tnext >= b)
      {
        return;
      }

      tprev = it->t;
      vprev = it->v;
    }

    seg.left = tprev;
    seg.right = b;
    seg.value = vprev;

    cb(seg);
  }
}

#endif
