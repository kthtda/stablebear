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

#ifndef SB_ALGORITHM_REDUCE_H
#define SB_ALGORITHM_REDUCE_H

#include "iterate_rectangles.hpp"
#include "../subdivide.hpp"
#include "../../executor.hpp"

#include <functional>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iterator>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/reduce.hpp>
#include <taskflow/algorithm/for_each.hpp>

namespace sb
{
  template <typename TPcf>
  using TOp = std::function<typename TPcf::value_type(const typename TPcf::rectangle_type&)>;

  template <typename TPcf>
  TPcf combine(const TPcf& f, const TPcf& g, TOp<TPcf> op)
  {
    using point_type = typename TPcf::point_type;
    using rectangle_type = typename TPcf::rectangle_type;
    using value_type = typename point_type::value_type;

    std::vector<point_type> retPts;

    std::size_t npts = 0_uz;
    iterate_rectangles(f.points(), g.points(), [&npts](const rectangle_type&){ ++npts; });
    retPts.reserve(npts);

    value_type vLast = static_cast<value_type>(0);
    bool first = true;

    iterate_rectangles(f.points(), g.points(), [&retPts, &op, &vLast, &first](const rectangle_type& rect){
      auto v = op(rect);

      // Avoid inserting duplicate plateaus with the same value, except at t=0
      if (first || v != vLast || rect.left == 0)
      {
        retPts.emplace_back(rect.left, v);
        first = false;
      }

      vLast = v;
    });

    retPts.shrink_to_fit();

    return TPcf(std::move(retPts));
  }

  template <typename TPcf>
  TPcf reduce(const std::vector<TPcf>& fs, TOp<TPcf> op)
  {
    return std::reduce(fs.begin(), fs.end(), TPcf(), [&op](const TPcf& f, const TPcf& g) {
      return combine(f, g, op);
    });
  }

  template <typename TPcf>
  class Accumulator
  {
  public:
    using point_type = typename TPcf::point_type;
    using time_type = typename TPcf::time_type;

    Accumulator(TOp<TPcf> op, size_t sizeHint)
      : m_op(op)
    { 
      m_pts.reserve(sizeHint);
      m_pts_temp.reserve(sizeHint);
    }

    Accumulator(TOp<TPcf> op, const TPcf& f)
      : m_op(op)
      , m_pts(f.points())
    {
    }

    Accumulator& operator+=(const TPcf& f)
    {
      if (m_pts.empty())
      {
        m_pts = f.points();
        return *this;
      }
      combine_with_(f.points());
      return *this;
    }

    Accumulator& operator+=(const Accumulator& other)
    {
      combine_with_(other.m_pts);
      return *this;
    }

    Accumulator& operator+=(Accumulator&& other)
    {
      combine_with_(other.m_pts);
      return *this;
    }

    Accumulator(Accumulator&& other)
      : m_op(other.m_op)
      , m_pts(std::move(other.m_pts))
      , m_pts_temp(std::move(other.m_pts_temp))
    { }

    Accumulator(const Accumulator& other)
      : m_op(other.m_op)
      , m_pts(other.m_pts)
      , m_pts_temp(other.m_pts_temp)
    { }

    Accumulator& operator=(Accumulator&& other)
    {
      if (this == &other)
      {
        return *this;
      }
      m_op = other.m_op;
      m_pts = std::move(other.m_pts);
      m_pts_temp = std::move(other.m_pts_temp);
      return *this;
    }

    Accumulator& operator=(const Accumulator& other)
    {
      if (this == &other)
      {
        return *this;
      }
      m_op = other.m_op;
      m_pts = other.m_pts;
      m_pts_temp = other.m_pts_temp;
      return *this;
    }

    void combine_with_(const std::vector<point_type>& other)
    {
      using rectangle_type = typename TPcf::rectangle_type;
      using value_type = typename point_type::value_type;

      auto i = 0ul;
      m_pts_temp.clear();
      m_pts_temp.resize(m_pts.size() + other.size() + 1);

      value_type vLast = 0;

      iterate_rectangles(m_pts, other, [&i, &vLast, this](const rectangle_type& rect){

        auto t = rect.left;
        auto v = m_op(rect);
        
        if (v != vLast || rect.left == 0)
        {
          m_pts_temp[i].t = t;
          m_pts_temp[i].v = v;

          ++i;
        }

        vLast = v;
      });

      m_pts_temp.resize(i);
      m_pts_temp.swap(m_pts);
    }

    TOp<TPcf> m_op;
    std::vector<point_type> m_pts;
    std::vector<point_type> m_pts_temp;
  };

  struct TaskWithTag
  {
    tf::Task task;
    size_t tag;
  };

  // TODO: FwdIt
  template <typename RandomAccessIt>
  typename std::iterator_traits<RandomAccessIt>::value_type parallel_reduce(RandomAccessIt begin, RandomAccessIt end, TOp<typename std::iterator_traits<RandomAccessIt>::value_type> op, size_t chunkszFirst = 8, Executor& exec = default_executor())
  {
    using pcf_type = typename std::iterator_traits<RandomAccessIt>::value_type;

    auto chunksz = 2;
    auto sz = std::distance(begin, end);
    auto blocks = subdivide(chunkszFirst , sz);

    auto nAll = std::accumulate(begin, end, static_cast<size_t>(0ul), [](size_t n, const pcf_type& f){ return f.points().size() + n; });

    std::vector<Accumulator<pcf_type>> accumulators;
    accumulators.resize(blocks.size(), Accumulator<pcf_type>(op, nAll));

    tf::Taskflow taskflow;

    std::list<std::vector<TaskWithTag>> taskLevels;
    auto & topLevelTasks = taskLevels.emplace_back(accumulators.size());
    for (auto iBlock = 0ul; iBlock < blocks.size(); ++iBlock)
    {
      auto const & block = blocks[iBlock];
      topLevelTasks[iBlock].tag = iBlock; // Accumulator id
      topLevelTasks[iBlock].task = taskflow.emplace([iBlock, block, &accumulators, begin](){
        for (auto i = block.first; i <= block.second; ++i)
        {
          accumulators[iBlock] += *(begin + i); // TODO: FwdIt
        }
      });
    }

    while (taskLevels.back().size() > 1ul)
    {
      auto const & prevLevel = taskLevels.back();
      blocks = subdivide(chunksz, prevLevel.size());
      
      auto & tasks = taskLevels.emplace_back(blocks.size());

      for (auto iBlock = 0ul; iBlock < blocks.size(); ++iBlock)
      {
        auto const & block = blocks[iBlock];
        auto targetAcc = prevLevel[block.first].tag;
        tasks[iBlock].tag = targetAcc;
        tasks[iBlock].task = taskflow.emplace([targetAcc, block, &accumulators, &prevLevel](){
          for (auto i = block.first + 1; i <= block.second; ++i)
          {
            auto srcAcc = prevLevel[i].tag;
            accumulators[targetAcc] += accumulators[srcAcc];
          }
        });
        for (auto iPrev = block.first; iPrev <= block.second; ++iPrev)
        {
          tasks[iBlock].task.succeed(prevLevel[iPrev].task);
        }
      }

    }

    exec.cpu()->run(taskflow).wait();

    return pcf_type(std::move(accumulators[0].m_pts));
  }
}

#endif
