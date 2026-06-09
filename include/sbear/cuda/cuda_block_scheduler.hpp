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

#ifndef SB_CUDA_BLOCK_SCHEDULER_HPP
#define SB_CUDA_BLOCK_SCHEDULER_HPP

#include "../algorithms/subdivide.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace sb
{
  struct BlockInfo
  {
    size_t rowStart;
    size_t rowHeight;
    size_t colStart;
    size_t colWidth;
    size_t blockIndex;
  };

  enum class BlockTriangleMode
  {
    LowerTriangle,
    Full
  };

  class CudaBlockScheduler
  {
  public:
    struct Config
    {
      size_t nRows;
      size_t nCols;
      size_t maxOutputElements;
      size_t nSplitsHint;
      BlockTriangleMode triangleMode = BlockTriangleMode::Full;
      size_t minBlockSide = 0;  ///< Minimum block side length for GPU occupancy. 0 = no floor.
    };

    explicit CudaBlockScheduler(const Config& config)
      : m_maxRowHeight(0)
      , m_maxColWidth(0)
      , m_triangleMode(config.triangleMode)
    {
      compute_blocks(config);
    }

    [[nodiscard]] const std::vector<BlockInfo>& blocks() const { return m_blocks; }
    [[nodiscard]] size_t max_row_height() const { return m_maxRowHeight; }
    [[nodiscard]] size_t max_col_width() const { return m_maxColWidth; }
    [[nodiscard]] BlockTriangleMode triangle_mode() const { return m_triangleMode; }

  private:
    void compute_blocks(const Config& config)
    {
      if (config.nRows == 0 || config.nCols == 0)
      {
        return;
      }

      auto blockSide = compute_block_side(config.maxOutputElements, config.nSplitsHint,
                                          std::max(config.nRows, config.nCols),
                                          config.minBlockSide);
      auto rowBands = subdivide(blockSide, config.nRows);
      auto colBands = subdivide(blockSide, config.nCols);

      size_t idx = 0;
      for (auto const& rowBand : rowBands)
      {
        size_t rowHeight = rowBand.second - rowBand.first + 1;
        for (auto const& colBand : colBands)
        {
          if (config.triangleMode == BlockTriangleMode::LowerTriangle
              && colBand.first > rowBand.second)
          {
            continue;
          }

          size_t colWidth = colBand.second - colBand.first + 1;

          BlockInfo block;
          block.rowStart = rowBand.first;
          block.rowHeight = rowHeight;
          block.colStart = colBand.first;
          block.colWidth = colWidth;
          block.blockIndex = idx++;

          m_blocks.push_back(block);
          m_maxRowHeight = std::max(m_maxRowHeight, rowHeight);
          m_maxColWidth = std::max(m_maxColWidth, colWidth);
        }
      }

      sort_by_descending_work();
    }

    static size_t compute_block_side(size_t maxOutputElements, size_t nSplitsHint,
                                     size_t maxDim, size_t minBlockSide)
    {
      size_t elementsPerBlock = maxOutputElements / std::max<size_t>(nSplitsHint, 1);
      size_t side = static_cast<size_t>(std::sqrt(static_cast<double>(elementsPerBlock)));
      side = std::max<size_t>(side, 1);
      if (minBlockSide > 0)
      {
        side = std::max(side, minBlockSide);
      }
      side = std::min(side, maxDim);
      return side;
    }

    void sort_by_descending_work()
    {
      std::sort(m_blocks.begin(), m_blocks.end(), [](const BlockInfo& a, const BlockInfo& b) {
        return (a.rowHeight * a.colWidth) > (b.rowHeight * b.colWidth);
      });

      for (size_t i = 0; i < m_blocks.size(); ++i)
      {
        m_blocks[i].blockIndex = i;
      }
    }

    std::vector<BlockInfo> m_blocks;
    size_t m_maxRowHeight;
    size_t m_maxColWidth;
    BlockTriangleMode m_triangleMode;
  };
}

#endif
