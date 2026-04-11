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

#ifndef MPCF_SETTINGS_HPP
#define MPCF_SETTINGS_HPP

#include <cstddef>

namespace mpcf
{
  struct Settings
  {
    bool forceCpu = false;
    size_t cudaThreshold = 500;
    bool deviceVerbose = false;

    unsigned int blockDimX = 1;
    unsigned int blockDimY = 32;

    /// Minimum block side length for the block scheduler.
    /// 0 = auto-detect from GPU hardware (SM count).
    size_t minBlockSide = 0;

    /// Minimum tensor size before tensor_eval uses parallel_walk.
    size_t parallelEvalThreshold = 500;
  };

  inline Settings& settings()
  {
    static Settings instance;
    return instance;
  }
}

#endif
