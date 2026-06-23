#ifndef SB_SETTINGS_HPP
#define SB_SETTINGS_HPP

#include <cstddef>

namespace sb
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
