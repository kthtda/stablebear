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


#ifndef SB_CUDA_UTIL_H
#define SB_CUDA_UTIL_H

#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

#define CHK_CUDA(x) { auto rv = x; if (rv != cudaSuccess) { \
  throw std::runtime_error(__FILE__ + std::string(":") + std::to_string(__LINE__) + ": " +  cudaGetErrorString(rv)); \
  } }

#endif
