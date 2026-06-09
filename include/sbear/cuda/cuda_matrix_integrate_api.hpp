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

// Pure C++ header — no NVCC required.
// Declares factory functions for CUDA-accelerated block-based matrix integration tasks.
// Implementations live in src/cuda/cuda_matrix_integrate.cu (compiled by NVCC).

#ifndef SB_CUDA_MATRIX_INTEGRATE_API_H
#define SB_CUDA_MATRIX_INTEGRATE_API_H

#include "../task.hpp"
#include "../functional/pcf.hpp"
#include "../tensor.hpp"
#include "../distance_matrix.hpp"
#include "../symmetric_matrix.hpp"

#include <vector>
#include <memory>

namespace sb
{
  // L1 distance → DistanceMatrix — f32
  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_l1_task(
      DistanceMatrix<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& pcfs,
      float32_t a = 0.f,
      float32_t b = std::numeric_limits<float32_t>::max());

  // L1 distance → DistanceMatrix — f64
  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_l1_task(
      DistanceMatrix<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& pcfs,
      float64_t a = 0.0,
      float64_t b = std::numeric_limits<float64_t>::max());

  // Lp distance → DistanceMatrix — f32
  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_lp_task(
      DistanceMatrix<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& pcfs,
      float32_t p,
      float32_t a = 0.f,
      float32_t b = std::numeric_limits<float32_t>::max());

  // Lp distance → DistanceMatrix — f64
  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_lp_task(
      DistanceMatrix<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& pcfs,
      float64_t p,
      float64_t a = 0.0,
      float64_t b = std::numeric_limits<float64_t>::max());

  // L2 inner product → SymmetricMatrix — f32
  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_l2_kernel_task(
      SymmetricMatrix<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& pcfs,
      float32_t a = 0.f,
      float32_t b = std::numeric_limits<float32_t>::max());

  // L2 inner product → SymmetricMatrix — f64
  std::unique_ptr<StoppableTask<void>> create_cuda_block_integrate_l2_kernel_task(
      SymmetricMatrix<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& pcfs,
      float64_t a = 0.0,
      float64_t b = std::numeric_limits<float64_t>::max());

  // === cdist factories — cross-distance/kernel between two sets → dense Tensor ===

  // L1 cdist — f32
  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_l1_task(
      Tensor<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& rowPcfs,
      const std::vector<Pcf<float32_t, float32_t>>& colPcfs,
      float32_t a = 0.f,
      float32_t b = std::numeric_limits<float32_t>::max());

  // L1 cdist — f64
  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_l1_task(
      Tensor<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& rowPcfs,
      const std::vector<Pcf<float64_t, float64_t>>& colPcfs,
      float64_t a = 0.0,
      float64_t b = std::numeric_limits<float64_t>::max());

  // Lp cdist — f32
  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_lp_task(
      Tensor<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& rowPcfs,
      const std::vector<Pcf<float32_t, float32_t>>& colPcfs,
      float32_t p,
      float32_t a = 0.f,
      float32_t b = std::numeric_limits<float32_t>::max());

  // Lp cdist — f64
  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_lp_task(
      Tensor<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& rowPcfs,
      const std::vector<Pcf<float64_t, float64_t>>& colPcfs,
      float64_t p,
      float64_t a = 0.0,
      float64_t b = std::numeric_limits<float64_t>::max());

  // L2 cross-kernel — f32
  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_l2_kernel_task(
      Tensor<float32_t>& out,
      const std::vector<Pcf<float32_t, float32_t>>& rowPcfs,
      const std::vector<Pcf<float32_t, float32_t>>& colPcfs,
      float32_t a = 0.f,
      float32_t b = std::numeric_limits<float32_t>::max());

  // L2 cross-kernel — f64
  std::unique_ptr<StoppableTask<void>> create_cuda_block_cdist_l2_kernel_task(
      Tensor<float64_t>& out,
      const std::vector<Pcf<float64_t, float64_t>>& rowPcfs,
      const std::vector<Pcf<float64_t, float64_t>>& colPcfs,
      float64_t a = 0.0,
      float64_t b = std::numeric_limits<float64_t>::max());
}

#endif
