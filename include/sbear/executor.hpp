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

#ifndef SB_EXECUTOR_H
#define SB_EXECUTOR_H

#include "platform.hpp"
#include <taskflow/core/executor.hpp>
#include <memory>

namespace sb
{
  enum class Hardware
  {
    CPU,
    CUDA,
    PREFER_CUDA
  };
  
  size_t get_num_cuda_devices();
  
  /// An Executor instance holds a tf::Executor for CPU-based execution,
  /// and (optionally) a separate tf::Executor for GPU-based execution.
  /// The GPU executor has the same number of threads as there are GPUs,
  /// and each thread operates on only one of the GPUs.
  class Executor
  {
  public:   
    /// Construct an Executor from an existing tf::Executor. It is the
    /// client's responsibility to ensure that the tf::Executor stays
    /// alive for the entire lifetime of this Executor object.
    explicit Executor(tf::Executor* tfExec, size_t nCudaDevices = get_num_cuda_devices()) noexcept
      : m_upCpuExec(nullptr)
      , m_cpuExec(tfExec)
      , m_upCudaExec(nCudaDevices > 0 ? std::make_unique<tf::Executor>(nCudaDevices) : nullptr)
    {
      
    }
    
    Executor(size_t nThreads = std::thread::hardware_concurrency(), size_t nCudaDevices = get_num_cuda_devices())
      : m_upCpuExec(std::make_unique<tf::Executor>(nThreads))
      , m_cpuExec(m_upCpuExec.get())
      , m_upCudaExec(nCudaDevices > 0 ? std::make_unique<tf::Executor>(nCudaDevices) : nullptr)
    {
      
    }
    
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    
    Executor(Executor&& other) noexcept
      : m_upCpuExec(std::move(other.m_upCpuExec))
      , m_cpuExec(other.m_cpuExec)
      , m_upCudaExec(std::move(other.m_upCudaExec))
    {
      other.m_cpuExec = nullptr;
    }

    Executor& operator=(Executor&& rhs) noexcept
    {
      if (this == &rhs)
      {
        return *this;
      }

      m_cpuExec = rhs.m_cpuExec; // capture before unique_ptr move
      m_upCpuExec = std::move(rhs.m_upCpuExec);
      rhs.m_cpuExec = nullptr;

      m_upCudaExec = std::move(rhs.m_upCudaExec);

      return *this;
    }
    
    tf::Executor* cpu() noexcept
    {
      return m_cpuExec;
    }
    
    tf::Executor* cuda() noexcept
    {
      return m_upCudaExec.get();
    }
    
    void limit_cpu_workers(size_t nWorkers)
    {
      if (m_upCpuExec)
      {
        m_upCpuExec->wait_for_all();
        m_cpuExec = nullptr;
        m_upCpuExec = std::make_unique<tf::Executor>(nWorkers);
        m_cpuExec = m_upCpuExec.get();
      }
      else
      {
        throw std::runtime_error("Unable to set worker count on external pool.");
      }
    }
    
    void limit_cuda_workers(size_t nWorkers)
    {
      if (nWorkers > get_num_cuda_devices())
      {
        throw std::runtime_error("Requested more CUDA workers than there are GPUs. (requested " + std::to_string(nWorkers) + " workers but there are only " + std::to_string(get_num_cuda_devices()) + " available");
      }
      
      if (m_upCudaExec)
      {
        m_upCudaExec->wait_for_all();
      }
      m_upCudaExec = std::make_unique<tf::Executor>(nWorkers);
    }
    
  private:
    std::unique_ptr<tf::Executor> m_upCpuExec;
    tf::Executor* m_cpuExec;
    
    std::unique_ptr<tf::Executor> m_upCudaExec;
  };
  
  SB_EXPORT_API Executor& default_executor();
}

#endif
