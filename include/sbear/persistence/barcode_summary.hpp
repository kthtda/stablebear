// Copyright 2024-2026 Bjorn Wehlin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STABLEBEAR_BARCODE_SUMMARY_H
#define STABLEBEAR_BARCODE_SUMMARY_H

#include "../task.hpp"
#include "../functional/pcf.hpp"
#include "barcode.hpp"

#include <string>

namespace sb::ph
{
  /**
   * Generic task that applies a barcode-to-PCF transformation across a
   * tensor of barcodes in parallel.
   *
   * @tparam T       Scalar type (e.g. float32_t, float64_t)
   * @tparam Func    A callable with signature Pcf<T,T>(const Barcode<T>&)
   */
  template <typename T, typename Func>
  class BarcodeSummaryTask : public StoppableTask<void>
  {
  public:
    BarcodeSummaryTask(const Tensor<Barcode<T>>& barcodes, Tensor<Pcf<T, T>>& ret,
                       Func func, std::string progressLabel)
        : m_barcodes(barcodes), m_ret(ret), m_func(std::move(func)),
          m_progressLabel(std::move(progressLabel))
    {
    }

  private:
    tf::Future<void> run_async(Executor& exec) override
    {
      tf::Taskflow flow;

      next_step(m_barcodes.size(), m_progressLabel, "barcode");

      m_ret = Tensor<Pcf<T, T>>(m_barcodes.shape());

      sb::walk(m_barcodes, [this, &flow](const std::vector<size_t>& index) {
        flow.emplace([this, index] {
          m_ret(index) = m_func(m_barcodes(index));
          add_progress(1);
        });
      });

      return exec.cpu()->run(std::move(flow));
    }

    const Tensor<Barcode<T>>& m_barcodes;
    Tensor<Pcf<T, T>>& m_ret;
    Func m_func;
    std::string m_progressLabel;
  };

} // namespace sb::ph

#endif // STABLEBEAR_BARCODE_SUMMARY_H
