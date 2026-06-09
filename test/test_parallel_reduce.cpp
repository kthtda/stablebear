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

#include <gtest/gtest.h>

#include <sbear/functional/pcf.hpp>
#include <sbear/algorithms/functional/reduce.hpp>

TEST(ParallelReduce, AddThreeFunctions)
{
  std::vector<sb::Pcf_f64> pcfs;

  pcfs.emplace_back(sb::Pcf_f64{ {0., 3.}, {1., 2.}, {4., 5.}, {6., 0.} });
  pcfs.emplace_back(sb::Pcf_f64{ {0., 2.}, {3., 4.}, {4., 2.}, {5., 1.}, {8., 3.} });
  pcfs.emplace_back(sb::Pcf_f64{ {0., 0.}, {3., 7.}, {5., 2.} });

  auto res = sb::parallel_reduce(pcfs.begin(), pcfs.end(), [](const typename sb::Pcf_f64::rectangle_type& rect) {
    return rect.f_value + rect.g_value;
    });

  EXPECT_EQ(res, (pcfs[0] + pcfs[1] + pcfs[2]));
}
