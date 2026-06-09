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

#include <sbear/special_tensors.hpp>

#include "sbear/tensor_to_string.hpp"

TEST(SpecialTensors, MappingTensor2x3x2)
{
  auto tensor = sb::mapping_tensor<int>({2, 3, 2});

  sb::Tensor<int> expected{tensor.shape()};

  expected({0, 0, 0}) = 0;
  expected({0, 0, 1}) = 1;
  expected({0, 1, 0}) = 10;
  expected({0, 1, 1}) = 11;
  expected({0, 2, 0}) = 20;
  expected({0, 2, 1}) = 21;

  expected({1, 0, 0}) = 100;
  expected({1, 0, 1}) = 101;
  expected({1, 1, 0}) = 110;
  expected({1, 1, 1}) = 111;
  expected({1, 2, 0}) = 120;
  expected({1, 2, 1}) = 121;

  EXPECT_EQ(tensor, expected) << sb::tensor_to_string(tensor);
}
