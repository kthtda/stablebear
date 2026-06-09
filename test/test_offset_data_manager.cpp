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

#include <sbear/cuda/offset_data_manager.hpp>

#include <vector>

namespace
{
  // Simple test element type
  struct TestElement
  {
    int value;
    bool operator==(const TestElement& rhs) const { return value == rhs.value; }
  };

  // Objects with variable-length element arrays
  struct TestObject
  {
    std::vector<TestElement> elements;
  };
}

TEST(OffsetDataManager, InitThreeObjects)
{
  std::vector<TestObject> objects = {
    {{TestElement{1}, TestElement{2}, TestElement{3}}},  // 3 elements
    {{TestElement{4}}},                                   // 1 element
    {{TestElement{5}, TestElement{6}}}                    // 2 elements
  };

  sb::OffsetDataManager<TestElement> mgr;
  mgr.init(objects.begin(), objects.end(),
      [](const TestObject& o) { return o.elements.size(); },
      [](const TestObject& o, size_t i) { return o.elements[i]; });

  EXPECT_EQ(mgr.num_objects(), 3);

  auto const& data = mgr.host_data();

  // Offsets: [0, 3, 4, 6]
  ASSERT_EQ(data.offsets.size(), 4);
  EXPECT_EQ(data.offsets[0], 0);
  EXPECT_EQ(data.offsets[1], 3);
  EXPECT_EQ(data.offsets[2], 4);
  EXPECT_EQ(data.offsets[3], 6);

  // Elements: [1, 2, 3, 4, 5, 6]
  ASSERT_EQ(data.elements.size(), 6);
  EXPECT_EQ(data.elements[0].value, 1);
  EXPECT_EQ(data.elements[1].value, 2);
  EXPECT_EQ(data.elements[2].value, 3);
  EXPECT_EQ(data.elements[3].value, 4);
  EXPECT_EQ(data.elements[4].value, 5);
  EXPECT_EQ(data.elements[5].value, 6);
}

TEST(OffsetDataManager, EmptyInput)
{
  std::vector<TestObject> objects;

  sb::OffsetDataManager<TestElement> mgr;
  mgr.init(objects.begin(), objects.end(),
      [](const TestObject& o) { return o.elements.size(); },
      [](const TestObject& o, size_t i) { return o.elements[i]; });

  EXPECT_EQ(mgr.num_objects(), 0);
  EXPECT_EQ(mgr.host_data().offsets.size(), 1);
  EXPECT_EQ(mgr.host_data().offsets[0], 0);
  EXPECT_TRUE(mgr.host_data().elements.empty());
}

TEST(OffsetDataManager, SingleEmptyObject)
{
  std::vector<TestObject> objects = {{{}}};  // one object with 0 elements

  sb::OffsetDataManager<TestElement> mgr;
  mgr.init(objects.begin(), objects.end(),
      [](const TestObject& o) { return o.elements.size(); },
      [](const TestObject& o, size_t i) { return o.elements[i]; });

  EXPECT_EQ(mgr.num_objects(), 1);
  EXPECT_EQ(mgr.host_data().offsets[0], 0);
  EXPECT_EQ(mgr.host_data().offsets[1], 0);
  EXPECT_TRUE(mgr.host_data().elements.empty());
}

TEST(OffsetDataManager, TotalElementsForRange)
{
  std::vector<TestObject> objects = {
    {{TestElement{1}, TestElement{2}, TestElement{3}}},  // 3
    {{TestElement{4}}},                                   // 1
    {{TestElement{5}, TestElement{6}}}                    // 2
  };

  sb::OffsetDataManager<TestElement> mgr;
  mgr.init(objects.begin(), objects.end(),
      [](const TestObject& o) { return o.elements.size(); },
      [](const TestObject& o, size_t i) { return o.elements[i]; });

  EXPECT_EQ(mgr.total_elements_for_range(0, 3), 6);  // all
  EXPECT_EQ(mgr.total_elements_for_range(0, 1), 3);  // first
  EXPECT_EQ(mgr.total_elements_for_range(1, 1), 1);  // second
  EXPECT_EQ(mgr.total_elements_for_range(2, 1), 2);  // third
  EXPECT_EQ(mgr.total_elements_for_range(0, 2), 4);  // first two
  EXPECT_EQ(mgr.total_elements_for_range(1, 2), 3);  // last two
}

TEST(OffsetDataManager, MaxElementsInRange)
{
  std::vector<TestObject> objects = {
    {{TestElement{1}, TestElement{2}, TestElement{3}}},  // 3
    {{TestElement{4}}},                                   // 1
    {{TestElement{5}, TestElement{6}}}                    // 2
  };

  sb::OffsetDataManager<TestElement> mgr;
  mgr.init(objects.begin(), objects.end(),
      [](const TestObject& o) { return o.elements.size(); },
      [](const TestObject& o, size_t i) { return o.elements[i]; });

  EXPECT_EQ(mgr.max_elements_in_range(0, 3), 3);
  EXPECT_EQ(mgr.max_elements_in_range(1, 2), 2);
  EXPECT_EQ(mgr.max_elements_in_range(1, 1), 1);
}

TEST(OffsetDataManager, WorksWithPrimitiveType)
{
  // Verify it works with plain int, not just structs
  std::vector<std::vector<int>> data = {{10, 20}, {30}, {40, 50, 60}};

  sb::OffsetDataManager<int> mgr;
  mgr.init(data.begin(), data.end(),
      [](const std::vector<int>& v) { return v.size(); },
      [](const std::vector<int>& v, size_t i) { return v[i]; });

  EXPECT_EQ(mgr.num_objects(), 3);
  EXPECT_EQ(mgr.total_elements_for_range(0, 3), 6);

  auto const& elems = mgr.host_data().elements;
  EXPECT_EQ(elems[0], 10);
  EXPECT_EQ(elems[1], 20);
  EXPECT_EQ(elems[2], 30);
  EXPECT_EQ(elems[5], 60);
}
