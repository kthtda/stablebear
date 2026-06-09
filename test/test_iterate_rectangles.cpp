/*
* Copyright 2024 Bjorn Wehlin
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
#include <sbear/algorithms/functional/iterate_rectangles.hpp>

#include <vector>

class IterateRectanglesFixture : public ::testing::Test
{
protected:
  using Rectangle = typename sb::Pcf_f64::rectangle_type;
  using Point = typename sb::Pcf_f64::point_type;
  
  void SetUp() override
  {
    collectRectangle = [this](const Rectangle& rect){ rectangles.emplace_back(rect); };
    
    pcfs.emplace_back(sb::Pcf_f64{{0., 3.}, {1., 2.}, {4., 5.}, {6., 0.}});
    pcfs.emplace_back(sb::Pcf_f64{{0., 2.}, {3., 4.}, {4., 2.}, {5., 1.}, {8., 3.}});
  }
  
  std::vector<sb::Pcf_f64> pcfs;
  std::vector<Rectangle> rectangles;
  
  std::function<void(const Rectangle&)> collectRectangle;
};

TEST_F(IterateRectanglesFixture, Full)
{
  sb::iterate_rectangles(pcfs[0].points(), pcfs[1].points(), collectRectangle);
  
  ASSERT_EQ(rectangles.size(), 7);
  
  EXPECT_EQ(rectangles[0], Rectangle().l(0.).r(1.).fv(3.).gv(2.));
  EXPECT_EQ(rectangles[1], Rectangle().l(1.).r(3.).fv(2.).gv(2.));
  EXPECT_EQ(rectangles[2], Rectangle().l(3.).r(4.).fv(2.).gv(4.));
  EXPECT_EQ(rectangles[3], Rectangle().l(4.).r(5.).fv(5.).gv(2.));
  EXPECT_EQ(rectangles[4], Rectangle().l(5.).r(6.).fv(5.).gv(1.));
  EXPECT_EQ(rectangles[5], Rectangle().l(6.).r(8.).fv(0.).gv(1.));
  EXPECT_EQ(rectangles[6], Rectangle().l(8.).r(Point::infinite_time()).fv(0.).gv(3.));
}

TEST_F(IterateRectanglesFixture, EndEarly)
{
  sb::iterate_rectangles(pcfs[0].points(), pcfs[1].points(), collectRectangle, 0, 4.5);
  
  ASSERT_EQ(rectangles.size(), 4);
  
  EXPECT_EQ(rectangles[0], Rectangle().l(0.).r(1.).fv(3.).gv(2.));
  EXPECT_EQ(rectangles[1], Rectangle().l(1.).r(3.).fv(2.).gv(2.));
  EXPECT_EQ(rectangles[2], Rectangle().l(3.).r(4.).fv(2.).gv(4.));
  EXPECT_EQ(rectangles[3], Rectangle().l(4.).r(4.5).fv(5.).gv(2.));
}

TEST_F(IterateRectanglesFixture, StartBeforeFirst)
{
  sb::iterate_rectangles(pcfs[0].points(), pcfs[1].points(), collectRectangle, -1., 1.);

  ASSERT_EQ(rectangles.size(), 1);

  EXPECT_EQ(rectangles[0], Rectangle().l(-1.).r(1.).fv(3.).gv(2.));
}

TEST_F(IterateRectanglesFixture, StartLate)
{
  sb::iterate_rectangles(pcfs[0].points(), pcfs[1].points(), collectRectangle, 2.);
  
  ASSERT_EQ(rectangles.size(), 6);
  
  EXPECT_EQ(rectangles[0], Rectangle().l(2.).r(3.).fv(2.).gv(2.));
  EXPECT_EQ(rectangles[1], Rectangle().l(3.).r(4.).fv(2.).gv(4.));
  EXPECT_EQ(rectangles[2], Rectangle().l(4.).r(5.).fv(5.).gv(2.));
  EXPECT_EQ(rectangles[3], Rectangle().l(5.).r(6.).fv(5.).gv(1.));
  EXPECT_EQ(rectangles[4], Rectangle().l(6.).r(8.).fv(0.).gv(1.));
  EXPECT_EQ(rectangles[5], Rectangle().l(8.).r(Point::infinite_time()).fv(0.).gv(3.));
}

TEST_F(IterateRectanglesFixture, StartLateAndEndEarly)
{
  sb::iterate_rectangles(pcfs[0].points(), pcfs[1].points(), collectRectangle, 2., 4.5);

  ASSERT_EQ(rectangles.size(), 3);

  EXPECT_EQ(rectangles[0], Rectangle().l(2.).r(3.).fv(2.).gv(2.));
  EXPECT_EQ(rectangles[1], Rectangle().l(3.).r(4.).fv(2.).gv(4.));
  EXPECT_EQ(rectangles[2], Rectangle().l(4.).r(4.5).fv(5.).gv(2.));
}

TEST_F(IterateRectanglesFixture, StartAfterEverything)
{
  sb::iterate_rectangles(pcfs[0].points(), pcfs[1].points(), collectRectangle, 10.);

  ASSERT_EQ(rectangles.size(), 1);

  EXPECT_EQ(rectangles[0], Rectangle().l(10.).r(Point::infinite_time()).fv(0.).gv(3.));
}
