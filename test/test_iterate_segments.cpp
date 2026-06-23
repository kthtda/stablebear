#include <gtest/gtest.h>

#include <sbear/functional/pcf.hpp>
#include <sbear/algorithms/functional/iterate_rectangles.hpp>

#include <limits>
#include <vector>

class IterateSegmentsFixture : public ::testing::Test
{
protected:
  using Segment = sb::Segment<double, double>;
  using Point = typename sb::Pcf_f64::point_type;

  void SetUp() override
  {
    collectSegment = [this](const Segment& seg){ segments.emplace_back(seg); };

    // PCF: value 3 on [0,1), 2 on [1,4), 5 on [4,6), 0 on [6,inf)
    pts = { {0., 3.}, {1., 2.}, {4., 5.}, {6., 0.} };
  }

  std::vector<Point> pts;
  std::vector<Segment> segments;

  std::function<void(const Segment&)> collectSegment;
};

TEST_F(IterateSegmentsFixture, Full)
{
  sb::iterate_segments(pts.begin(), pts.end(), 0., Point::infinite_time(), collectSegment);

  ASSERT_EQ(segments.size(), 4u);

  EXPECT_EQ(segments[0].left,  0.); EXPECT_EQ(segments[0].right, 1.); EXPECT_EQ(segments[0].value, 3.);
  EXPECT_EQ(segments[1].left,  1.); EXPECT_EQ(segments[1].right, 4.); EXPECT_EQ(segments[1].value, 2.);
  EXPECT_EQ(segments[2].left,  4.); EXPECT_EQ(segments[2].right, 6.); EXPECT_EQ(segments[2].value, 5.);
  EXPECT_EQ(segments[3].left,  6.); EXPECT_EQ(segments[3].right, Point::infinite_time()); EXPECT_EQ(segments[3].value, 0.);
}

TEST_F(IterateSegmentsFixture, StartLate)
{
  sb::iterate_segments(pts.begin(), pts.end(), 2., Point::infinite_time(), collectSegment);

  ASSERT_EQ(segments.size(), 3u);

  EXPECT_EQ(segments[0].left,  2.); EXPECT_EQ(segments[0].right, 4.); EXPECT_EQ(segments[0].value, 2.);
  EXPECT_EQ(segments[1].left,  4.); EXPECT_EQ(segments[1].right, 6.); EXPECT_EQ(segments[1].value, 5.);
  EXPECT_EQ(segments[2].left,  6.); EXPECT_EQ(segments[2].right, Point::infinite_time()); EXPECT_EQ(segments[2].value, 0.);
}

TEST_F(IterateSegmentsFixture, EndEarly)
{
  sb::iterate_segments(pts.begin(), pts.end(), 0., 5., collectSegment);

  ASSERT_EQ(segments.size(), 3u);

  EXPECT_EQ(segments[0].left,  0.); EXPECT_EQ(segments[0].right, 1.); EXPECT_EQ(segments[0].value, 3.);
  EXPECT_EQ(segments[1].left,  1.); EXPECT_EQ(segments[1].right, 4.); EXPECT_EQ(segments[1].value, 2.);
  EXPECT_EQ(segments[2].left,  4.); EXPECT_EQ(segments[2].right, 5.); EXPECT_EQ(segments[2].value, 5.);
}

TEST_F(IterateSegmentsFixture, StartLateAndEndEarly)
{
  sb::iterate_segments(pts.begin(), pts.end(), 2., 5., collectSegment);

  ASSERT_EQ(segments.size(), 2u);

  EXPECT_EQ(segments[0].left,  2.); EXPECT_EQ(segments[0].right, 4.); EXPECT_EQ(segments[0].value, 2.);
  EXPECT_EQ(segments[1].left,  4.); EXPECT_EQ(segments[1].right, 5.); EXPECT_EQ(segments[1].value, 5.);
}

TEST_F(IterateSegmentsFixture, StartAfterEverything)
{
  sb::iterate_segments(pts.begin(), pts.end(), 10., Point::infinite_time(), collectSegment);

  ASSERT_EQ(segments.size(), 1u);

  EXPECT_EQ(segments[0].left,  10.); EXPECT_EQ(segments[0].right, Point::infinite_time()); EXPECT_EQ(segments[0].value, 0.);
}

TEST_F(IterateSegmentsFixture, StartOnPoint)
{
  // Starting exactly on a breakpoint should use that point's value
  sb::iterate_segments(pts.begin(), pts.end(), 4., Point::infinite_time(), collectSegment);

  ASSERT_EQ(segments.size(), 2u);

  EXPECT_EQ(segments[0].left,  4.); EXPECT_EQ(segments[0].right, 6.); EXPECT_EQ(segments[0].value, 5.);
  EXPECT_EQ(segments[1].left,  6.); EXPECT_EQ(segments[1].right, Point::infinite_time()); EXPECT_EQ(segments[1].value, 0.);
}

TEST_F(IterateSegmentsFixture, StartBeforeFirst)
{
  // Starting before the first breakpoint: value should be taken from the first point
  sb::iterate_segments(pts.begin(), pts.end(), -1., 2., collectSegment);

  ASSERT_EQ(segments.size(), 2u);

  EXPECT_EQ(segments[0].left, -1.); EXPECT_EQ(segments[0].right, 1.); EXPECT_EQ(segments[0].value, 3.);
  EXPECT_EQ(segments[1].left,  1.); EXPECT_EQ(segments[1].right, 2.); EXPECT_EQ(segments[1].value, 2.);
}

TEST_F(IterateSegmentsFixture, SinglePoint)
{
  std::vector<Point> single = { {0., 7.} };
  sb::iterate_segments(single.begin(), single.end(), 0., Point::infinite_time(), collectSegment);

  ASSERT_EQ(segments.size(), 1u);

  EXPECT_EQ(segments[0].left,  0.); EXPECT_EQ(segments[0].right, Point::infinite_time()); EXPECT_EQ(segments[0].value, 7.);
}

TEST_F(IterateSegmentsFixture, Empty)
{
  std::vector<Point> empty;
  sb::iterate_segments(empty.begin(), empty.end(), 0., Point::infinite_time(), collectSegment);

  EXPECT_EQ(segments.size(), 0u);
}
