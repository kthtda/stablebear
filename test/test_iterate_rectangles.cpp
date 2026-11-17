#include <limits>
#include <gtest/gtest.h>

#include <sbear/functional/pcf.hpp>
#include <sbear/algorithms/functional/iterate_rectangles.hpp>

#include <array>
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

namespace
{
  // One rectangle of nonzero width: {left, fRow, gRow}.
  using RowsOfRectangle = std::array<double, 3>;

  std::vector<RowsOfRectangle> rows_of_rectangles(const sb::Pcf_f64& f, const sb::Pcf_f64& g)
  {
    std::vector<RowsOfRectangle> rows;
    sb::iterate_rectangles_with_rows(f.points(), g.points(),
      [&rows](const sb::Pcf_f64::rectangle_type& rect, size_t fRow, size_t gRow) {
        if (rect.right > rect.left)
        {
          rows.push_back({rect.left, static_cast<double>(fRow), static_cast<double>(gRow)});
        }
      });
    return rows;
  }
}

TEST(IterateRectanglesWithRows, ReportsTheBreakpointsOfEachRectangle)
{
  sb::Pcf_f64 f{{0., 3.}, {1., 2.}, {4., 5.}, {6., 0.}};
  sb::Pcf_f64 g{{0., 2.}, {3., 4.}, {4., 2.}, {5., 1.}};

  std::vector<RowsOfRectangle> expected = {
    {0., 0., 0.},
    {1., 1., 0.},
    {3., 1., 1.},
    {4., 2., 2.},
    {5., 2., 3.},
    {6., 3., 3.},
  };
  EXPECT_EQ(rows_of_rectangles(f, g), expected);
}

TEST(IterateRectanglesWithRows, OneFunctionEndsFirst)
{
  sb::Pcf_f64 f{{0., 3.}};
  sb::Pcf_f64 g{{0., 2.}, {1., 4.}, {2., 1.}};

  std::vector<RowsOfRectangle> expected = {
    {0., 0., 0.},
    {1., 0., 1.},
    {2., 0., 2.},
  };
  EXPECT_EQ(rows_of_rectangles(f, g), expected);
}

TEST(IterateRectanglesWithRows, BothFunctionsHaveABreakpointAtInfinity)
{
  const double inf = std::numeric_limits<double>::infinity();
  sb::Pcf_f64 f{{0., 3.}, {1., 2.}, {inf, 0.}};
  sb::Pcf_f64 g{{0., 2.}, {2., 4.}, {inf, 0.}};

  std::vector<RowsOfRectangle> expected = {
    {0., 0., 0.},
    {1., 1., 0.},
    {2., 1., 1.},
  };
  EXPECT_EQ(rows_of_rectangles(f, g), expected);
}

TEST(IterateRectanglesWithRows, MatchesIterateRectangles)
{
  sb::Pcf_f64 f{{0., 3.}, {1., 2.}, {4., 5.}, {6., 0.}};
  sb::Pcf_f64 g{{0., 2.}, {3., 4.}, {4., 2.}, {5., 1.}, {8., 3.}};

  std::vector<sb::Pcf_f64::rectangle_type> plain;
  sb::iterate_rectangles(f.points(), g.points(),
    [&plain](const sb::Pcf_f64::rectangle_type& rect) { plain.push_back(rect); });

  std::vector<sb::Pcf_f64::rectangle_type> withRows;
  sb::iterate_rectangles_with_rows(f.points(), g.points(),
    [&](const sb::Pcf_f64::rectangle_type& rect, size_t fRow, size_t gRow) {
      EXPECT_EQ(rect.f_value, f.points()[fRow].v);
      EXPECT_EQ(rect.g_value, g.points()[gRow].v);
      withRows.push_back(rect);
    });

  EXPECT_EQ(withRows, plain);
}
