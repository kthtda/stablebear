#include <gtest/gtest.h>

#include <sbear/functional/plf.hpp>

#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

  using PlfTypes = ::testing::Types<sb::Plf_f32, sb::Plf_f64>;

  template<typename PlfT>
  class PlfTest : public ::testing::Test
  {
  };

  TYPED_TEST_SUITE(PlfTest, PlfTypes);

  // ============================================================================
  // Construction
  // ============================================================================

  TYPED_TEST(PlfTest, DefaultIsZeroFunction)
  {
    using PlfT = TypeParam;
    using Pt = typename PlfT::point_type;

    PlfT f;

    EXPECT_EQ(f.points(), std::vector<Pt>{ Pt(0, 0) });
  }

  TYPED_TEST(PlfTest, NoBreakpointsThrows)
  {
    using PlfT = TypeParam;
    using Pt = typename PlfT::point_type;

    EXPECT_THROW(PlfT(std::vector<Pt>{}), std::invalid_argument);
  }

  TYPED_TEST(PlfTest, NonzeroFirstTimeThrows)
  {
    using PlfT = TypeParam;

    EXPECT_THROW(PlfT({ { 1, 10 }, { 2, 20 } }), std::invalid_argument);
  }

  TYPED_TEST(PlfTest, RepeatedTimeThrows)
  {
    using PlfT = TypeParam;

    EXPECT_THROW(PlfT({ { 0, 0 }, { 2, 20 }, { 2, 30 } }), std::invalid_argument);
  }

  TYPED_TEST(PlfTest, DecreasingTimeThrows)
  {
    using PlfT = TypeParam;

    EXPECT_THROW(PlfT({ { 0, 0 }, { 2, 20 }, { 1, 10 } }), std::invalid_argument);
  }

  TYPED_TEST(PlfTest, InfiniteTimeThrows)
  {
    using PlfT = TypeParam;
    using T = typename PlfT::time_type;

    EXPECT_THROW(PlfT({ { 0, 0 }, { std::numeric_limits<T>::infinity(), 20 } }), std::invalid_argument);
  }

  // ============================================================================
  // Evaluation
  // ============================================================================

  TYPED_TEST(PlfTest, ValueAtEachBreakpoint)
  {
    using PlfT = TypeParam;

    PlfT f({ { 0, 0 }, { 2, 20 }, { 4, 10 } });

    EXPECT_EQ(f.evaluate(0), 0);
    EXPECT_EQ(f.evaluate(2), 20);
    EXPECT_EQ(f.evaluate(4), 10);
  }

  TYPED_TEST(PlfTest, ValueHalfwayBetweenBreakpoints)
  {
    using PlfT = TypeParam;

    PlfT f({ { 0, 0 }, { 2, 20 }, { 4, 10 } });

    EXPECT_EQ(f.evaluate(1), 10);  // halfway from 0 to 20
    EXPECT_EQ(f.evaluate(3), 15);  // halfway from 20 to 10
  }

  TYPED_TEST(PlfTest, AfterLastBreakpointContinuesLastSegment)
  {
    using PlfT = TypeParam;

    PlfT f({ { 0, 0 }, { 1, 2 }, { 2, 1 } });

    // The last segment has slope -1.
    EXPECT_EQ(f.evaluate(3), 0);
    EXPECT_EQ(f.evaluate(4), -1);
  }

  TYPED_TEST(PlfTest, FinalFlatSegmentKeepsValueConstant)
  {
    using PlfT = TypeParam;

    PlfT f({ { 0, 0 }, { 2, 20 }, { 4, 20 } });

    EXPECT_EQ(f.evaluate(8), 20);
  }

  TYPED_TEST(PlfTest, SingleBreakpointIsConstant)
  {
    using PlfT = TypeParam;

    PlfT f({ { 0, 10 } });

    EXPECT_EQ(f.evaluate(0), 10);
    EXPECT_EQ(f.evaluate(5), 10);
  }

  TYPED_TEST(PlfTest, NegativeTimeThrows)
  {
    using PlfT = TypeParam;

    PlfT f({ { 0, 0 }, { 2, 20 } });

    EXPECT_THROW((void)f.evaluate(-1), std::domain_error);
  }

  TYPED_TEST(PlfTest, SortedTimesMatchScalarEvaluate)
  {
    using PlfT = TypeParam;
    using T = typename PlfT::value_type;

    PlfT f({ { 0, 0 }, { 2, 20 }, { 4, 10 } });

    // At and between breakpoints, and after the last one.
    std::vector<T> times = { 0, 1, 2, 3, 4, 8 };
    std::vector<T> values(times.size());

    f.evaluate(times, values);

    for (size_t i = 0; i < times.size(); ++i)
    {
      EXPECT_EQ(values[i], f.evaluate(times[i])) << "at t = " << times[i];
    }
  }

  TYPED_TEST(PlfTest, SortedTimesNegativeTimeThrows)
  {
    using PlfT = TypeParam;
    using T = typename PlfT::value_type;

    PlfT f({ { 0, 0 }, { 2, 20 } });

    std::vector<T> times = { -1, 1 };
    std::vector<T> values(times.size());

    EXPECT_THROW(f.evaluate(times, values), std::domain_error);
  }

}
