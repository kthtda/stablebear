#include <gtest/gtest.h>

#include <sbear/tensor.hpp>
#include <sbear/functional/pcf.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/persistence_pair.hpp>
#include <sbear/persistence/stable_rank.hpp>
#include <sbear/task.hpp>
#include <sbear/executor.hpp>

#include <sstream>

namespace
{
  using ScalarTypes = ::testing::Types<sb::float32_t, sb::float64_t>;

  template<typename T>
  class BarcodeAndStableRankTest : public ::testing::Test
  {
  };

  TYPED_TEST_SUITE(BarcodeAndStableRankTest, ScalarTypes);

  // ============================================================================
  // Barcode equality vs. is_isomorphic_to
  // ============================================================================

  TYPED_TEST(BarcodeAndStableRankTest, EqualityVsIsomorphic)
  {
    using T = TypeParam;
    using Pair = sb::ph::PersistencePair<T>;
    using Barcode = sb::ph::Barcode<T>;

    std::vector<Pair> bars1{ Pair(T(0), T(1)), Pair(T(2), T(3)) };
    std::vector<Pair> bars2{ Pair(T(2), T(3)), Pair(T(0), T(1)) }; // permuted

    Barcode b1(bars1);
    Barcode b2(bars2);

    EXPECT_FALSE(b1 == b2);
    EXPECT_TRUE(b1.is_isomorphic_to(b2));

    // Different content should not be isomorphic
    std::vector<Pair> bars3{ Pair(T(0), T(1)), Pair(T(2), T(4)) };
    Barcode b3(bars3);
    EXPECT_FALSE(b1.is_isomorphic_to(b3));
  }

  TYPED_TEST(BarcodeAndStableRankTest, IsomorphicWithinTolerance)
  {
    using T = TypeParam;
    using Pair = sb::ph::PersistencePair<T>;
    using Barcode = sb::ph::Barcode<T>;

    // Endpoints differing by a tiny amount (as can happen when the same
    // barcode is computed from a point cloud versus a distance matrix).
    auto eps = static_cast<T>(1e-6);
    Barcode a(std::vector<Pair>{ Pair(T(0), T(0.678862)), Pair(T(0), T(0.880222)) });
    Barcode b(std::vector<Pair>{ Pair(T(0), T(0.678862) + eps), Pair(T(0), T(0.880222) - eps) });

    // Tolerant by default.
    EXPECT_TRUE(a.is_isomorphic_to(b));

    // Bitwise comparison rejects the same tiny difference.
    EXPECT_FALSE(a.is_isomorphic_to(b, 0.0, 0.0));

    // Infinite endpoints must match exactly, even with generous tolerance.
    Barcode inf(std::vector<Pair>{ Pair(T(0), std::numeric_limits<T>::infinity()) });
    Barcode fin(std::vector<Pair>{ Pair(T(0), T(1e12)) });
    EXPECT_FALSE(inf.is_isomorphic_to(fin, 1.0, 1.0));
  }

  // ============================================================================
  // is_infinite and streaming with infinities
  // ============================================================================

  TYPED_TEST(BarcodeAndStableRankTest, IsInfiniteAndStreamFormatting)
  {
    using T = TypeParam;
    using Pair = sb::ph::PersistencePair<T>;
    using Barcode = sb::ph::Barcode<T>;

    EXPECT_TRUE(Barcode::is_infinite(std::numeric_limits<T>::infinity()));
    EXPECT_TRUE(Barcode::is_infinite(std::numeric_limits<T>::max()));
    EXPECT_FALSE(Barcode::is_infinite(static_cast<T>(1)));

    std::vector<Pair> bars;
    bars.emplace_back(static_cast<T>(0), std::numeric_limits<T>::infinity());
    bars.emplace_back(-std::numeric_limits<T>::infinity(), static_cast<T>(1));

    Barcode bc(std::move(bars));

    std::stringstream ss;
    ss << bc;
    auto s = ss.str();

    EXPECT_NE(s.find("oo"), std::string::npos);
    EXPECT_NE(s.find("-oo"), std::string::npos);
  }

  // ============================================================================
  // barcode_to_stable_rank on simple examples
  // ============================================================================

  TYPED_TEST(BarcodeAndStableRankTest, EmptyBarcodeGivesZeroPcf)
  {
    using T = TypeParam;
    using Barcode = sb::ph::Barcode<T>;
    using PcfT = sb::Pcf<T, T>;
    using Pt = typename PcfT::point_type;

    Barcode empty;
    auto f = sb::ph::barcode_to_stable_rank(empty);

    ASSERT_EQ(f.points().size(), 1u);
    EXPECT_EQ(f.points()[0], Pt(static_cast<T>(0), static_cast<T>(0)));
  }

  TYPED_TEST(BarcodeAndStableRankTest, FiniteBarsStableRank)
  {
    using T = TypeParam;
    using Pair = sb::ph::PersistencePair<T>;
    using Barcode = sb::ph::Barcode<T>;
    using PcfT = sb::Pcf<T, T>;
    using Pt = typename PcfT::point_type;

    // Two finite bars: lifetimes 1 and 2
    std::vector<Pair> bars{ Pair(T(0), T(1)), Pair(T(0), T(2)) };
    Barcode bc(std::move(bars));

    auto f = sb::ph::barcode_to_stable_rank(bc);

    ASSERT_EQ(f.points().size(), 3u);
    EXPECT_EQ(f.points()[0], Pt(static_cast<T>(0), static_cast<T>(2))); // both alive at t=0
    EXPECT_EQ(f.points()[1], Pt(static_cast<T>(1), static_cast<T>(1))); // one bar has died
    EXPECT_EQ(f.points()[2], Pt(static_cast<T>(2), static_cast<T>(0))); // all bars dead
  }

  // ============================================================================
  // BarcodeToStableRankTask over a small tensor
  // ============================================================================

  TYPED_TEST(BarcodeAndStableRankTest, BarcodeToStableRankTaskMatchesDirectConversion)
  {
    using T = TypeParam;
    using Barcode = sb::ph::Barcode<T>;
    using PcfT = sb::Pcf<T, T>;

    sb::Tensor<Barcode> barcodes({ 2 });

    std::vector<sb::ph::PersistencePair<T>> bars0;
    bars0.emplace_back(static_cast<T>(0), static_cast<T>(1));
    std::vector<sb::ph::PersistencePair<T>> bars1;
    bars1.emplace_back(static_cast<T>(0), static_cast<T>(2));

    barcodes(0) = Barcode(bars0);
    barcodes(1) = Barcode(bars1);

    sb::Tensor<PcfT> out;

    auto task = sb::ph::make_stable_rank_task(barcodes, out);
    task->start_async(sb::default_executor()).future().wait();

    ASSERT_EQ(out.shape().size(), 1u);
    ASSERT_EQ(out.shape(0), 2u);

    auto f0 = sb::ph::barcode_to_stable_rank(barcodes(0));
    auto f1 = sb::ph::barcode_to_stable_rank(barcodes(1));

    EXPECT_EQ(out(0), f0);
    EXPECT_EQ(out(1), f1);
  }

} // namespace

