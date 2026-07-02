// Regression tests for issue #196 (low-severity C++ core defects).

#include <gtest/gtest.h>

#include <sbear/functional/pcf.hpp>
#include <sbear/algorithms/functional/matrix_reduce.hpp>
#include <sbear/persistence/barcode.hpp>
#include <sbear/persistence/betti_curve.hpp>
#include <sbear/persistence/accumulated_persistence.hpp>
#include <sbear/tensor.hpp>

#include <stdexcept>
#include <vector>

namespace
{

  // --------------------------------------------------------------------------
  // Empty PCFs are invalid: evaluate/iterate_rectangles dereference the first
  // point unconditionally, so the constructor must reject an empty vector.
  // --------------------------------------------------------------------------

  TEST(BugscanCore, EmptyPcfConstructionThrows)
  {
    std::vector<sb::Pcf_f64::point_type> pts;
    EXPECT_THROW(sb::Pcf_f64(std::move(pts)), std::invalid_argument);
  }

  // --------------------------------------------------------------------------
  // mean() over a zero-length dimension used to divide by zero and silently
  // return all-NaN PCFs; it must throw like max_element does.
  // --------------------------------------------------------------------------

  TEST(BugscanCore, MeanOverEmptyDimensionThrows)
  {
    sb::Tensor<sb::Pcf_f64> t({ 0, 3 });
    EXPECT_THROW(sb::mean(t, 0), std::invalid_argument);

    sb::Tensor<sb::Pcf_f64> t2({ 3, 0 });
    EXPECT_THROW(sb::mean(t2, 1), std::invalid_argument);
  }

  TEST(BugscanCore, MeanOverNonEmptyDimensionStillWorks)
  {
    sb::Tensor<sb::Pcf_f64> t({ 2 });
    t({ 0ul }) = sb::Pcf_f64(2.0);
    t({ 1ul }) = sb::Pcf_f64(4.0);

    auto m = sb::mean(t, 0);
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m({ 0ul }).points()[0].v, 3.0);
  }

  // --------------------------------------------------------------------------
  // Barcode summaries with negative event times (sublevel-set filtrations)
  // used to emit a (0, 0) breakpoint followed by earlier negative times -- an
  // unsorted PCF. The breakpoints must come out strictly increasing.
  // --------------------------------------------------------------------------

  template <typename PcfT>
  void expect_strictly_increasing(const PcfT& f)
  {
    auto const& pts = f.points();
    ASSERT_FALSE(pts.empty());
    for (size_t i = 1; i < pts.size(); ++i)
    {
      EXPECT_LT(pts[i - 1].t, pts[i].t) << "breakpoints out of order at index " << i;
    }
  }

  TEST(BugscanCore, BettiCurveWithNegativeBirthsIsSorted)
  {
    using T = sb::float64_t;
    sb::ph::Barcode<T> barcode(std::vector<sb::ph::PersistencePair<T>>{
        sb::ph::PersistencePair<T>(-2.0, -0.5),
        sb::ph::PersistencePair<T>(-1.0, 1.0),
        sb::ph::PersistencePair<T>(0.5, 2.0) });

    auto f = sb::ph::barcode_to_betti_curve(barcode);
    expect_strictly_increasing(f);

    // Curve starts at the earliest birth, and the value on [0, inf) -- the
    // PCF evaluation domain -- is correct instead of corrupt: at t = 0 the
    // alive bars are (-1, 1) and (0.5 is not yet born), so count is 1.
    EXPECT_EQ(f.points().front().t, -2.0);
    EXPECT_EQ(f.evaluate(0.0), 1.0);  // bar (-1, 1) alive
    EXPECT_EQ(f.evaluate(0.75), 2.0); // bars (-1, 1) and (0.5, 2) alive
  }

  TEST(BugscanCore, BettiCurveWithNonNegativeBirthsStillStartsAtZero)
  {
    using T = sb::float64_t;
    sb::ph::Barcode<T> barcode(std::vector<sb::ph::PersistencePair<T>>{
        sb::ph::PersistencePair<T>(1.0, 2.0) });

    auto f = sb::ph::barcode_to_betti_curve(barcode);
    expect_strictly_increasing(f);
    EXPECT_EQ(f.points().front().t, 0.0);
    EXPECT_EQ(f.evaluate(0.5), 0.0);
    EXPECT_EQ(f.evaluate(1.5), 1.0);
  }

  TEST(BugscanCore, AccumulatedPersistenceWithNegativeMidpointsIsSorted)
  {
    using T = sb::float64_t;
    sb::ph::Barcode<T> barcode(std::vector<sb::ph::PersistencePair<T>>{
        sb::ph::PersistencePair<T>(-3.0, -1.0),
        sb::ph::PersistencePair<T>(-1.0, 2.0) });

    auto f = sb::ph::barcode_to_accumulated_persistence(barcode);
    expect_strictly_increasing(f);
    EXPECT_EQ(f.points().front().t, -2.0); // earliest midpoint
  }

} // namespace
