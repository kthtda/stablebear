#include <gtest/gtest.h>

#include <sbear/tensor.hpp>
#include <sbear/nested_tensor.hpp>
#include <sbear/point_cloud.hpp>
#include <sbear/walk.hpp>
#include <sbear/distance_matrix.hpp>
#include <sbear/symmetric_matrix.hpp>
#include <sbear/distance_matrix.hpp>
#include <sbear/symmetric_matrix.hpp>

// ============================================================================
// Helpers
// ============================================================================

namespace
{

  static_assert(sb::IsTensor<sb::NestedTensor<uint64_t>>);

  TEST(NestedTensor, SatisfiesTensorInterfaceAtEveryDepth)
  {
    using Nested = sb::NestedTensor<uint64_t>;

    sb::Tensor<uint64_t> leaf({ 2 });
    leaf(0) = 3;
    leaf(1) = 7;
    Nested leafTensor(std::move(leaf));

    EXPECT_EQ(leafTensor.shape(), (std::vector<size_t>{ 2 }));
    EXPECT_EQ(leafTensor.rank(), 1);
    EXPECT_EQ(leafTensor.size(), 2);
    EXPECT_EQ(std::get<uint64_t>(leafTensor({ 1 })), 7);

    sb::Tensor<Nested> children({ 1 });
    children(0) = leafTensor;
    Nested nestedTensor(std::move(children));

    EXPECT_EQ(nestedTensor.shape(), (std::vector<size_t>{ 1 }));
    EXPECT_EQ(nestedTensor.rank(), 1);
    EXPECT_EQ(nestedTensor.size(), 1);
    const auto child = std::get<Nested>(nestedTensor({ 0 }));
    EXPECT_EQ(child.depth(), 1);
    EXPECT_EQ(std::get<uint64_t>(child({ 0 })), 3);
  }

  TEST(NestedTensor, ConvertsStaticallyNestedTensorTypes)
  {
    sb::Tensor<uint64_t> leaf({ 2 });
    leaf(0) = 3;
    leaf(1) = 7;
    sb::Tensor<sb::Tensor<uint64_t>> levelTwo({ 1 }, leaf);
    sb::Tensor<sb::Tensor<sb::Tensor<uint64_t>>> levelThree(
      { 1 }, levelTwo);

    const auto nested = sb::to_nested_tensor(levelThree);
    const auto empty = sb::to_nested_tensor(
      sb::Tensor<sb::Tensor<sb::Tensor<uint64_t>>>({ 0 }));

    static_assert(std::same_as<
      std::remove_cvref_t<decltype(nested)>,
      sb::NestedTensor<uint64_t>>);
    EXPECT_EQ(nested.depth(), 3);
    EXPECT_EQ(std::get<uint64_t>(
      nested.nested()(0).nested()(0)({ 1 })), 7);
    EXPECT_EQ(empty.depth(), 3);
    EXPECT_EQ(empty.shape(), (std::vector<size_t>{ 0 }));
  }

  TEST(NestedTensor, DistinguishesValueConstructionFromOuterViewWrapping)
  {
    using Nested = sb::NestedTensor<uint64_t>;

    sb::Tensor<uint64_t> firstLeaf({ 1 });
    firstLeaf(0) = 1;
    sb::Tensor<Nested> children({ 1 });
    children(0) = Nested(firstLeaf);

    const Nested value(children, 1);
    const Nested view = Nested::from_outer_view(children.flatten(), 1);

    firstLeaf(0) = 9;
    EXPECT_EQ(std::get<uint64_t>(value.nested()(0)({ 0 })), 1);
    EXPECT_EQ(std::get<uint64_t>(view.nested()(0)({ 0 })), 9);

    sb::Tensor<uint64_t> replacement({ 1 });
    replacement(0) = 7;
    children(0) = Nested(replacement);

    EXPECT_EQ(std::get<uint64_t>(value.nested()(0)({ 0 })), 1);
    EXPECT_EQ(std::get<uint64_t>(view.nested()(0)({ 0 })), 7);
  }

  struct IndexableValue
  {
    using index_type = size_t;

    int value = 0;

    [[nodiscard]] IndexableValue index_into(const size_t& index) const
    {
      return { value + static_cast<int>(index) };
    }
  };

  template<typename T>
  sb::Tensor<T> make_sequential(const std::vector<size_t>& shape)
  {
    sb::Tensor<T> t(shape);
    size_t n = 0;
    sb::walk(t, [&t, &n](const std::vector<size_t>& idx)
    {
      t(idx) = static_cast<T>(n++);
    });
    return t;
  }

  TEST(TensorProperties, NonIndexedPropertiesComposeWithOrdinaryStorage)
  {
    constexpr sb::TensorProperties TestProperty = 1 << 8;
    using PropertyTensor = sb::Tensor<int, TestProperty>;

    PropertyTensor tensor({ 2, 3 }, 4);
    static_assert(std::same_as<decltype(tensor.flatten()), PropertyTensor>);

    EXPECT_EQ(tensor.shape(), (std::vector<size_t>{ 2, 3 }));
    EXPECT_EQ(tensor(std::vector<size_t>{ 1, 2 }), 4);
    tensor(std::vector<size_t>{ 1, 2 }) = 9;
    EXPECT_EQ(tensor.flatten()(5), 9);
  }

  TEST(TensorProperties, MakeIndexedTensorBroadcastsSourceAcrossAdditionalIndexAxes)
  {
    // The source [10, 20] has shape (2), while the indices
    // [[1, 2, 3], [4, 5, 6]] have shape (2, 3). Broadcasting each source
    // element across the additional axis and adding its index produces
    // [[11, 12, 13], [24, 25, 26]]. Point-cloud subsampling uses this extra
    // axis for multiple independently indexed samples of each source cloud.
    sb::Tensor<IndexableValue> source({ 2 });
    source(0) = { 10 };
    source(1) = { 20 };
    sb::Tensor<size_t> indices({ 2, 3 });
    indices({ 0, 0 }) = 1;
    indices({ 0, 1 }) = 2;
    indices({ 0, 2 }) = 3;
    indices({ 1, 0 }) = 4;
    indices({ 1, 1 }) = 5;
    indices({ 1, 2 }) = 6;

    const auto indexed = sb::make_indexed_tensor(source, indices);

    static_assert(decltype(indexed)::IsIndexed);
    EXPECT_EQ(indexed.shape(), (std::vector<size_t>{ 2, 3 }));
    EXPECT_EQ(indexed({ 0, 2 }).value, 13);
    EXPECT_EQ(indexed({ 1, 1 }).value, 25);
  }

  TEST(TensorProperties, MakeIndexedTensorBroadcastsSingletonSourceAxes)
  {
    sb::Tensor<IndexableValue> source({ 2, 1 });
    source({ 0, 0 }) = { 10 };
    source({ 1, 0 }) = { 20 };
    sb::Tensor<size_t> indices({ 2, 3 });
    indices({ 0, 0 }) = 1;
    indices({ 0, 1 }) = 2;
    indices({ 0, 2 }) = 3;
    indices({ 1, 0 }) = 4;
    indices({ 1, 1 }) = 5;
    indices({ 1, 2 }) = 6;

    const auto indexed = sb::make_indexed_tensor(source, indices);

    EXPECT_EQ(indexed({ 0, 2 }).value, 13);
    EXPECT_EQ(indexed({ 1, 1 }).value, 25);
  }

  TEST(TensorProperties, IndexedViewsReplaceSharedSourceWhenMaterialized)
  {
    sb::Tensor<IndexableValue> source({ 2 });
    source(0) = { 10 };
    source(1) = { 20 };
    sb::Tensor<size_t> indices({ 2 });
    indices(0) = 1;
    indices(1) = 2;

    auto indexed = sb::make_indexed_tensor(source, indices);
    auto view = indexed.flatten();

    view.ensure_materialized();

    EXPECT_THROW((void)indexed.indices_view(), std::logic_error);
    EXPECT_EQ(indexed.source_view()(0).value, 11);
    EXPECT_EQ(indexed.source_view()(1).value, 22);

    view.writable_at(0).value = 99;
    EXPECT_EQ(indexed(0).value, 99);
    EXPECT_EQ(source(0).value, 10);

    // A second transition is a no-op and must not restore the lazy source.
    indexed.ensure_materialized();
    EXPECT_EQ(view(0).value, 99);
  }

  TEST(TensorProperties, TensorValuedIndicesUseNestedTensorStorage)
  {
    using PointCloud = sb::PointCloud<float>;

    sb::Tensor<float> coordinates({ 3, 1 });
    coordinates({ 0, 0 }) = 10;
    coordinates({ 1, 0 }) = 20;
    coordinates({ 2, 0 }) = 30;

    sb::Tensor<PointCloud> source({ 1 });
    source(0) = PointCloud(std::move(coordinates));

    sb::Tensor<uint64_t> rows({ 2 });
    rows(0) = 2;
    rows(1) = 0;
    const sb::Tensor<sb::Tensor<uint64_t>> selectionsByCloud({ 1 }, rows);
    const auto selections = sb::to_nested_tensor(selectionsByCloud);

    const auto indexed = sb::make_indexed_tensor(source, selections);

    static_assert(std::same_as<
      typename decltype(indexed)::index_tensor_type,
      sb::NestedTensor<uint64_t>>);
    EXPECT_EQ(indexed.indices_view().depth(), 2);
    const PointCloud selected = indexed(0);
    EXPECT_EQ(selected(0, 0), 30);
    EXPECT_EQ(selected(1, 0), 10);

    PointCloud mutableSelected = indexed(0);
    EXPECT_THROW(mutableSelected(0, 0) = 99, std::logic_error);
  }

  TEST(TensorProperties, MakeIndexedTensorRejectsNonBroadcastableSourceShape)
  {
    const sb::Tensor<IndexableValue> source({ 2, 2 });
    const sb::Tensor<size_t> indices({ 2, 3 });

    EXPECT_THROW(sb::make_indexed_tensor(source, indices), std::invalid_argument);
  }



// ============================================================================
// Typed test suite — runs every test body for int, float, and double
// ============================================================================

  using TensorTypes = ::testing::Types<int, float, double>;

  template <typename T>
  class TensorTppTyped : public ::testing::Test {};
  TYPED_TEST_SUITE(TensorTppTyped, TensorTypes);

// operator=(const T& val)

  TYPED_TEST(TensorTppTyped, AssignScalarFillsAllElements)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 2, 3 });
    t = T(7);
    sb::walk(t, [&t](const std::vector<size_t>& idx) { EXPECT_EQ(t(idx), T(7)); });
  }

  TYPED_TEST(TensorTppTyped, AssignScalarOverwritesPreviousValues)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 4 });
    for (size_t i = 0; i < 4; ++i) t(i) = T(100 + i);
    t = T(0);
    for (size_t i = 0; i < 4; ++i) EXPECT_EQ(t(i), T(0));
  }

// operator== / operator!=

  TYPED_TEST(TensorTppTyped, EqualityShapeMismatch)
  {
    using T = TypeParam;
    sb::Tensor<T> a({ 2, 3 });
    sb::Tensor<T> b({ 3, 2 });
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
  }

  TYPED_TEST(TensorTppTyped, EqualityIdentical)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({ 3, 4 });
    auto b = make_sequential<T>({ 3, 4 });
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
  }

  TYPED_TEST(TensorTppTyped, InequalityOneElementDiffers)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({ 2, 2 });
    auto b = make_sequential<T>({ 2, 2 });
    b({ 1, 1 }) = T(999);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
  }

// assign_from

  TYPED_TEST(TensorTppTyped, AssignFromCopiesValues)
  {
    using T = TypeParam;
    auto src = make_sequential<T>({ 2, 3 });
    sb::Tensor<T> dst({ 2, 3 });
    dst.assign_from(src);
    EXPECT_EQ(dst, src);
  }

  TYPED_TEST(TensorTppTyped, AssignFromShapeMismatchThrows)
  {
    using T = TypeParam;
    sb::Tensor<T> src({ 2, 3 });
    sb::Tensor<T> dst({ 3, 2 });
    EXPECT_THROW(dst.assign_from(src), std::invalid_argument);
  }

// size()

  TYPED_TEST(TensorTppTyped, SizeOfEmptyTensor)
  {
    using T = TypeParam;
    sb::Tensor<T> t;
    EXPECT_EQ(t.size(), 0u);
  }

  TYPED_TEST(TensorTppTyped, Size1d)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 7 });
    EXPECT_EQ(t.size(), 7u);
  }

  TYPED_TEST(TensorTppTyped, Size3d)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 2, 3, 5 });
    EXPECT_EQ(t.size(), 30u);
  }

// copy()

  TYPED_TEST(TensorTppTyped, CopyIsDeepCopy)
  {
    using T = TypeParam;
    auto original = make_sequential<T>({ 3, 3 });
    auto copy = original.copy();
    EXPECT_EQ(original, copy);
    copy({ 0, 0 }) = T(9999);
    EXPECT_NE(original({ 0, 0 }), copy({ 0, 0 }));
  }

  TYPED_TEST(TensorTppTyped, CopyOfNonContiguousTensorIsContiguous)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 4 });
    auto sliced = t[std::vector<sb::Slice>{ sb::range(1, 3, std::nullopt), sb::all() }];
    EXPECT_FALSE(sliced.is_contiguous());
    auto c = sliced.copy();
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_EQ(c.shape(0), 2u);
    EXPECT_EQ(c.shape(1), 4u);
    for (size_t i = 0; i < 2; ++i)
      for (size_t j = 0; j < 4; ++j)
        EXPECT_EQ(c({ i, j }), sliced({ i, j }));
  }

// flatten()

  TYPED_TEST(TensorTppTyped, FlattenContiguous)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 2, 3 });
    auto flat = t.flatten();
    ASSERT_EQ(flat.shape().size(), 1u);
    EXPECT_EQ(flat.shape(0), 6u);
    for (size_t i = 0; i < 6; ++i)
      EXPECT_EQ(flat(i), T(i));
  }

  TYPED_TEST(TensorTppTyped, FlattenNonContiguous)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 4 });
    auto sliced = t[std::vector<sb::Slice>{ sb::range(0, 4, 2), sb::all() }];
    EXPECT_FALSE(sliced.is_contiguous());
    auto flat = sliced.flatten();
    EXPECT_TRUE(flat.is_contiguous());
    EXPECT_NE(sliced.data(), flat.data());
  }

// walk()

  TYPED_TEST(TensorTppTyped, WalkBoolFunctorStopsEarly)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 10 });
    int count = 0;
    sb::walk(t, [&count](const std::vector<size_t>&) -> bool { return ++count < 5; });
    EXPECT_EQ(count, 5);
  }

  TYPED_TEST(TensorTppTyped, WalkBoolFunctorAllTrue)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4 });
    int count = 0;
    sb::walk(t, [&count](const std::vector<size_t>&) -> bool { ++count; return true; });
    EXPECT_EQ(count, 4);
  }

  TYPED_TEST(TensorTppTyped, WalkRank0ShapeVisitsSingleElement)
  {
    using T = TypeParam;
    // A default-constructed (rank-0, shape ()) tensor holds exactly one
    // element, so walk must visit it once with an empty index. Only a
    // zero-size extent is skipped (see WalkZeroDimensionDoesNothing).
    sb::Tensor<T> t;
    int count = 0;
    sb::walk(t, [&count](const std::vector<size_t>& idx) { ++count; EXPECT_TRUE(idx.empty()); });
    EXPECT_EQ(count, 1);
  }

  TYPED_TEST(TensorTppTyped, WalkZeroDimensionDoesNothing)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 0, 3 });
    int count = 0;
    sb::walk(t, [&count](const std::vector<size_t>&) { ++count; });
    EXPECT_EQ(count, 0);
  }

// apply()

  TYPED_TEST(TensorTppTyped, ApplyMultipliesAllElements)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 3 });
    t.apply([](T& v) { v = v * T(2); });
    size_t n = 0;
    sb::walk(t, [&t, &n](const std::vector<size_t>& idx) {
      EXPECT_EQ(t(idx), T(n) * T(2));
      ++n;
    });
  }

  TYPED_TEST(TensorTppTyped, ApplyOnEmptyDimensionDoesNothing)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 0 });
    int calls = 0;
    t.apply([&calls](T&) { ++calls; });
    EXPECT_EQ(calls, 0);
  }

// extract() / operator[]

  TYPED_TEST(TensorTppTyped, SliceAllPreservesShape)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 4 });
    auto view = t[std::vector<sb::Slice>{ sb::all(), sb::all() }];
    EXPECT_EQ(view.shape(0), 3u);
    EXPECT_EQ(view.shape(1), 4u);
  }

  TYPED_TEST(TensorTppTyped, SliceIndexDropsDimension)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 4 });
    auto view = t[std::vector<sb::Slice>{ sb::index(1), sb::all() }];
    EXPECT_EQ(view.shape().size(), 1u);
    EXPECT_EQ(view.shape(0), 4u);
    for (size_t j = 0; j < 4; ++j)
      EXPECT_EQ(view({ j }), T(4 + j));
  }

  TYPED_TEST(TensorTppTyped, SliceRangeStopClampedToShape)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    auto view = t[std::vector<sb::Slice>{ sb::range(1, 100, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 4u);
  }

  TYPED_TEST(TensorTppTyped, SliceRangeNegativeStartResolvesAgainstSize)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });  // [0, 1, 2, 3, 4]

    // An in-range negative start is resolved against the dimension size
    // (NumPy slice.indices): -2 -> 3, so [3, 4].
    auto view = t[std::vector<sb::Slice>{ sb::range(-2, std::nullopt, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 2u);
    EXPECT_EQ(view({ 0 }), T(3));
    EXPECT_EQ(view({ 1 }), T(4));

    // A start more negative than the size clamps to 0 (not an error).
    auto clamped = t[std::vector<sb::Slice>{ sb::range(-10, 3, std::nullopt) }];
    EXPECT_EQ(clamped.shape(0), 3u);
    EXPECT_EQ(clamped({ 0 }), T(0));
    EXPECT_EQ(clamped({ 1 }), T(1));
    EXPECT_EQ(clamped({ 2 }), T(2));
  }

  TYPED_TEST(TensorTppTyped, SliceIndexNegativeResolvesAgainstSize)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 5 });
    // Row -1 is the last row.
    auto view = t[std::vector<sb::Slice>{ sb::index(-1), sb::all() }];
    EXPECT_EQ(view.shape().size(), 1u);
    EXPECT_EQ(view.shape(0), 5u);
    for (size_t j = 0; j < 5; ++j)
      EXPECT_EQ(view({ j }), T(3 * 5 + j));
  }

  TYPED_TEST(TensorTppTyped, SliceIndexOutOfBoundsThrows)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 5 });
    std::vector<sb::Slice> too_large{ sb::index(4), sb::all() };
    std::vector<sb::Slice> too_negative{ sb::index(-5), sb::all() };
    EXPECT_THROW((void)t[too_large], std::out_of_range);
    EXPECT_THROW((void)t[too_negative], std::out_of_range);
  }

  TYPED_TEST(TensorTppTyped, SliceRangeStopLessThanStartGivesZeroSize)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    auto view = t[std::vector<sb::Slice>{ sb::range(3, 1, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 0u);
  }

  TYPED_TEST(TensorTppTyped, SliceRangeZeroStepThrows)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    // A zero step is invalid (NumPy: "slice step cannot be zero").
    EXPECT_THROW((void)t[std::vector<sb::Slice>{ sb::range(0, 5, 0) }],
                 std::invalid_argument);
  }

  TYPED_TEST(TensorTppTyped, SliceRangeNegativeStep)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });  // [0, 1, 2, 3, 4]
    auto view = t[std::vector<sb::Slice>{ sb::range(4, 0, -1) }];
    EXPECT_EQ(view.shape(0), 4u);
    EXPECT_EQ(view({0}), T(4));
    EXPECT_EQ(view({1}), T(3));
    EXPECT_EQ(view({2}), T(2));
    EXPECT_EQ(view({3}), T(1));
  }

  TYPED_TEST(TensorTppTyped, SliceRangeWithDefaultStartStop)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 5 });
    auto view = t[std::vector<sb::Slice>{ sb::range(std::nullopt, std::nullopt, std::nullopt) }];
    EXPECT_EQ(view.shape(0), 5u);
    for (size_t i = 0; i < 5; ++i)
      EXPECT_EQ(view({ i }), T(i));
  }

  TYPED_TEST(TensorTppTyped, Slice3dMixed)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 5, 6 });
    auto view = t[std::vector<sb::Slice>{
        sb::index(1),
        sb::range(2, 4, std::nullopt),
        sb::range(std::nullopt, std::nullopt, 2)
    }];
    EXPECT_EQ(view.shape().size(), 2u);
    EXPECT_EQ(view.shape(0), 2u);
    EXPECT_EQ(view.shape(1), 3u);
    EXPECT_EQ(view({ 0, 0 }), T(1 * 30 + 2 * 6 + 0));
    EXPECT_EQ(view({ 0, 1 }), T(1 * 30 + 2 * 6 + 2));
    EXPECT_EQ(view({ 0, 2 }), T(1 * 30 + 2 * 6 + 4));
    EXPECT_EQ(view({ 1, 0 }), T(1 * 30 + 3 * 6 + 0));
  }

// flatten() non-1d index throws

  TYPED_TEST(TensorTppTyped, FlattenedViewNon1dIndexThrows)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 2, 3 });
    auto flat = t.flatten();
    EXPECT_THROW((void)flat({ 0, 0 }), std::runtime_error);
  }

// 1d operator()(size_t) overload

  TYPED_TEST(TensorTppTyped, SingleIndexOverload1d)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 5 });
    for (size_t i = 0; i < 5; ++i) t(i) = T(i * 10);
    for (size_t i = 0; i < 5; ++i) EXPECT_EQ(t(i), T(i * 10));
  }

// any_of / any_of_idx

  TYPED_TEST(TensorTppTyped, AnyOfReturnsTrueWhenPredicateMatches)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4 });
    EXPECT_TRUE(t.any_of([](const T& v) { return v == T(3); }));
  }

  TYPED_TEST(TensorTppTyped, AnyOfReturnsFalseWhenNoMatch)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4 });
    EXPECT_FALSE(t.any_of([](const T& v) { return v > T(100); }));
  }

  TYPED_TEST(TensorTppTyped, AnyOfIdxReturnsTrueWhenIndexMatches)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 3 });
    EXPECT_TRUE(t.any_of_idx([&t](const std::vector<size_t>& idx) { return t(idx) == T(8); }));
  }

  TYPED_TEST(TensorTppTyped, AnyOfIdxReturnsFalseWhenNoMatch)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 3, 3 });
    EXPECT_FALSE(t.any_of_idx([&t](const std::vector<size_t>& idx) { return t(idx) > T(100); }));
  }

// rank() / strides() / stride() / offset()

  TYPED_TEST(TensorTppTyped, RankMatchesDimensionCount)
  {
    using T = TypeParam;
    EXPECT_EQ(sb::Tensor<T>({ 5 }).rank(), 1u);
    EXPECT_EQ(sb::Tensor<T>({ 3, 4 }).rank(), 2u);
    EXPECT_EQ(sb::Tensor<T>({ 2, 3, 5 }).rank(), 3u);
  }

  TYPED_TEST(TensorTppTyped, StridesCorrectForRowMajor2d)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3, 4 });
    ASSERT_EQ(t.strides().size(), 2u);
    EXPECT_EQ(t.stride(0), 4u);
    EXPECT_EQ(t.stride(1), 1u);
  }

  TYPED_TEST(TensorTppTyped, OffsetNonZeroAfterIndexSlice)
  {
    using T = TypeParam;
    auto t = make_sequential<T>({ 4, 4 });
    auto view = t[std::vector<sb::Slice>{ sb::index(1), sb::all() }];
    EXPECT_EQ(view.offset(), 4u);
  }

// ============================================================================
// Arithmetic operators
// ============================================================================

  TYPED_TEST(TensorTppTyped, DivideAssign)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(6); t(1) = T(4); t(2) = T(2);
    t /= T(2);
    EXPECT_EQ(t(0), T(3));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(1));
  }

  TYPED_TEST(TensorTppTyped, Divide)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(6); t(1) = T(4); t(2) = T(2);
    auto result = t / T(2);
    EXPECT_EQ(result(0), T(3));
    EXPECT_EQ(result(1), T(2));
    EXPECT_EQ(result(2), T(1));
    EXPECT_EQ(t(0), T(6));
    EXPECT_EQ(t(1), T(4));
    EXPECT_EQ(t(2), T(2));
  }

  TYPED_TEST(TensorTppTyped, MultiplyAssign)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    t *= T(4);
    EXPECT_EQ(t(0), T(4));
    EXPECT_EQ(t(1), T(8));
    EXPECT_EQ(t(2), T(12));
  }

  TYPED_TEST(TensorTppTyped, Multiply)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    auto result = t * T(4);
    EXPECT_EQ(result(0), T(4));
    EXPECT_EQ(result(1), T(8));
    EXPECT_EQ(result(2), T(12));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(3));
  }

  TYPED_TEST(TensorTppTyped, AddAssign)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    t += T(10);
    EXPECT_EQ(t(0), T(11));
    EXPECT_EQ(t(1), T(12));
    EXPECT_EQ(t(2), T(13));
  }

  TYPED_TEST(TensorTppTyped, Add)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    auto result = t + T(10);
    EXPECT_EQ(result(0), T(11));
    EXPECT_EQ(result(1), T(12));
    EXPECT_EQ(result(2), T(13));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(3));
  }

  TYPED_TEST(TensorTppTyped, SubtractAssign)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(5); t(1) = T(7); t(2) = T(9);
    t -= T(3);
    EXPECT_EQ(t(0), T(2));
    EXPECT_EQ(t(1), T(4));
    EXPECT_EQ(t(2), T(6));
  }

  TYPED_TEST(TensorTppTyped, Subtract)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(5); t(1) = T(7); t(2) = T(9);
    auto result = t - T(3);
    EXPECT_EQ(result(0), T(2));
    EXPECT_EQ(result(1), T(4));
    EXPECT_EQ(result(2), T(6));
    EXPECT_EQ(t(0), T(5));
    EXPECT_EQ(t(1), T(7));
    EXPECT_EQ(t(2), T(9));
  }

  TYPED_TEST(TensorTppTyped, FreeMultiply)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    auto result = T(4) * t;
    EXPECT_EQ(result(0), T(4));
    EXPECT_EQ(result(1), T(8));
    EXPECT_EQ(result(2), T(12));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(3));
  }

  TYPED_TEST(TensorTppTyped, FreeAdd)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    auto result = T(10) + t;
    EXPECT_EQ(result(0), T(11));
    EXPECT_EQ(result(1), T(12));
    EXPECT_EQ(result(2), T(13));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(3));
  }

  TYPED_TEST(TensorTppTyped, FreeSubtract)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(3);
    auto result = T(10) - t;
    EXPECT_EQ(result(0), T(9));
    EXPECT_EQ(result(1), T(8));
    EXPECT_EQ(result(2), T(7));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(3));
  }

  TYPED_TEST(TensorTppTyped, FreeDivide)
  {
    using T = TypeParam;
    sb::Tensor<T> t({ 3 });
    t(0) = T(1); t(1) = T(2); t(2) = T(4);
    auto result = T(8) / t;
    EXPECT_EQ(result(0), T(8));
    EXPECT_EQ(result(1), T(4));
    EXPECT_EQ(result(2), T(2));
    EXPECT_EQ(t(0), T(1));
    EXPECT_EQ(t(1), T(2));
    EXPECT_EQ(t(2), T(4));
  }

// ============================================================================
// Cross-type tests (not parameterizable on a single T)
// ============================================================================

  TEST(TensorTpp, AssignFromCrossTypeIntToDouble)
  {
    sb::Tensor<int> src({ 2, 2 });
    src({ 0, 0 }) = 1; src({ 0, 1 }) = 2;
    src({ 1, 0 }) = 3; src({ 1, 1 }) = 4;

    sb::Tensor<double> dst({ 2, 2 });
    dst.assign_from(src);

    EXPECT_DOUBLE_EQ(dst({ 0, 0 }), 1.0);
    EXPECT_DOUBLE_EQ(dst({ 0, 1 }), 2.0);
    EXPECT_DOUBLE_EQ(dst({ 1, 0 }), 3.0);
    EXPECT_DOUBLE_EQ(dst({ 1, 1 }), 4.0);
  }

  TEST(TensorTpp, AssignFromCrossTypeShapeMismatchThrows)
  {
    sb::Tensor<float> src({ 2, 3 });
    sb::Tensor<double> dst({ 3, 2 });
    EXPECT_THROW(dst.assign_from(src), std::invalid_argument);
  }

  TEST(TensorTpp, CrossTypeEqualityIntAndDouble)
  {
    sb::Tensor<int>    a({ 3 });
    sb::Tensor<double> b({ 3 });
    for (size_t i = 0; i < 3; ++i)
    {
      a(i) = static_cast<int>(i);
      b(i) = static_cast<double>(i);
    }
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
  }

  TEST(TensorTpp, CrossTypeInequalityIntAndDouble)
  {
    sb::Tensor<int>    a({ 3 });
    sb::Tensor<double> b({ 3 });
    for (size_t i = 0; i < 3; ++i)
    {
      a(i) = static_cast<int>(i);
      b(i) = static_cast<double>(i) + 0.5;
    }
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
  }

// ============================================================================
// allclose — Tensor
// ============================================================================

  using FloatTypes = ::testing::Types<float, double>;

  template <typename T>
  class AllcloseTensorTyped : public ::testing::Test {};
  TYPED_TEST_SUITE(AllcloseTensorTyped, FloatTypes);

  TYPED_TEST(AllcloseTensorTyped, IdenticalAreClose)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({3, 4});
    EXPECT_TRUE(sb::allclose(a, a));
  }

  TYPED_TEST(AllcloseTensorTyped, ShapeMismatch)
  {
    using T = TypeParam;
    sb::Tensor<T> a({2, 3}, T(1));
    sb::Tensor<T> b({3, 2}, T(1));
    EXPECT_FALSE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseTensorTyped, SmallPerturbationWithinTolerance)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({3, 4});
    auto b = make_sequential<T>({3, 4});
    b({1, 2}) = b({1, 2}) + T(1e-9);
    EXPECT_TRUE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseTensorTyped, LargePerturbationOutsideTolerance)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({3, 4});
    auto b = make_sequential<T>({3, 4});
    b({2, 0}) = T(999);
    EXPECT_FALSE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseTensorTyped, CustomAtol)
  {
    using T = TypeParam;
    sb::Tensor<T> a({2, 3}, T(0));
    sb::Tensor<T> b({2, 3}, T(0));
    b({1, 1}) = T(0.5);
    EXPECT_FALSE(sb::allclose(a, b));
    EXPECT_TRUE(sb::allclose(a, b, T(1)));
  }

  TYPED_TEST(AllcloseTensorTyped, EmptyTensorsAreClose)
  {
    using T = TypeParam;
    sb::Tensor<T> a({0});
    sb::Tensor<T> b({0});
    EXPECT_TRUE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseTensorTyped, MemberDelegatesToFree)
  {
    using T = TypeParam;
    auto a = make_sequential<T>({2, 3});
    auto b = make_sequential<T>({2, 3});
    b({0, 1}) = b({0, 1}) + T(1e-9);
    EXPECT_EQ(a.allclose(b), sb::allclose(a, b));
  }

// ============================================================================
// allclose — compressed matrices (DistanceMatrix, SymmetricMatrix)
// ============================================================================

  template <typename T>
  struct CompressedMatrixTraits;

  template <typename T>
  struct CompressedMatrixTraits<sb::DistanceMatrix<T>>
  {
    using Scalar = T;
    static sb::DistanceMatrix<T> make(size_t n, T init) { return sb::DistanceMatrix<T>(n, init); }
    static void perturb(sb::DistanceMatrix<T>& m, T val) { m(0, 1) = val; }
  };

  template <typename T>
  struct CompressedMatrixTraits<sb::SymmetricMatrix<T>>
  {
    using Scalar = T;
    static sb::SymmetricMatrix<T> make(size_t n, T init) { return sb::SymmetricMatrix<T>(n, init); }
    static void perturb(sb::SymmetricMatrix<T>& m, T val) { m(0, 1) = val; }
  };

  using CompressedMatrixTypes = ::testing::Types<
    sb::DistanceMatrix<float>, sb::DistanceMatrix<double>,
    sb::SymmetricMatrix<float>, sb::SymmetricMatrix<double>>;

  template <typename T>
  class AllcloseCompressedTyped : public ::testing::Test {};
  TYPED_TEST_SUITE(AllcloseCompressedTyped, CompressedMatrixTypes);

  TYPED_TEST(AllcloseCompressedTyped, IdenticalAreClose)
  {
    using Traits = CompressedMatrixTraits<TypeParam>;
    using T = typename Traits::Scalar;
    auto a = Traits::make(3, T(1));
    auto b = Traits::make(3, T(1));
    EXPECT_TRUE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseCompressedTyped, SizeMismatch)
  {
    using Traits = CompressedMatrixTraits<TypeParam>;
    using T = typename Traits::Scalar;
    auto a = Traits::make(3, T(1));
    auto b = Traits::make(4, T(1));
    EXPECT_FALSE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseCompressedTyped, SmallPerturbationWithinTolerance)
  {
    using Traits = CompressedMatrixTraits<TypeParam>;
    using T = typename Traits::Scalar;
    auto a = Traits::make(3, T(1));
    auto b = Traits::make(3, T(1));
    Traits::perturb(b, T(1) + T(1e-9));
    EXPECT_TRUE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseCompressedTyped, LargePerturbationOutsideTolerance)
  {
    using Traits = CompressedMatrixTraits<TypeParam>;
    using T = typename Traits::Scalar;
    auto a = Traits::make(3, T(1));
    auto b = Traits::make(3, T(1));
    Traits::perturb(b, T(5));
    EXPECT_FALSE(sb::allclose(a, b));
  }

  TYPED_TEST(AllcloseCompressedTyped, CustomAtol)
  {
    using Traits = CompressedMatrixTraits<TypeParam>;
    using T = typename Traits::Scalar;
    auto a = Traits::make(3, T(0));
    auto b = Traits::make(3, T(0));
    Traits::perturb(b, T(0.5));
    EXPECT_FALSE(sb::allclose(a, b));
    EXPECT_TRUE(sb::allclose(a, b, T(1)));
  }

} // namespace
