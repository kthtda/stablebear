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
#include <sbear/tensor.hpp>
#include <sbear/walk.hpp>
#include <sbear/random_generator.hpp>
#include <sbear/executor.hpp>

#include <random>

TEST(WalkRandom, SameSeedProducesSameValues)
{
  sb::Tensor<double> a({3, 4});
  sb::Tensor<double> b({3, 4});

  sb::DefaultRandomGenerator gen1(42);
  sb::DefaultRandomGenerator gen2(42);

  sb::walk(a, gen1, [&a](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    a(idx) = dist(engine);
  });

  sb::walk(b, gen2, [&b](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    b(idx) = dist(engine);
  });

  EXPECT_TRUE(sb::allclose(a, b));
}

TEST(WalkRandom, DifferentSeedsProduceDifferentValues)
{
  sb::Tensor<double> a({3, 4});
  sb::Tensor<double> b({3, 4});

  sb::DefaultRandomGenerator gen1(42);
  sb::DefaultRandomGenerator gen2(99);

  sb::walk(a, gen1, [&a](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    a(idx) = dist(engine);
  });

  sb::walk(b, gen2, [&b](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    b(idx) = dist(engine);
  });

  EXPECT_FALSE(sb::allclose(a, b));
}

TEST(WalkRandom, ParallelWalkMatchesSequentialWalk)
{
  sb::Tensor<double> seq({80, 10});
  sb::Tensor<double> par({80, 10});

  sb::DefaultRandomGenerator gen1(123);
  sb::DefaultRandomGenerator gen2(123);

  sb::walk(seq, gen1, [&seq](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    seq(idx) = dist(engine);
  });

  sb::parallel_walk(par, gen2, [&par](const std::vector<size_t>& idx, auto& engine) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    par(idx) = dist(engine);
  }, sb::default_executor());

  EXPECT_TRUE(sb::allclose(seq, par));
}

TEST(WalkRandom, ParallelWalkIsDeterministicAcrossRuns)
{
  sb::Tensor<double> a({80, 10});
  sb::Tensor<double> b({80, 10});

  sb::DefaultRandomGenerator gen1(77);
  sb::DefaultRandomGenerator gen2(77);

  sb::parallel_walk(a, gen1, [&a](const std::vector<size_t>& idx, auto& engine) {
    std::normal_distribution<double> dist(0.0, 1.0);
    a(idx) = dist(engine);
  }, sb::default_executor());

  sb::parallel_walk(b, gen2, [&b](const std::vector<size_t>& idx, auto& engine) {
    std::normal_distribution<double> dist(0.0, 1.0);
    b(idx) = dist(engine);
  }, sb::default_executor());

  EXPECT_TRUE(sb::allclose(a, b));
}

namespace
{
  sb::Tensor<double> fill_uniform(sb::DefaultRandomGenerator& gen)
  {
    sb::Tensor<double> t({3, 4});
    sb::walk(t, gen, [&t](const std::vector<size_t>& idx, auto& engine) {
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      t(idx) = dist(engine);
    });
    return t;
  }
}

TEST(RandomGenerator, ConsecutiveDrawsDiffer)
{
  // Each draw reserves a fresh block of seed slots, so the generator advances
  // itself: two consecutive draws from one generator must differ.
  sb::DefaultRandomGenerator gen(5);
  auto a = fill_uniform(gen);
  auto b = fill_uniform(gen);
  EXPECT_FALSE(sb::allclose(a, b));
}

TEST(RandomGenerator, ConsecutiveDrawsAreReproducible)
{
  sb::DefaultRandomGenerator gen1(5);
  sb::DefaultRandomGenerator gen2(5);

  auto a1 = fill_uniform(gen1);
  auto b1 = fill_uniform(gen1);

  auto a2 = fill_uniform(gen2);
  auto b2 = fill_uniform(gen2);

  EXPECT_TRUE(sb::allclose(a1, a2));
  EXPECT_TRUE(sb::allclose(b1, b2));
}

TEST(RandomGenerator, FirstBlockUsesRawSeed)
{
  // The first reservation after (re-)seeding starts at offset 0, so its engines
  // are seeded by the raw seed exactly as the pre-advance design did.
  sb::DefaultRandomGenerator gen(5);
  auto block = gen.reserve(8);
  auto expected = sb::detail::make_engine<sb::Xoroshiro128pp>(5 + 7);
  EXPECT_EQ(block.sub_generator(7)(), expected());
}

TEST(RandomGenerator, SeedResetsStream)
{
  sb::DefaultRandomGenerator gen(5);
  auto a = fill_uniform(gen);
  fill_uniform(gen);  // advance past the first block
  gen.seed(5);
  auto a_again = fill_uniform(gen);
  EXPECT_TRUE(sb::allclose(a, a_again));
}
