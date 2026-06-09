#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import stablebear as sb


def test_noisy_trig():
    X = sb.random.noisy_sin((10, 20))
    assert X.shape == (10, 20)


def test_global_seed_determinism():
    sb.random.seed(42)
    A = sb.random.noisy_sin((5, 10))

    sb.random.seed(42)
    B = sb.random.noisy_sin((5, 10))

    assert A.array_equal(B)


def test_global_seed_different_seeds_differ():
    sb.random.seed(42)
    A = sb.random.noisy_sin((5, 10))

    sb.random.seed(99)
    B = sb.random.noisy_sin((5, 10))

    assert not A.array_equal(B)


def test_generator_determinism():
    gen = sb.random.Generator(seed=123)
    A = sb.random.noisy_sin((5, 10), generator=gen)

    gen2 = sb.random.Generator(seed=123)
    B = sb.random.noisy_sin((5, 10), generator=gen2)

    assert A.array_equal(B)


def test_generator_noisy_cos():
    gen = sb.random.Generator(seed=77)
    A = sb.random.noisy_cos((3, 4), generator=gen)

    gen2 = sb.random.Generator(seed=77)
    B = sb.random.noisy_cos((3, 4), generator=gen2)

    assert A.array_equal(B)


def test_determinism_pcf64():
    gen = sb.random.Generator(seed=42)
    A = sb.random.noisy_sin((3, 4), dtype=sb.pcf64, generator=gen)

    gen2 = sb.random.Generator(seed=42)
    B = sb.random.noisy_sin((3, 4), dtype=sb.pcf64, generator=gen2)

    assert A.array_equal(B)
