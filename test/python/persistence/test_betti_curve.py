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

import numpy as np

import stablebear as sb
import stablebear.persistence as pers


def test_empty_barcode():
    bc = pers.Barcode(np.zeros((0, 2)))
    betti = pers.barcode_to_betti_curve(bc)

    expected = sb.Pcf(np.zeros((1, 2)))
    assert betti == expected


def test_single_bar_starting_at_zero():
    bc = pers.Barcode(np.array([[0.0, 2.0]]))
    betti = pers.barcode_to_betti_curve(bc)

    expected = sb.Pcf(np.array([[0.0, 1.0], [2.0, 0.0]]))
    assert betti == expected


def test_single_bar_starting_after_zero():
    bc = pers.Barcode(np.array([[1.0, 3.0]]))
    betti = pers.barcode_to_betti_curve(bc)

    expected = sb.Pcf(np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]]))
    assert betti == expected


def test_overlapping_bars():
    # [0,2), [1,3), [0.5,1.5)
    bc = pers.Barcode(np.array([[0.0, 2.0], [1.0, 3.0], [0.5, 1.5]]))
    betti = pers.barcode_to_betti_curve(bc)

    expected = sb.Pcf(np.array([
        [0.0, 1.0], [0.5, 2.0], [1.0, 3.0],
        [1.5, 2.0], [2.0, 1.0], [3.0, 0.0],
    ]))
    assert betti == expected


def test_bars_with_same_birth():
    bc = pers.Barcode(np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]))
    betti = pers.barcode_to_betti_curve(bc)

    expected = sb.Pcf(np.array([
        [0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0],
    ]))
    assert betti == expected


def test_infinite_bar():
    bc = pers.Barcode(np.array([[0.0, np.inf], [1.0, 2.0]]))
    betti = pers.barcode_to_betti_curve(bc)

    # Infinite bar never dies, so count never drops to 0
    expected = sb.Pcf(np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]]))
    assert betti == expected


def test_tensor_barcode_conversion():
    bcs = sb.zeros((3, 4), dtype=sb.barcode64)

    for i in range(bcs.shape[0]):
        for j in range(bcs.shape[1]):
            n_bars = np.random.randint(1, 10)
            births = np.sort(np.abs(np.random.randn(n_bars)))
            deaths = births + np.abs(np.random.randn(n_bars)) + 0.01
            bcs[i, j] = pers.Barcode(np.column_stack([births, deaths]))

    bettis = pers.barcode_to_betti_curve(bcs)

    assert bettis.shape == bcs.shape

    for i in range(bcs.shape[0]):
        for j in range(bcs.shape[1]):
            expected = pers.barcode_to_betti_curve(bcs[i, j])
            assert bettis[i, j] == expected
