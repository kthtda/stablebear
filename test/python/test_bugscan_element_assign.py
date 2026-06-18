import numpy as np
import pytest

import stablebear as sb
from stablebear.distance_matrix import DistanceMatrix
from stablebear.symmetric_matrix import SymmetricMatrix


# ---------------------------------------------------------------------------
# Bug #39: element assignment into PointCloud / DistanceMatrix / SymmetricMatrix
# tensors shared the source's std::shared_ptr buffer (a shallow copy), so cells
# aliased the source object and each other. Storing an element must copy its
# data (NumPy object-array .copy() semantics).
# ---------------------------------------------------------------------------


def test_distmat_cell_independent_of_source_after_assign():
    t = sb.zeros((2,), dtype=sb.distmat64)
    m = DistanceMatrix(3)
    m[0, 1] = 2.0
    t[0] = m
    m[0, 2] = 8.0                       # mutate the source after assignment
    assert t[0][0, 2] == 0.0           # cell not affected
    assert t[0][0, 1] == 2.0           # value at assignment time is preserved


@pytest.mark.parametrize(
    "dt, mk",
    [
        pytest.param(sb.distmat64, lambda: DistanceMatrix(3), id="distmat"),
        pytest.param(sb.symmat64, lambda: SymmetricMatrix(3), id="symmat"),
    ],
)
def test_matrix_cells_from_same_source_are_independent(dt, mk):
    t = sb.zeros((2,), dtype=dt)
    shared = mk()
    t[0] = shared
    t[1] = shared
    t[0][0, 1] = 99.0
    assert t[0][0, 1] == 99.0          # the write landed on cell 0
    assert t[1][0, 1] == 0.0           # sibling unaffected
    assert shared[0, 1] == 0.0         # source unaffected


def test_pcloud_cell_independent_of_source_and_sibling():
    t = sb.zeros((2,), dtype=sb.pcloud64)
    src = sb.FloatTensor(np.zeros((3, 2)))
    t[0] = src
    t[1] = src
    t[0][0, 0] = 99.0
    assert float(t[1][0, 0]) == 0.0    # sibling unaffected
    assert float(src[0, 0]) == 0.0     # source FloatTensor unaffected


@pytest.mark.parametrize(
    "dt, mk",
    [
        pytest.param(sb.distmat64, lambda: DistanceMatrix(3), id="distmat"),
        pytest.param(sb.symmat64, lambda: SymmetricMatrix(3), id="symmat"),
    ],
)
def test_matrix_slice_assignment_is_independent(dt, mk):
    t = sb.zeros((2,), dtype=dt)
    srct = sb.zeros((2,), dtype=dt)
    srct[0] = mk()
    srct[1] = mk()
    t[0:2] = srct
    srct[0][0, 1] = 77.0               # mutate the source tensor's cell
    assert t[0][0, 1] == 0.0


def test_mutating_one_cell_leaves_others_untouched():
    t = sb.zeros((3,), dtype=sb.distmat64)
    for i in range(3):
        t[i] = DistanceMatrix(3)
    t[1][0, 1] = 5.0
    assert t[0][0, 1] == 0.0
    assert t[2][0, 1] == 0.0
    assert t[1][0, 1] == 5.0


# ---------------------------------------------------------------------------
# Bug #129: self-aliasing slice assignment of an object-element tensor into
# itself (t[1:] = t[:-1], t[:] = t[::-1]) must leave every cell independent.
# The overlap-copy path only duplicated the outer buffer, so the matrix
# elements stayed shallow-shared with the source cells; it now deep-copies
# each stored element too.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dt, mk",
    [
        pytest.param(sb.distmat64, lambda v: _distmat(v), id="distmat"),
        pytest.param(sb.symmat64, lambda v: _symmat(v), id="symmat"),
    ],
)
def test_forward_overlap_self_assignment_independent(dt, mk):
    t = sb.zeros((3,), dtype=dt)
    for i in range(3):
        t[i] = mk(i + 1)                  # values 1, 2, 3
    t[1:] = t[:-1]                         # shift down -> values 1, 1, 2
    assert [t[i][0, 1] for i in range(3)] == [1.0, 1.0, 2.0]
    # every cell is an independent copy
    t[0][0, 1] = 99.0
    assert t[1][0, 1] == 1.0
    assert t[2][0, 1] == 2.0


@pytest.mark.parametrize(
    "dt, mk",
    [
        pytest.param(sb.distmat64, lambda v: _distmat(v), id="distmat"),
        pytest.param(sb.symmat64, lambda v: _symmat(v), id="symmat"),
    ],
)
def test_reverse_self_assignment_independent(dt, mk):
    t = sb.zeros((2,), dtype=dt)
    t[0] = mk(1)
    t[1] = mk(2)
    t[:] = t[::-1]                         # swap
    assert t[0][0, 1] == 2.0
    assert t[1][0, 1] == 1.0
    t[0][0, 1] = 77.0                      # mutate one cell
    assert t[1][0, 1] == 1.0              # sibling unaffected


def _distmat(v):
    m = DistanceMatrix(3)
    m[0, 1] = float(v)
    return m


def _symmat(v):
    m = SymmetricMatrix(3)
    m[0, 1] = float(v)
    return m
