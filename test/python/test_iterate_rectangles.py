"""Tests for iterate_rectangles — direct rectangle decomposition of PCF pairs."""

import math
import os
import sys

import numpy as np
import pytest

from plot_helpers import FigureGallery, gallery_fixture, rect_plot_fixture, SHOW

import stablebear as sb

_gallery = FigureGallery()
_show_gallery = gallery_fixture(_gallery)
rect_plot = rect_plot_fixture(_gallery)


def _pcf(points, dtype):
    return sb.Pcf(np.array(points), dtype=dtype)


def _assert_rect(rect, *, l, r, fv, gv):
    """Assert that a Rectangle has the expected values (with approx tolerance)."""
    assert rect.left == pytest.approx(l)
    assert rect.right == pytest.approx(r)
    assert rect.f_value == pytest.approx(fv)
    assert rect.g_value == pytest.approx(gv)


def _assert_rect_inf(rect, *, l, fv, gv):
    """Assert a Rectangle with infinite right boundary."""
    assert rect.left == pytest.approx(l)
    assert math.isinf(rect.right)
    assert rect.f_value == pytest.approx(fv)
    assert rect.g_value == pytest.approx(gv)


# --- Baseline (mirrors C++ Full test) ---

def test_full_iteration(pcf_dtype, rect_plot):
    # f: {0->3, 1->2, 4->5, 6->0}
    # g: {0->2, 3->4, 4->2, 5->1, 8->3}
    f = _pcf([[0, 3], [1, 2], [4, 5], [6, 0]], pcf_dtype)
    g = _pcf([[0, 2], [3, 4], [4, 2], [5, 1], [8, 3]], pcf_dtype)

    rects = sb.iterate_rectangles(f, g)
    rect_plot(f, g, rects, max_time=10)

    assert len(rects) == 7
    _assert_rect(rects[0], l=0, r=1, fv=3, gv=2)
    _assert_rect(rects[1], l=1, r=3, fv=2, gv=2)
    _assert_rect(rects[2], l=3, r=4, fv=2, gv=4)
    _assert_rect(rects[3], l=4, r=5, fv=5, gv=2)
    _assert_rect(rects[4], l=5, r=6, fv=5, gv=1)
    _assert_rect(rects[5], l=6, r=8, fv=0, gv=1)
    _assert_rect_inf(rects[6], l=8, fv=0, gv=3)


# --- Gap: g exhausts before f ---

def test_g_shorter_than_f(pcf_dtype, rect_plot):
    # f has more changepoints than g, so g runs out first.
    f = _pcf([[0, 1], [2, 3], [4, 5], [6, 0]], pcf_dtype)
    g = _pcf([[0, 10], [1, 0]], pcf_dtype)

    rects = sb.iterate_rectangles(f, g)
    rect_plot(f, g, rects, max_time=8)

    assert len(rects) == 5
    _assert_rect(rects[0], l=0, r=1, fv=1, gv=10)
    _assert_rect(rects[1], l=1, r=2, fv=1, gv=0)
    _assert_rect(rects[2], l=2, r=4, fv=3, gv=0)
    _assert_rect(rects[3], l=4, r=6, fv=5, gv=0)
    _assert_rect_inf(rects[4], l=6, fv=0, gv=0)


# --- Gap: simultaneous changes at every step ---

def test_simultaneous_changes(pcf_dtype, rect_plot):
    # Same breakpoints, different values. Every step advances both pointers.
    f = _pcf([[0, 1], [2, 3], [4, 0]], pcf_dtype)
    g = _pcf([[0, 10], [2, 20], [4, 0]], pcf_dtype)

    rects = sb.iterate_rectangles(f, g)
    rect_plot(f, g, rects, max_time=6)

    assert len(rects) == 3
    _assert_rect(rects[0], l=0, r=2, fv=1, gv=10)
    _assert_rect(rects[1], l=2, r=4, fv=3, gv=20)
    _assert_rect_inf(rects[2], l=4, fv=0, gv=0)


# --- Gap: identical PCFs ---

def test_identical_pcfs(pcf_dtype):
    f = _pcf([[0, 5], [1, 3], [3, 0]], pcf_dtype)
    g = _pcf([[0, 5], [1, 3], [3, 0]], pcf_dtype)

    rects = sb.iterate_rectangles(f, g)

    for r in rects:
        assert r.f_value == pytest.approx(r.g_value)


# --- Gap: single-point (constant) PCFs ---

def test_single_point_pcfs(pcf_dtype):
    f = _pcf([[0, 7]], pcf_dtype)
    g = _pcf([[0, 3]], pcf_dtype)

    rects = sb.iterate_rectangles(f, g)

    assert len(rects) == 1
    _assert_rect_inf(rects[0], l=0, fv=7, gv=3)


# --- Gap: b exactly on a changepoint ---

def test_b_on_changepoint(pcf_dtype):
    f = _pcf([[0, 3], [1, 2], [4, 5], [6, 0]], pcf_dtype)
    g = _pcf([[0, 2], [3, 4], [4, 2], [5, 1], [8, 3]], pcf_dtype)

    rects = sb.iterate_rectangles(f, g, b=4.0)

    assert len(rects) == 3
    _assert_rect(rects[0], l=0, r=1, fv=3, gv=2)
    _assert_rect(rects[1], l=1, r=3, fv=2, gv=2)
    _assert_rect(rects[2], l=3, r=4, fv=2, gv=4)


# --- Gap: a == b (zero-width interval) ---

def test_a_equals_b(pcf_dtype):
    f = _pcf([[0, 3], [1, 2]], pcf_dtype)
    g = _pcf([[0, 1], [2, 0]], pcf_dtype)

    rects = sb.iterate_rectangles(f, g, a=1.0, b=1.0)

    assert len(rects) == 0


# --- Gap: swapped arguments ---

def test_swapped_args(pcf_dtype):
    f = _pcf([[0, 3], [1, 2], [4, 5], [6, 0]], pcf_dtype)
    g = _pcf([[0, 2], [3, 4], [4, 2], [5, 1], [8, 3]], pcf_dtype)

    rects_fg = sb.iterate_rectangles(f, g, b=6.0)
    rects_gf = sb.iterate_rectangles(g, f, b=6.0)

    assert len(rects_fg) == len(rects_gf)
    for r1, r2 in zip(rects_fg, rects_gf):
        assert r1.left == pytest.approx(r2.left)
        assert r1.right == pytest.approx(r2.right)
        assert r1.f_value == pytest.approx(r2.g_value)
        assert r1.g_value == pytest.approx(r2.f_value)


# --- Gap: non-overlapping active regions ---

def test_non_overlapping(pcf_dtype, rect_plot):
    # f active on [0,2), g active on [5,7). No overlap.
    f = _pcf([[0, 4], [2, 0]], pcf_dtype)
    g = _pcf([[0, 0], [5, 3], [7, 0]], pcf_dtype)

    rects = sb.iterate_rectangles(f, g, b=10.0)
    rect_plot(f, g, rects, max_time=10)

    for r in rects:
        assert r.f_value == pytest.approx(0) or r.g_value == pytest.approx(0)


# --- Bounded iteration ---

def test_start_and_end_bounds(pcf_dtype, rect_plot):
    f = _pcf([[0, 3], [1, 2], [4, 5], [6, 0]], pcf_dtype)
    g = _pcf([[0, 2], [3, 4], [4, 2], [5, 1], [8, 3]], pcf_dtype)

    rects = sb.iterate_rectangles(f, g, a=2.0, b=4.5)
    rect_plot(f, g, rects, max_time=9)

    assert len(rects) == 3
    _assert_rect(rects[0], l=2, r=3, fv=2, gv=2)
    _assert_rect(rects[1], l=3, r=4, fv=2, gv=4)
    _assert_rect(rects[2], l=4, r=4.5, fv=5, gv=2)


if __name__ == "__main__":
    if not os.environ.get("SB_SHOW_PLOTS"):
        os.environ["SB_SHOW_PLOTS"] = "1"
        os.execv(sys.executable, [sys.executable, "-m", "pytest", __file__, "-v"] + sys.argv[1:])
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
