import numpy as np
import pytest

from plot_helpers import FigureGallery, gallery_fixture, ax_fixture

import stablebear as sb
from stablebear.plotting import plot_distance_weight_heatmap

_gallery = FigureGallery()
_show_gallery = gallery_fixture(_gallery)
ax = ax_fixture(_gallery)


def test_geometry_and_values(ax):
    gaussian = sb.distributions.Gaussian(0, 1)

    # A 3x3 grid of unit pixels centered at x, y = 0, 1, 2.
    im = plot_distance_weight_heatmap(
        gaussian, query=(1, 0), extent=(-0.5, 2.5, -0.5, 2.5), resolution=3, ax=ax,
    )

    # Euclidean distances from the query (1, 0) to each pixel center.
    # The first row is y = 0, drawn at the bottom.
    distances = np.array([
        [1.0,        0.0, 1.0],
        [np.sqrt(2), 1.0, np.sqrt(2)],
        [np.sqrt(5), 2.0, np.sqrt(5)],
    ])
    assert np.asarray(im.get_array()) == pytest.approx(gaussian.weight(distances))
    assert im.get_extent() == pytest.approx([-0.5, 2.5, -0.5, 2.5])
    assert im.origin == "lower"



def test_automatic_extent_is_a_square_around_the_query(ax):
    # Without extent, the heatmap draws a square centered on the query.
    # Its half-width is the largest |endpoint| of plot_range(), plus 10%:
    # Gaussian(0, 1).plot_range() is (-3, 3), so the half-width is 3.3.
    im = plot_distance_weight_heatmap(
        sb.distributions.Gaussian(0, 1), query=(10, -5), resolution=3, ax=ax,
    )

    assert im.get_extent() == pytest.approx([6.7, 13.3, -8.3, -1.7])


def test_documented_defaults(ax):
    # extent, resolution, and the styling keywords are omitted to test their defaults.
    im = plot_distance_weight_heatmap(sb.distributions.Gaussian(0, 1), ax=ax)

    assert np.asarray(im.get_array()).shape == (512, 512)
    assert ax.get_aspect() == 1.0
    assert im.get_cmap().name == "viridis"
    assert im.get_interpolation() == "nearest"


def test_overflowing_plot_range_asks_for_an_extent(ax):
    # plot_range() is mean +/- 3 std, which overflows to (-inf, inf).
    gaussian = sb.distributions.Gaussian(0, 1e308)

    with pytest.raises(ValueError, match="cannot choose a finite plot region; pass extent explicitly"):
        plot_distance_weight_heatmap(gaussian, resolution=3, ax=ax)


def test_overflowing_plot_range_works_with_an_extent(ax):
    gaussian = sb.distributions.Gaussian(0, 1e308)

    im = plot_distance_weight_heatmap(gaussian, extent=(-1, 1, -1, 1), resolution=3, ax=ax)

    assert im.get_extent() == pytest.approx([-1, 1, -1, 1])
