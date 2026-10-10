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
