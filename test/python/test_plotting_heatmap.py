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
