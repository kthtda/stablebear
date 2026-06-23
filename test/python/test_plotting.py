import os
import sys

import numpy as np
import pytest

from plot_helpers import FigureGallery, gallery_fixture, ax_fixture, SHOW

import stablebear as sb
from stablebear.persistence import Barcode, BarcodeTensor
from stablebear.plotting import plot, plot_barcode

_gallery = FigureGallery()
_show_gallery = gallery_fixture(_gallery)
ax = ax_fixture(_gallery)


def _make_pcf(pairs, dtype=np.float32):
    return sb.Pcf(np.array(pairs, dtype=dtype))


# ---------- plot() tests ----------

class TestPlot:
    def test_single_pcf(self, ax):
        f = _make_pcf([[0, 1], [1, 3], [2, 0]])
        plot(f, ax=ax)
        assert len(ax.lines) == 1
        line = ax.lines[0]
        np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2])
        np.testing.assert_array_equal(line.get_ydata(), [1, 3, 0])

    def test_tensor_plots_all_functions(self, ax):
        t = sb.PcfTensor([
            _make_pcf([[0, 1], [1, 2]]),
            _make_pcf([[0, 3], [1, 4]]),
            _make_pcf([[0, 5], [1, 6]]),
        ])
        plot(t, ax=ax)
        assert len(ax.lines) == 3
        for i, (y0, y1) in enumerate([(1, 2), (3, 4), (5, 6)]):
            np.testing.assert_array_equal(ax.lines[i].get_xdata(), [0, 1])
            np.testing.assert_array_equal(ax.lines[i].get_ydata(), [y0, y1])

    def test_auto_label(self, ax):
        t = sb.PcfTensor([
            _make_pcf([[0, 1], [1, 2]]),
            _make_pcf([[0, 3], [1, 4]]),
        ])
        plot(t, ax=ax, auto_label=True)
        ax.legend()
        labels = [line.get_label() for line in ax.lines]
        assert labels == ["f0", "f1"]

    def test_no_auto_label_by_default(self, ax):
        t = sb.PcfTensor([
            _make_pcf([[0, 1], [1, 2]]),
            _make_pcf([[0, 3], [1, 4]]),
        ])
        plot(t, ax=ax)
        for line in ax.lines:
            assert not line.get_label().startswith("f")

    def test_max_time_extends_single(self, ax):
        f = _make_pcf([[0, 5], [1, 10]])
        plot(f, ax=ax, max_time=3.0)
        xdata = ax.lines[0].get_xdata()
        assert xdata[-1] == pytest.approx(3.0)

    def test_max_time_extends_tensor(self, ax):
        t = sb.PcfTensor([
            _make_pcf([[0, 1], [1, 2]]),
            _make_pcf([[0, 3], [2, 4]]),
        ])
        plot(t, ax=ax, max_time=5.0)
        for line in ax.lines:
            assert line.get_xdata()[-1] == pytest.approx(5.0)

    def test_tensor_default_max_time_extends_to_latest_breakpoint(self, ax):
        t = sb.PcfTensor([
            _make_pcf([[0, 1], [1, 2]]),
            _make_pcf([[0, 3], [3, 4]]),
        ])
        plot(t, ax=ax)
        # The shorter PCF should be extended to time 3
        xdata0 = ax.lines[0].get_xdata()
        assert xdata0[-1] == pytest.approx(3.0)

    def test_kwargs_passed_through(self, ax):
        f = _make_pcf([[0, 1], [1, 2]])
        plot(f, ax=ax, color="red", linewidth=3)
        line = ax.lines[0]
        assert line.get_color() == "red"
        assert line.get_linewidth() == 3

    def test_squeezable_tensor_is_accepted(self, ax):
        t = sb.PcfTensor([
            [_make_pcf([[0, 1], [1, 2]]), _make_pcf([[0, 3], [1, 4]])],
        ])
        plot(t, ax=ax)
        assert len(ax.lines) == 2

    def test_multidim_tensor_raises(self, ax):
        t = sb.PcfTensor([
            [_make_pcf([[0, 1], [1, 2]]), _make_pcf([[0, 3], [1, 4]])],
            [_make_pcf([[0, 5], [1, 6]]), _make_pcf([[0, 7], [1, 8]])],
        ])
        with pytest.raises(ValueError, match="1-dimensional"):
            plot(t, ax=ax)


# ---------- plot_barcode() tests ----------

def _make_barcode(pairs, dtype=np.float64):
    return Barcode(np.array(pairs, dtype=dtype))


class TestPlotBarcode:
    def test_finite_bars_creates_collection(self, ax):
        bc = _make_barcode([[0, 1], [0.5, 2]])
        plot_barcode(bc, ax=ax)
        assert len(ax.collections) == 1

    def test_return_value_equals_num_bars(self, ax):
        bc = _make_barcode([[0, 1], [0.5, 2], [1, 3]])
        y = plot_barcode(bc, ax=ax)
        assert y == 3

    def test_return_value_with_y_offset(self, ax):
        bc = _make_barcode([[0, 1], [0.5, 2]])
        y = plot_barcode(bc, ax=ax, y_offset=5)
        assert y == 7

    def test_empty_barcode_returns_offset(self, ax):
        bc = _make_barcode(np.empty((0, 2)))
        y = plot_barcode(bc, ax=ax, y_offset=3)
        assert y == 3
        assert len(ax.collections) == 0

    def test_infinite_bars_create_annotations(self, ax):
        bc = _make_barcode([[0, np.inf], [1, np.inf]])
        plot_barcode(bc, ax=ax)
        # Infinite bars produce one LineCollection and one arrow annotation per bar
        assert len(ax.collections) == 1
        assert len(ax.texts) == 2
        for ann in ax.texts:
            assert ann.arrow_patch is not None

    def test_mixed_finite_and_infinite(self, ax):
        bc = _make_barcode([[0, 1], [0.5, np.inf]])
        y = plot_barcode(bc, ax=ax)
        assert y == 2
        assert len(ax.collections) == 1
        # One arrow annotation for the infinite bar
        assert len(ax.texts) == 1
        assert ax.texts[0].arrow_patch is not None

    def test_stacking_barcode_tensor(self, ax):
        bc1 = _make_barcode([[0, 1], [0.5, 2]])
        bc2 = _make_barcode([[0, 3]])
        bt = BarcodeTensor([bc1, bc2])
        y = plot_barcode(bt, ax=ax)
        assert y == 3  # 2 bars from bc1 + 1 bar from bc2
        # Each barcode gets its own LineCollection
        assert len(ax.collections) == 2
        assert len(ax.collections[0].get_segments()) == 2  # bc1: 2 bars
        assert len(ax.collections[1].get_segments()) == 1  # bc2: 1 bar

    def test_kwargs_passed_to_collection(self, ax):
        bc = _make_barcode([[0, 1]])
        plot_barcode(bc, ax=ax, color="blue", linewidth=5)
        lc = ax.collections[0]
        assert lc.get_linewidth()[0] == 5

    def test_squeezable_tensor_is_accepted(self, ax):
        bc = _make_barcode([[0, 1]])
        bt = BarcodeTensor([bc, bc])
        bt2d = bt.reshape((1, 2))
        y = plot_barcode(bt2d, ax=ax)
        assert y == 2

    def test_bars_sorted_by_birth_then_length_descending(self, ax):
        # Bars given out of order
        bc = _make_barcode([[1, 3], [0, np.inf], [0, 2], [0, 5]])
        plot_barcode(bc, ax=ax)
        segs = ax.collections[0].get_segments()
        # Sorted descending by (birth, length): (1,3) len=2, (0,inf), (0,5) len=5, (0,2) len=2
        births = [s[0][0] for s in segs]
        deaths_or_xmax = [s[1][0] for s in segs]
        assert births == [1.0, 0.0, 0.0, 0.0]
        assert deaths_or_xmax[0] == 3.0
        assert deaths_or_xmax[1] > 5.0    # xmax for infinite bar
        assert deaths_or_xmax[2] == 5.0
        assert deaths_or_xmax[3] == 2.0

    def test_y_axis_ticks_hidden(self, ax):
        bc = _make_barcode([[0, 1], [0.5, 2]])
        plot_barcode(bc, ax=ax)
        assert len(ax.yaxis.get_ticklocs()) == 0

    def test_multidim_barcode_tensor_raises(self, ax):
        bc = _make_barcode([[0, 1]])
        bt = BarcodeTensor([bc, bc, bc, bc])
        bt2d = bt.reshape((2, 2))
        with pytest.raises(ValueError, match="1-dimensional"):
            plot_barcode(bt2d, ax=ax)


if __name__ == "__main__":
    # Re-exec with the env var set so the backend selection at the top of
    # this file picks up SB_SHOW_PLOTS before matplotlib is configured.
    if not os.environ.get("SB_SHOW_PLOTS"):
        os.environ["SB_SHOW_PLOTS"] = "1"
        os.execv(sys.executable, [sys.executable, "-m", "pytest", __file__, "-v"] + sys.argv[1:])
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
