"""Shared helpers for test files that produce plots viewable via SB_SHOW_PLOTS.

When SB_SHOW_PLOTS is not set, all fixtures are no-ops and matplotlib is
never imported.
"""

import math
import os

import numpy as np
import pytest


SHOW = bool(os.environ.get("SB_SHOW_PLOTS"))


def _get_plt():
    """Lazily import matplotlib.pyplot with the correct backend."""
    import matplotlib
    if SHOW:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt_
    plt_.rcParams["figure.max_open_warning"] = 0
    return plt_


class FigureGallery:
    """Collects figures during a test session and shows them in a Tk viewer."""

    def __init__(self):
        self.entries = []  # list of (fig, source_str_or_None)

    def append(self, fig, source=None):
        self.entries.append((fig, source))

    def show(self, title="Plot Test Viewer"):
        if not self.entries:
            return

        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        images = []
        titles = []
        sources = []
        plt_ = _get_plt()
        for i, (fig, src) in enumerate(self.entries):
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba()).copy()
            t = fig._suptitle.get_text() if fig._suptitle else f"Test {i}"
            images.append(img)
            titles.append(t)
            sources.append(src)
            plt_.close(fig)

        plt_.close("all")

        root = tk.Tk()
        root.title(title)
        root.geometry("1200x600")

        # Left panel: test list
        list_frame = tk.Frame(root)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        tk.Label(list_frame, text="Tests", font=("sans-serif", 12, "bold")).pack(anchor=tk.W)
        listbox = tk.Listbox(list_frame, width=35, font=("monospace", 9),
                             activestyle="dotbox", selectmode=tk.SINGLE)
        listbox.pack(fill=tk.Y, expand=True)
        for i, t in enumerate(titles):
            listbox.insert(tk.END, f"{i + 1}. {t}")

        # Right panel: plot on top, source below
        right_frame = tk.Frame(root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        view_fig = Figure(figsize=(5, 4))
        view_ax = view_fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(view_fig, master=right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        source_text = tk.Text(right_frame, height=12, font=("monospace", 10),
                              wrap=tk.NONE, state=tk.DISABLED, bg="#f5f5f5")
        source_text.pack(fill=tk.X, padx=5, pady=(0, 5))

        def show_idx(idx):
            view_ax.clear()
            view_ax.imshow(images[idx])
            view_ax.set_title(f"[{idx + 1}/{len(images)}] {titles[idx]}")
            view_ax.axis("off")
            view_fig.tight_layout()
            canvas.draw()

            source_text.config(state=tk.NORMAL)
            source_text.delete("1.0", tk.END)
            if sources[idx]:
                source_text.insert(tk.END, sources[idx])
            else:
                source_text.insert(tk.END, "(source not available)")
            source_text.config(state=tk.DISABLED)

        def on_select(_event):
            sel = listbox.curselection()
            if sel:
                show_idx(sel[0])

        listbox.bind("<<ListboxSelect>>", on_select)
        listbox.selection_set(0)
        show_idx(0)

        root.mainloop()


def gallery_fixture(gallery):
    """Create a session-scoped autouse fixture that shows the gallery after tests."""
    @pytest.fixture(scope="session", autouse=True)
    def _show_gallery():
        yield
        if SHOW:
            gallery.show()
    return _show_gallery


def _get_test_source(request):
    """Extract test function source code from a pytest request, or None."""
    import inspect
    try:
        return inspect.getsource(request.node.obj)
    except (OSError, TypeError):
        return None


def ax_fixture(gallery):
    """Create an ``ax`` fixture that collects figures into the gallery.

    Always creates a real axes — use this in tests that need to assert on
    plot state (line counts, collections, etc.).
    """
    @pytest.fixture
    def ax(request):
        plt_ = _get_plt()
        fig, ax = plt_.subplots()
        if SHOW:
            fig.suptitle(request.node.name)
        yield ax
        if SHOW:
            fig.tight_layout()
            gallery.append(fig, source=_get_test_source(request))
        else:
            plt_.close(fig)
    return ax


def rect_plot_fixture(gallery):
    """Create a ``rect_plot`` fixture for visualizing rectangle iteration.

    Call ``rect_plot(f, g, rects)`` from a test to register data for plotting.
    When SB_SHOW_PLOTS is not set, the call is a no-op.
    """
    @pytest.fixture
    def rect_plot(request):
        if not SHOW:
            yield lambda *args, **kwargs: None
            return

        data = {}

        def capture(f, g, rects, *, max_time=None):
            data["f"] = f
            data["g"] = g
            data["rects"] = rects
            data["max_time"] = max_time

        yield capture

        if not data:
            return

        f, g, rects = data["f"], data["g"], data["rects"]
        max_time = data["max_time"]

        plt_ = _get_plt()
        from stablebear.plotting import plot as plot_pcf

        fig, ax = plt_.subplots(figsize=(10, 5))
        fig.suptitle(request.node.name)
        plot_pcf(f, ax=ax, label="f", max_time=max_time)
        plot_pcf(g, ax=ax, label="g", max_time=max_time)

        # Collect all boundary times for dashed vlines
        boundaries = set()
        for rect in rects:
            boundaries.add(rect.left)
            if not math.isinf(rect.right):
                boundaries.add(rect.right)

        for t in sorted(boundaries):
            ax.axvline(t, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

        # Shade each rectangle and label it
        for i, rect in enumerate(rects):
            r_right = rect.right if not math.isinf(rect.right) else (max_time or ax.get_xlim()[1])
            lo = min(rect.f_value, rect.g_value)
            hi = max(rect.f_value, rect.g_value)
            if hi == lo:
                hi = lo + 0.1  # give zero-height rects a sliver of visibility

            from matplotlib.patches import Rectangle as MplRect
            patch = MplRect((rect.left, lo), r_right - rect.left, hi - lo,
                            alpha=0.15, facecolor="C2", edgecolor="C2",
                            linewidth=0.5)
            ax.add_patch(patch)

            # Number label centered in the rectangle
            cx = (rect.left + r_right) / 2
            cy = (lo + hi) / 2
            ax.text(cx, cy, str(i), ha="center", va="center",
                    fontsize=11, fontweight="bold", color="C2")

            # Value annotation below the rectangle
            label = f"l={rect.left:.4g} r={'inf' if math.isinf(rect.right) else f'{rect.right:.4g}'}\nfv={rect.f_value:.4g} gv={rect.g_value:.4g}"
            ax.text(cx, lo, label, ha="center", va="top",
                    fontsize=8, color="grey")

        ax.legend()
        fig.tight_layout()
        gallery.append(fig, source=_get_test_source(request))

    return rect_plot
