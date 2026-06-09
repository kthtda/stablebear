"""Generate the tensor evaluation example figures for the docs."""

import numpy as np
import stablebear as sb
from stablebear.plotting import plot as plotpcf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent

# -- PCFs used in the docs example --
# f(t) = 1 on [0,1), 4 on [1,3), 2 on [3,inf)
f = sb.Pcf(np.array([[0, 1], [1, 4], [3, 2]], dtype=np.float32))

# g(t) = 1 on [0,2), 2 on [2,inf)
g = sb.Pcf(np.array([[0, 1], [2, 2]], dtype=np.float32))

X = sb.zeros((2, 2))
X[0, 0] = f
X[0, 1] = g
X[1, 0] = 0.5 * g
X[1, 1] = f


# -- included in docs via literalinclude :pyobject: --
def plot_pcf_definitions():
    fig, (ax_f, ax_g) = plt.subplots(1, 2, figsize=(6, 2.5),
                                      sharex=True, sharey=True)
    plotpcf(f, ax=ax_f, max_time=5, linewidth=2)
    ax_f.set_title("f")
    ax_f.set_ylim(-0.3, 5)

    plotpcf(g, ax=ax_g, max_time=5, linewidth=2)
    ax_g.set_title("g")
    ax_g.set_ylim(-0.3, 5)

    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_tensor_eval_example():
    t_eval = 2
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=True)

    for ax, pcf, label in [
        (axes[0, 0], f, "X[0,0] = f"),
        (axes[0, 1], g, "X[0,1] = g"),
        (axes[1, 0], 0.5 * g, "X[1,0] = 0.5g"),
        (axes[1, 1], f, "X[1,1] = f"),
    ]:
        plotpcf(pcf, ax=ax, max_time=5, linewidth=2)
        val = pcf(t_eval)
        ax.axvline(t_eval, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(t_eval, val, "ro", markersize=6, zorder=5)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-0.3, 5)

    fig.suptitle(f"X({t_eval}) = {X(t_eval).tolist()}", fontsize=11)
    fig.tight_layout()
    return fig


# -- included in docs via literalinclude :pyobject: --
def plot_tensor_eval_array():
    times = [1, 2, 4]
    fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=True)

    for ax, pcf, label in [
        (axes[0, 0], f, "X[0,0] = f"),
        (axes[0, 1], g, "X[0,1] = g"),
        (axes[1, 0], 0.5 * g, "X[1,0] = 0.5g"),
        (axes[1, 1], f, "X[1,1] = f"),
    ]:
        plotpcf(pcf, ax=ax, max_time=5, linewidth=2)
        for t in times:
            val = pcf(t)
            ax.axvline(t, color="red", linestyle="--", linewidth=1, alpha=0.5)
            ax.plot(t, val, "ro", markersize=6, zorder=5)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-0.3, 5)

    fig.suptitle(f"X({times}) — shape (2, 2, 3)", fontsize=11)
    fig.tight_layout()
    return fig


# -- Generate light and dark variants --
def _save_themed(plot_func, style, bg_color, fg_color, line_color, outfile):
    with plt.style.context(style), \
         plt.rc_context({"axes.facecolor": bg_color,
                         "axes.edgecolor": fg_color,
                         "axes.labelcolor": fg_color,
                         "axes.titlecolor": fg_color,
                         "figure.facecolor": bg_color,
                         "text.color": fg_color,
                         "xtick.color": fg_color,
                         "ytick.color": fg_color,
                         "lines.color": line_color}):
        fig = plot_func()
        fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        print(f"saved {outfile}")


LIGHT = ("default", "white", "black", "steelblue")
DARK = ("dark_background", "#1a1a2e", "#e0e0e0", "#5dade2")

if __name__ == "__main__":
    for name, plot_func in [("pcf_definitions", plot_pcf_definitions),
                             ("tensor_eval_example", plot_tensor_eval_example),
                             ("tensor_eval_array", plot_tensor_eval_array)]:
        _save_themed(plot_func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(plot_func, *DARK, HERE / f"{name}_dark.png")
