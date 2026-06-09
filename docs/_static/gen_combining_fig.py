"""Generate the 'combining it all' example figure for the docs."""

import matplotlib
matplotlib.use("Agg")
from pathlib import Path

HERE = Path(__file__).parent

# -- docs snippet start --
import stablebear as sb
from stablebear.random import noisy_sin, noisy_cos
from stablebear.plotting import plot as plotpcf
import matplotlib.pyplot as plt


def plot_combining_example(sin_color="b", cos_color="r"):
    M = 10
    A = sb.zeros((2, M))

    A[0, :] = noisy_sin((M,), n_points=100)
    A[1, :] = noisy_cos((M,), n_points=15)

    fig, ax = plt.subplots(figsize=(6, 2))

    # Plot individual noisy functions
    plotpcf(A[0, :], ax=ax, color=sin_color, linewidth=0.5, alpha=0.4)
    plotpcf(A[1, :], ax=ax, color=cos_color, linewidth=0.5, alpha=0.4)

    # Compute and plot means
    Aavg = sb.mean(A, dim=1)
    plotpcf(Aavg[0], ax=ax, color=sin_color, linewidth=2, label="sin")
    plotpcf(Aavg[1], ax=ax, color=cos_color, linewidth=2, label="cos")

    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.legend()
    fig.tight_layout()
    return fig
# -- docs snippet end --


# -- Generate light and dark variants --
def _save_themed(plot_func, style, bg_color, fg_color, outfile):
    with plt.style.context(style), \
         plt.rc_context({"axes.facecolor": bg_color,
                         "axes.edgecolor": fg_color,
                         "axes.labelcolor": fg_color,
                         "axes.titlecolor": fg_color,
                         "figure.facecolor": bg_color,
                         "text.color": fg_color,
                         "xtick.color": fg_color,
                         "ytick.color": fg_color,
                         "legend.facecolor": bg_color,
                         "legend.edgecolor": fg_color,
                         "legend.labelcolor": fg_color}):
        fig = plot_func()
        fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        print(f"saved {outfile}")


if __name__ == "__main__":
    _save_themed(lambda: plot_combining_example("b", "r"),
                 "default", "white", "black",
                 HERE / "combining_example_light.png")
    _save_themed(lambda: plot_combining_example("#5dade2", "#ff6b6b"),
                 "dark_background", "#1a1a2e", "#e0e0e0",
                 HERE / "combining_example_dark.png")
