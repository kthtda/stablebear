"""Generate the plotting gallery figures for the docs."""

import numpy as np
import stablebear as sb
from stablebear.random import noisy_sin, noisy_cos
from stablebear.plotting import plot as plotpcf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent


# -- docs snippet start single_pcf --
def plot_single_pcf():
    f = sb.Pcf(np.array([[0, 1], [1, 4], [3, 2], [4, 3]], dtype=np.float32))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    plotpcf(f, ax=ax, max_time=6, linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    fig.tight_layout()
    return fig
# -- docs snippet end single_pcf --


# -- docs snippet start overlaid --
def plot_overlaid():
    X = noisy_sin((20,), n_points=80)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    plotpcf(X, ax=ax, alpha=0.3, linewidth=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.set_title("20 noisy sine functions")
    fig.tight_layout()
    return fig
# -- docs snippet end overlaid --


# -- docs snippet start arithmetic --
def plot_arithmetic():
    f = sb.Pcf(np.array([[0, 1], [1, 3], [3, 1]], dtype=np.float32))
    g = sb.Pcf(np.array([[0, 2], [2, 0]], dtype=np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.5), sharex=True, sharey=True)
    for ax, pcf, title in [
        (axes[0], f, "f"),
        (axes[1], g, "g"),
        (axes[2], f + g, "f + g"),
    ]:
        plotpcf(pcf, ax=ax, max_time=5, linewidth=2)
        ax.set_title(title)
        ax.set_ylim(-0.3, 6)
    axes[0].set_ylabel("value")
    fig.tight_layout()
    return fig
# -- docs snippet end arithmetic --


# -- docs snippet start mean_highlight --
def plot_mean_highlight(sin_color="b", cos_color="r"):
    sines = noisy_sin((15,), n_points=100)
    cosines = noisy_cos((15,), n_points=100)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    plotpcf(sines, ax=ax, color=sin_color, linewidth=0.5, alpha=0.2)
    plotpcf(cosines, ax=ax, color=cos_color, linewidth=0.5, alpha=0.2)

    plotpcf(sb.mean(sines), ax=ax, color=sin_color, linewidth=2.5, label="mean(sin)")
    plotpcf(sb.mean(cosines), ax=ax, color=cos_color, linewidth=2.5, label="mean(cos)")

    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.legend()
    fig.tight_layout()
    return fig
# -- docs snippet end mean_highlight --


# -- docs snippet start barcode --
def plot_barcode_example(h0_color="steelblue", h1_color="orangered"):
    from stablebear.persistence import Barcode
    from stablebear.plotting import plot_barcode

    bc_h0 = Barcode(np.array([
        [0.0, np.inf], [0.0, 1.8], [0.0, 0.6], [0.1, 1.2],
    ], dtype=np.float64))

    bc_h1 = Barcode(np.array([
        [0.5, 2.0], [0.8, 1.5], [1.0, 3.0],
    ], dtype=np.float64))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    y = plot_barcode(bc_h0, ax=ax, color=h0_color, linewidth=2, label="H0")
    plot_barcode(bc_h1, ax=ax, y_offset=y + 1, color=h1_color, linewidth=2, label="H1")
    ax.set_xlabel("t")
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    return fig
# -- docs snippet end barcode --


# -- docs snippet start tda_pipeline --
def plot_tda_pipeline(h0_color="steelblue", h1_color="orangered"):
    from stablebear import persistence
    from stablebear.plotting import plot_barcode

    # 1. Noisy circle (clear H1 topology)
    rng = np.random.RandomState(10)
    theta = rng.uniform(0, 2 * np.pi, 30)
    r = 1.0 + rng.normal(0, 0.15, 30)
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).astype(np.float64)

    # 2. Compute persistent homology
    bcs = persistence.compute_persistent_homology(points, max_dim=1, verbose=False)
    bc_h0, bc_h1 = bcs[0], bcs[1]

    # 3. Convert to stable rank
    sranks = persistence.barcode_to_stable_rank(bcs)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3),
                                         gridspec_kw={"width_ratios": [1, 1, 1.2]})

    # Left: point cloud
    ax1.scatter(points[:, 0], points[:, 1], s=15, color="grey", edgecolors="black",
                linewidths=0.5)
    ax1.set_aspect("equal")
    ax1.set_title("Point cloud")

    # Middle: persistence diagram (via persim)
    import persim
    persim.plot_diagrams(
        [np.asarray(bc_h0), np.asarray(bc_h1)],
        ax=ax2, legend=True, show=False,
    )
    legend = ax2.get_legend()
    legend.get_frame().set_alpha(0)
    fg = ax2.xaxis.label.get_color()
    for text in legend.get_texts():
        text.set_color(fg)
    for line in ax2.get_lines():
        line.set_color(fg)
    ax2.set_title("Persistence diagram")

    # Right: stable rank
    plotpcf(sranks[0], ax=ax3, max_time=2, color=h0_color, linewidth=2,
            label="H0")
    plotpcf(sranks[1], ax=ax3, max_time=2, color=h1_color, linewidth=2,
            label="H1")
    ax3.set_xlabel("t")
    ax3.set_ylabel("rank")
    ax3.set_title("Stable rank")
    leg3 = ax3.legend(fontsize=8)
    leg3.get_frame().set_alpha(0)
    for text in leg3.get_texts():
        text.set_color(fg)

    fig.tight_layout(w_pad=1.5)
    return fig
# -- docs snippet end tda_pipeline --


# -- docs snippet start betti_pipeline --
def plot_betti_pipeline(h0_color="steelblue", h1_color="orangered"):
    from stablebear import persistence
    from stablebear.plotting import plot_barcode

    # 1. Noisy circle
    rng = np.random.RandomState(10)
    theta = rng.uniform(0, 2 * np.pi, 30)
    r = 1.0 + rng.normal(0, 0.15, 30)
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).astype(np.float64)

    # 2. Compute persistent homology
    bcs = persistence.compute_persistent_homology(points, max_dim=1, verbose=False)
    bc_h0, bc_h1 = bcs[0], bcs[1]

    # 3. Convert to Betti curves
    bettis = persistence.barcode_to_betti_curve(bcs, verbose=False)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3),
                                         gridspec_kw={"width_ratios": [1, 1, 1.2]})

    # Left: point cloud
    ax1.scatter(points[:, 0], points[:, 1], s=15, color="grey",
                edgecolors="black", linewidths=0.5)
    ax1.set_aspect("equal")
    ax1.set_title("Point cloud")

    # Middle: persistence barcode
    y = plot_barcode(bc_h0, ax=ax2, color=h0_color, linewidth=2, label="H0")
    plot_barcode(bc_h1, ax=ax2, y_offset=y + 1, color=h1_color, linewidth=2,
                 label="H1")
    ax2.set_yticks([])
    ax2.set_xlabel("t")
    ax2.set_title("Persistence barcode")
    ax2.legend(fontsize=8)

    # Right: Betti curves
    plotpcf(bettis[0], ax=ax3, max_time=2, color=h0_color, linewidth=2,
            label="H0")
    plotpcf(bettis[1], ax=ax3, max_time=2, color=h1_color, linewidth=2,
            label="H1")
    ax3.set_xlabel("t")
    ax3.set_ylabel("count")
    ax3.set_title("Betti curve")
    ax3.legend(fontsize=8)

    fig.tight_layout(w_pad=1.5)
    return fig
# -- docs snippet end betti_pipeline --


# -- docs snippet start apf --
def plot_apf_example(h0_color="steelblue", h1_color="orangered"):
    from stablebear import persistence
    from stablebear.plotting import plot_barcode

    # Noisy circle
    rng = np.random.RandomState(10)
    theta = rng.uniform(0, 2 * np.pi, 30)
    r = 1.0 + rng.normal(0, 0.15, 30)
    points = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).astype(np.float64)

    bcs = persistence.compute_persistent_homology(points, max_dim=1, verbose=False)
    bc_h0, bc_h1 = bcs[0], bcs[1]
    apfs = persistence.barcode_to_accumulated_persistence(bcs, verbose=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # Left: barcode
    y = plot_barcode(bc_h0, ax=ax1, color=h0_color, linewidth=2, label="H0")
    plot_barcode(bc_h1, ax=ax1, y_offset=y + 1, color=h1_color, linewidth=2,
                 label="H1")
    ax1.set_yticks([])
    ax1.set_xlabel("t")
    ax1.set_title("Persistence barcode")
    ax1.legend(fontsize=8)

    # Right: APF
    plotpcf(apfs[0], ax=ax2, max_time=2, color=h0_color, linewidth=2, label="H0")
    plotpcf(apfs[1], ax=ax2, max_time=2, color=h1_color, linewidth=2, label="H1")
    ax2.set_xlabel("m")
    ax2.set_ylabel("APF(m)")
    ax2.set_title("Accumulated persistence function")
    ax2.legend(fontsize=8)

    fig.tight_layout(w_pad=1.5)
    return fig
# -- docs snippet end apf --


# -- docs snippet start apf_max_death --
def plot_apf_max_death(color1="steelblue", color2="orangered"):
    from stablebear.persistence import Barcode, barcode_to_accumulated_persistence
    from stablebear.plotting import plot_barcode

    bc = Barcode(np.array([
        [0.0, 0.5], [0.0, 0.5], [0.0, 0.5],
        [0.62, 0.75], [0.1, 1.8],
    ], dtype=np.float64))

    apf_all = barcode_to_accumulated_persistence(bc)
    apf_cut = barcode_to_accumulated_persistence(bc, max_death=1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5))

    # Left: barcode with cutoff line
    plot_barcode(bc, ax=ax1, color=color1, linewidth=2)
    ax1.axvline(1.0, color=color2, linestyle="--", linewidth=1.5, label="max_death=1.0")
    ax1.set_yticks([])
    ax1.set_xlabel("t")
    ax1.set_title("Barcode")
    ax1.legend(fontsize=8, loc="lower right")

    # Right: APF with and without cutoff
    plotpcf(apf_all, ax=ax2, max_time=1.5, color=color1, linewidth=2,
            label="no cutoff")
    plotpcf(apf_cut, ax=ax2, max_time=1.5, color=color2, linewidth=2,
            linestyle="--", label="max_death=1.0")
    ax2.set_xlabel("m")
    ax2.set_ylabel("APF(m)")
    ax2.set_title("Accumulated persistence function")
    ax2.legend(fontsize=8)

    fig.tight_layout(w_pad=1.5)
    return fig
# -- docs snippet end apf_max_death --


# -- docs snippet start poisson_samples --
def plot_poisson_samples(color="steelblue"):
    from stablebear.point_process import sample_poisson
    import stablebear as sb

    gen = sb.random.Generator(seed=42)
    X = sample_poisson((3,), dim=2, rate=50.0, generator=gen)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, ax in enumerate(axes):
        pc = np.asarray(X[i])
        ax.scatter(pc[:, 0], pc[:, 1], s=8, color=color, alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"Sample {i + 1} ({pc.shape[0]} points)")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    fig.tight_layout()
    return fig
# -- docs snippet end poisson_samples --


# -- docs snippet start subsample_weights --
def plot_subsample_weights(query_color="red", cmap="viridis"):
    import stablebear as sb

    # A reference cloud and a single query point to view it from.
    gen = np.random.default_rng(0)
    reference = gen.standard_normal((400, 2))
    query = np.array([1.0, 0.5])

    # The default filter is Euclidean distance to the query point.
    dist = np.linalg.norm(reference - query, axis=1)

    # Each distribution turns those distances into sampling weights.
    distributions = [
        ("Gaussian(mean=0, sigma=0.4)", sb.Gaussian(mean=0.0, sigma=0.4)),
        ("Uniform(inner=1.0, outer=1.8)", sb.Uniform(inner=1.0, outer=1.8)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for ax, (title, distribution) in zip(axes, distributions):
        weights = distribution(dist)
        sc = ax.scatter(reference[:, 0], reference[:, 1], c=weights,
                        cmap=cmap, s=18, edgecolors="none")
        ax.scatter(query[0], query[1], marker="*", s=260, color=query_color,
                   edgecolors="black", linewidths=0.6, label="query point", zorder=3)
        fig.colorbar(sc, ax=ax, label="sampling weight", shrink=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("$x_1$")
        ax.set_aspect("equal")
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig
# -- docs snippet end subsample_weights --


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
                         "legend.facecolor": bg_color,
                         "legend.edgecolor": fg_color,
                         "legend.labelcolor": fg_color,
                         "lines.color": line_color}):
        fig = plot_func()
        fig.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        print(f"saved {outfile}")


LIGHT = ("default", "white", "black", "steelblue")
DARK = ("dark_background", "#1a1a2e", "#e0e0e0", "#5dade2")

if __name__ == "__main__":
    gallery = [
        ("gallery_single_pcf", lambda: plot_single_pcf()),
        ("gallery_overlaid", lambda: plot_overlaid()),
        ("gallery_arithmetic", lambda: plot_arithmetic()),
        ("gallery_mean_highlight", lambda: plot_mean_highlight("b", "r")),
        ("gallery_barcode", lambda: plot_barcode_example("steelblue", "orangered")),
        ("gallery_tda_pipeline", lambda: plot_tda_pipeline("steelblue", "orangered")),
        ("gallery_betti_pipeline", lambda: plot_betti_pipeline("steelblue", "orangered")),
        ("gallery_apf", lambda: plot_apf_example("steelblue", "orangered")),
        ("gallery_apf_max_death", lambda: plot_apf_max_death("steelblue", "orangered")),
        ("gallery_poisson_samples", lambda: plot_poisson_samples("steelblue")),
        ("gallery_subsample_weights", lambda: plot_subsample_weights("red")),
    ]
    gallery_dark = [
        ("gallery_single_pcf", lambda: plot_single_pcf()),
        ("gallery_overlaid", lambda: plot_overlaid()),
        ("gallery_arithmetic", lambda: plot_arithmetic()),
        ("gallery_mean_highlight", lambda: plot_mean_highlight("#5dade2", "#ff6b6b")),
        ("gallery_barcode", lambda: plot_barcode_example("#5dade2", "#ff6b6b")),
        ("gallery_tda_pipeline", lambda: plot_tda_pipeline("#5dade2", "#ff6b6b")),
        ("gallery_betti_pipeline", lambda: plot_betti_pipeline("#5dade2", "#ff6b6b")),
        ("gallery_apf", lambda: plot_apf_example("#5dade2", "#ff6b6b")),
        ("gallery_apf_max_death", lambda: plot_apf_max_death("#5dade2", "#ff6b6b")),
        ("gallery_poisson_samples", lambda: plot_poisson_samples("#5dade2")),
        ("gallery_subsample_weights", lambda: plot_subsample_weights("#ff6b6b")),
    ]
    for (name, func), (_, func_dark) in zip(gallery, gallery_dark):
        _save_themed(func, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(func_dark, *DARK, HERE / f"{name}_dark.png")
