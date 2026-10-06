"""Generate the homological kernel guide figures for the docs."""

import numpy as np
import stablebear as sb
from stablebear import persistence
from stablebear.plotting import plot as plotpcf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

HERE = Path(__file__).parent


# -- docs snippet start hkernel_quartet_kernel --
def diagonal_projection(cloud):
    """Project every point of a 2D point cloud onto the diagonal y = x."""
    midpoint = cloud.mean(axis=1, keepdims=True)
    return np.broadcast_to(midpoint, cloud.shape).copy()


def kernel_stable_rank(cloud):
    """Stable rank of the homological kernel of a point cloud and its projection."""
    kernels = persistence.compute_homological_kernel(cloud, diagonal_projection(cloud))
    return persistence.barcode_to_stable_rank(kernels)


def kernel_score(stable_ranks):
    """The scalar score: the area under the stable rank."""
    return float(np.asarray(sb.lp_norm(stable_ranks, p=1))[0])
# -- docs snippet end hkernel_quartet_kernel --


# The small "correlation point cloud" used throughout the Background section: two
# points on the diagonal and two on opposite sides of it. All four project to
# distinct spots, so every merge happens at a visibly positive scale.
BACKGROUND_CLOUD = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 4.0], [3.0, 3.0]])


# -- docs snippet start hkernel_projection --
def plot_projection(color="#2a78d6", highlight="#eb6834",
                    proj_color="#444444", grid_color="0.6"):
    # Two points on the same side of the diagonal.
    cloud = np.array([[2.0, 2.5], [2.0, 5.5]])
    projected = diagonal_projection(cloud)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.axline((0, 0), slope=1, color=grid_color, linestyle="--", linewidth=0.8)
    for p, q in zip(cloud, projected):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=grid_color,
                linewidth=0.8, alpha=0.8)

    # The pair before and after: the projection pulls the two points closer
    # together.
    pair, proj_pair = cloud, projected
    d = float(np.linalg.norm(pair[0] - pair[1]))
    dp = float(np.linalg.norm(proj_pair[0] - proj_pair[1]))
    ax.plot(pair[:, 0], pair[:, 1], color=highlight, linewidth=1.8)
    ax.plot(proj_pair[:, 0], proj_pair[:, 1], color=highlight, linewidth=2.6)
    ax.annotate(f"$d$ = {d:.2f}", xy=pair.mean(axis=0),
                xytext=(8, 0), textcoords="offset points",
                fontsize=10, color=highlight)
    ax.annotate(f"$d'$ = {dp:.2f}", xy=proj_pair.mean(axis=0),
                xytext=(12, -12), textcoords="offset points",
                fontsize=10, color=highlight)

    ax.scatter(cloud[:, 0], cloud[:, 1], s=36, color=color, zorder=3,
               label="original points")
    ax.scatter(projected[:, 0], projected[:, 1], s=44, color=proj_color,
               marker="x", linewidth=1.6, zorder=4,
               label="projected onto the diagonal")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_xlim(1.0, 6.0)
    ax.set_ylim(1.0, 6.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig
# -- docs snippet end hkernel_projection --


def _components(points, t):
    """Map each point index to the smallest index in its component when points
    within distance t are considered connected."""
    n = len(points)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) <= t + 1e-9:
                parent[find(i)] = find(j)
    roots = {}
    for i in range(n):
        roots.setdefault(find(i), []).append(i)
    return {i: min(members) for members in roots.values() for i in members}


def _mst_edges(points):
    """Kruskal: the (length, i, j) edges of the minimum spanning tree."""
    n = len(points)
    pairs = sorted((float(np.linalg.norm(points[i] - points[j])), i, j)
                   for i in range(n) for j in range(i + 1, n))
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = []
    for w, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            edges.append((w, i, j))
    return edges


# -- docs snippet start hkernel_merging --
def plot_merging(colors=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
                 grid_color="0.6"):
    cloud = BACKGROUND_CLOUD
    projected = diagonal_projection(cloud)
    bars = persistence.compute_homological_kernel(cloud, projected)[0].to_numpy()
    # Snapshot the actual merge scales, so each column shows the situation at
    # a birth or death of a bar: the first and last merge under d', then the
    # final merge under d.
    scales = [0.0, bars[:, 0].min(), bars[:, 0].max(), bars[:, 1].max()]

    fig = plt.figure(figsize=(10, 6.4), layout="constrained")
    gs = fig.add_gridspec(3, len(scales), height_ratios=(2, 2, 1.15),
                          hspace=0.06)

    for row, (points, label) in enumerate([(cloud, "under $d$\n(original)"),
                                           (projected, "under $d'$\n(projected)")]):
        # H0 only sees connectivity, so draw just the tree edges that cause
        # the merges (the minimum spanning tree) -- the remaining edges of the
        # full Rips graph add nothing and would clutter the picture.
        mst = _mst_edges(points)
        for col, t in enumerate(scales):
            ax = fig.add_subplot(gs[row, col])
            # The diagonal the points are projected onto, in both rows: in the
            # original point cloud it shows where each point will land.
            ax.axline((0, 0), slope=1, color=grid_color, linestyle="--",
                      linewidth=0.8)
            comp = _components(points, t)
            for w, i, j in mst:
                if w <= t + 1e-9:
                    ax.plot(points[[i, j], 0], points[[i, j], 1],
                            color=grid_color, linewidth=1.0, zorder=1)
            # One color per cluster, stable across panels.
            point_colors = [colors[comp[i]] for i in range(len(points))]
            ax.scatter(points[:, 0], points[:, 1], s=42,
                       c=point_colors, zorder=2)
            if row == 0:
                n_d = len(set(_components(cloud, t).values()))
                n_dp = len(set(_components(projected, t).values()))
                ax.set_title(f"$t$ = {round(t, 2):g}\n{n_d} vs {n_dp} clusters",
                             fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_xlim(-0.9, 5.4)
            ax.set_ylim(-0.9, 5.4)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

    # The resulting homological kernel barcode on the same scale axis: each
    # bar is born at a merge under d' and dies at the matching merge under d.
    ax = fig.add_subplot(gs[2, :])
    for k, (birth, death) in enumerate(sorted(bars, key=lambda b: b[0])):
        ax.plot([birth, death], [k, k], color=colors[0], linewidth=3,
                solid_capstyle="round")
        ax.scatter([birth], [k], s=26, color=colors[0], zorder=3)
        ax.scatter([death], [k], s=34, facecolors="none",
                   edgecolors=colors[0], zorder=3)
    for t in scales:
        ax.axvline(t, color=grid_color, linestyle=":", linewidth=0.8)
    ax.set_xlim(-0.25, 4.6)
    ax.set_ylim(-0.6, len(bars) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("$t$")
    ax.set_ylabel("kernel bars", fontsize=10)
    return fig
# -- docs snippet end hkernel_merging --


def anscombes_quartet():
    x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=np.float64)
    x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=np.float64)
    ys = {
        "I": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        "II": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
        "III": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
        "IV": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
    }
    return {name: np.column_stack([x4 if name == "IV" else x,
                                   np.array(y, dtype=np.float64)])
            for name, y in ys.items()}


# -- docs snippet start hkernel_quartet --
def plot_quartet(colors=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
                 proj_color="#444444", grid_color="0.6"):
    quartet = anscombes_quartet()

    # Each dataset keeps one color across both of its panels (the same colors
    # the invariants figure uses); the projections stay neutral so they never
    # collide with a dataset's color.
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
    for column, color, (name, cloud) in zip(range(4), colors, quartet.items()):
        cloud_ax, srank_ax = axes[0, column], axes[1, column]

        # Top row: the point cloud, the diagonal, and where each point projects to.
        projected = diagonal_projection(cloud)
        r = np.corrcoef(cloud[:, 0], cloud[:, 1])[0, 1]
        cloud_ax.axline((0, 0), slope=1, color=grid_color, linestyle="--",
                        linewidth=0.8)
        for p, q in zip(cloud, projected):
            cloud_ax.plot([p[0], q[0]], [p[1], q[1]], color=grid_color,
                          linewidth=0.6, alpha=0.6)
        cloud_ax.scatter(cloud[:, 0], cloud[:, 1], s=22, color=color, zorder=2)
        cloud_ax.scatter(projected[:, 0], projected[:, 1], s=28, color=proj_color,
                         marker="x", linewidth=1.4, zorder=3)
        cloud_ax.set_title(f"{name}\n$r$ = {r:+.2f}", fontsize=10)
        # A common square window, so the diagonal runs corner to corner and the
        # four columns are directly comparable.
        cloud_ax.set_xlim(2.5, 20)
        cloud_ax.set_ylim(2.5, 20)
        cloud_ax.set_aspect("equal")
        cloud_ax.set_xticks([])
        cloud_ax.set_yticks([])

        # Bottom row: the kernel's stable rank, with its integral annotated.
        sranks = kernel_stable_rank(cloud)
        plotpcf(sranks[0], ax=srank_ax, color=color, linewidth=2)
        # The score annotates the curve it is the area of, rather than sitting
        # in a title next to Pearson's r as if it were a rival scalar.
        srank_ax.text(0.95, 0.93, f"score = {kernel_score(sranks):.3f}",
                      transform=srank_ax.transAxes, ha="right", va="top",
                      fontsize=9)
        # Shared limits: the curves are only comparable on common axes.
        srank_ax.set_xlim(0, 2.05)
        srank_ax.set_ylim(-0.4, 10.6)
        srank_ax.set_xlabel("bar length $t$")
        if column == 0:
            srank_ax.set_ylabel("stable rank")
        else:
            srank_ax.set_yticklabels([])

    legend_handles = [
        Line2D([], [], linestyle="none", marker="o", color=grid_color,
               label="original points (one color per dataset)"),
        Line2D([], [], linestyle="none", marker="x", color=proj_color,
               markeredgewidth=1.4, label="projection onto the diagonal"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
# -- docs snippet end hkernel_quartet --


# -- docs snippet start hkernel_invariants --
def plot_invariants(colors=("#2a78d6", "#eb6834", "#1baf7a", "#eda100")):
    quartet = anscombes_quartet()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for color, (name, cloud) in zip(colors, quartet.items()):
        # reduced=True drops the essential H0 bar, which never dies and would
        # give every dataset an infinite stable rank integral.
        bcs = persistence.compute_persistent_homology(cloud, max_dim=1, reduced=True)
        kernel = persistence.compute_homological_kernel(cloud,
                                                        diagonal_projection(cloud))

        for ax, bc in zip(axes, [bcs[0], bcs[1], kernel[0]]):
            srank = persistence.barcode_to_stable_rank(bc)
            plotpcf(srank, ax=ax, color=color, linewidth=2, label=name)

    for ax, title in zip(axes, ["$H_0$ of the point cloud", "$H_1$ of the point cloud",
                                "homological kernel"]):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t")
        # Every panel carries its own legend: the H1 panel draws only dataset I
        # (the others have empty barcodes), so it cannot be read off a shared one.
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("stable rank")
    fig.tight_layout()
    return fig
# -- docs snippet end hkernel_invariants --


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


LIGHT = ("default", "white", "black")
DARK = ("dark_background", "#1a1a2e", "#e0e0e0")

if __name__ == "__main__":
    figures = [
        ("hkernel_projection",
         plot_projection,
         lambda: plot_projection("#3987e5", "#d95926", "#d0d0d0", "0.45")),
        ("hkernel_merging",
         plot_merging,
         lambda: plot_merging(("#3987e5", "#d95926", "#199e70", "#c98500"),
                              "0.45")),
        ("hkernel_quartet",
         plot_quartet,
         lambda: plot_quartet(("#3987e5", "#d95926", "#199e70", "#c98500"),
                              "#d0d0d0", "0.45")),
        ("hkernel_invariants",
         plot_invariants,
         lambda: plot_invariants(("#3987e5", "#d95926", "#199e70", "#c98500"))),
    ]
    for name, func_light, func_dark in figures:
        _save_themed(func_light, *LIGHT, HERE / f"{name}_light.png")
        _save_themed(func_dark, *DARK, HERE / f"{name}_dark.png")
