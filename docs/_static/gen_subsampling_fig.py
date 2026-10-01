"""Generate point-cloud and distance-matrix subsampling figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# -- docs snippet start point_cloud_subsamples --
import matplotlib.pyplot as plt
import numpy as np
import stablebear as sb


def plot_point_cloud_subsamples(point_color="#94a3b8", selected_color="#d55e00"):
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    cloud = sb.PointCloud(points)
    samples = sb.random.subsample(
        cloud, n_points=10, n_samples=3,
        generator=sb.random.Generator(seed=5),
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        selected = np.asarray(samples[i])
        ax.scatter(points[:, 0], points[:, 1], s=28, color=point_color,
                   label="Original point cloud (40 points)", zorder=2)
        ax.scatter(selected[:, 0], selected[:, 1], s=75, color=selected_color,
                   edgecolors=ax.get_facecolor(), linewidths=1.2,
                   label="Selected points (10)", zorder=3)
        ax.set(title=f"Sample {i + 1}", xlabel="x", aspect="equal",
               xlim=(-1.15, 1.15), ylim=(-1.15, 1.15),
               xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        ax.grid(alpha=0.15)

    axes[0].set_ylabel("y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig
# -- docs snippet end point_cloud_subsamples --


# -- docs snippet start distance_matrix_subsamples --
def plot_distance_matrix_subsamples(point_color="#94a3b8", selected_color="#d55e00"):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.distance import pdist
    import stablebear as sb

    points = np.array([
        [0., 0.],
        [2., 0.],
        [4., 0.],
        [1., 1.],
        [3., 1.],
        [0., 2.],
        [4., 2.],
        [1., 3.],
        [3., 3.],
        [2., 4.],
    ])
    # DistanceMatrix accepts SciPy's condensed pairwise distances directly.
    matrix = sb.DistanceMatrix(pdist(points))
    samples = sb.random.subsample(
        matrix, n_points=4, n_samples=3,
        generator=sb.random.Generator(seed=5),
    )
    indices = samples.indices

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        # Each matrix vertex index is a row of the original point array.
        selected_points = points[indices[i]]
        ax.scatter(points[:, 0], points[:, 1], s=28, color=point_color,
                   label="Original points (10)", zorder=2)
        ax.scatter(selected_points[:, 0], selected_points[:, 1], s=75,
                   color=selected_color, edgecolors=ax.get_facecolor(),
                   linewidths=1.2, label="Selected via indices (4)", zorder=3)
        ax.set(title=f"Sample {i + 1}", xlabel="x", aspect="equal",
               xlim=(-0.5, 4.5), ylim=(-0.5, 4.5),
               xticks=[0, 2, 4], yticks=[0, 2, 4])
        ax.grid(alpha=0.15)

    axes[0].set_ylabel("y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig
# -- docs snippet end distance_matrix_subsamples --


if __name__ == "__main__":
    here = Path(__file__).parent
    for theme, style, background, points, selected in [
        ("light", "default", "white", "#94a3b8", "#d55e00"),
        ("dark", "dark_background", "#1a1a2e", "#8892a5", "#ffb36b"),
    ]:
        with plt.style.context(style), plt.rc_context({
            "figure.facecolor": background,
            "axes.facecolor": background,
        }):
            for name, plot in [
                ("cloud", plot_point_cloud_subsamples),
                ("matrix", plot_distance_matrix_subsamples),
            ]:
                fig = plot(points, selected)
                output = here / f"subsampling_{name}_{theme}.png"
                fig.savefig(output, dpi=180, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {output}")
