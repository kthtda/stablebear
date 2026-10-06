"""Plot distribution weights on a two-dimensional grid."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# -- docs snippet start weight_heatmaps --
import matplotlib.pyplot as plt
import stablebear as sb
from stablebear.plotting import plot_distance_weight_heatmap


def plot_weight_heatmaps():
    gaussian = sb.distributions.Gaussian(std=0.5)
    uniform = sb.distributions.Uniform(0.8, 1.2)
    mixture = sb.distributions.Mixture([gaussian, uniform], [0.8, 0.2])
    extent = (-1.65, 1.65, -1.65, 1.65)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), layout="constrained")
    for ax, distribution, title in zip(
        axes, [gaussian, uniform, mixture], ["Gaussian", "Uniform", "Mixture"]
    ):
        heatmap = plot_distance_weight_heatmap(distribution, extent=extent, ax=ax)
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(heatmap, ax=ax, label="Weight", shrink=0.75)
    return fig
# -- docs snippet end weight_heatmaps --


if __name__ == "__main__":
    for theme, style in [("light", "default"), ("dark", "dark_background")]:
        with plt.style.context(style):
            fig = plot_weight_heatmaps()
            output = Path(__file__).with_name(f"weight_heatmaps_{theme}.png")
            fig.savefig(output, dpi=180, bbox_inches="tight", transparent=True)
            plt.close(fig)
            print(f"Saved {output}")
