"""Plot the densities from the distribution construction examples."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# -- docs snippet start distribution_densities --
import matplotlib.pyplot as plt
import numpy as np
import stablebear as sb


def plot_distribution_densities():
    gaussian = sb.distributions.Gaussian(2.0, 0.3)
    uniform = sb.distributions.Uniform(start=1.0, end=2.0)
    mixture = sb.distributions.Mixture(
        [gaussian, uniform], coefficients=[0.25, 0.75]
    )
    x = np.linspace(0, 3.2, 1024)

    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    for distribution, label, style in zip(
        [gaussian, uniform, mixture],
        ["Gaussian", "Uniform", "Mixture"],
        ["--", ":", "-"],
    ):
        ax.plot(
            x, distribution.weight(x), linestyle=style,
            color="0.5" if label != "Mixture" else "#009e73",
            linewidth=2.5, label=label,
        )
    ax.set(xlabel="x", ylabel="Probability density", xlim=(0, 3.2))
    ax.set_ylim(bottom=0)
    ax.legend()
    return fig
# -- docs snippet end distribution_densities --


if __name__ == "__main__":
    for theme, style in [("light", "default"), ("dark", "dark_background")]:
        with plt.style.context(style):
            fig = plot_distribution_densities()
            output = Path(__file__).with_name(f"distribution_densities_{theme}.png")
            fig.savefig(output, dpi=180, bbox_inches="tight", transparent=True)
            plt.close(fig)
            print(f"Saved {output}")
