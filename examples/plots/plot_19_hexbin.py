"""
Hexbin Plot Examples
====================

This example demonstrates :func:`publiplots.hexbinplot`, a bivariate 2D
density plot that aggregates ``(x, y)`` points into a hexagonal grid.
Each panel exercises a different facet of the API: plain count density,
per-hex reduction of a third column, log-scaled binning for heavy-tailed
data, and a custom colormap routed through the shared legend reactor.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(42)

    # --- Data -----------------------------------------------------------------
    # Panel 1/4 data: mixture of two 2D Gaussians with overlapping tails.
    n = 10_000
    cluster_a = rng.multivariate_normal([-1.5, -1.0], [[1.0, 0.4], [0.4, 1.0]], n // 2)
    cluster_b = rng.multivariate_normal([2.0, 1.5], [[1.2, -0.3], [-0.3, 0.8]], n // 2)
    mixture = pd.DataFrame(
        np.vstack([cluster_a, cluster_b]),
        columns=["x", "y"],
    )
    # Panel 2 data: reuse the mixture, add a score that trends with x.
    scored = mixture.copy()
    scored["score"] = scored["x"] + rng.normal(scale=0.3, size=len(scored))

    # Panel 3 data: heavy-tailed lognormal distribution on both axes.
    heavy = pd.DataFrame({
        "x": rng.lognormal(mean=0.0, sigma=1.0, size=n),
        "y": rng.lognormal(mean=0.0, sigma=1.0, size=n),
    })

    # --- Figure ---------------------------------------------------------------
    fig, axes = pp.subplots(2, 2, axes_size=(55, 45))

    # Panel 1: count density.
    pp.hexbinplot(
        data=mixture, x="x", y="y",
        title="count density",
        xlabel="x", ylabel="y",
        ax=axes[0, 0],
    )

    # Panel 2: mean of a third variable per hex.
    pp.hexbinplot(
        data=scored, x="x", y="y",
        C="score", reduce_C_function=np.mean,
        title="mean(score) per hex",
        xlabel="x", ylabel="y",
        ax=axes[0, 1],
    )

    # Panel 3: log-scaled density for heavy-tailed data.
    pp.hexbinplot(
        data=heavy, x="x", y="y",
        bins="log",
        title="log density",
        xlabel="x (lognormal)", ylabel="y (lognormal)",
        ax=axes[1, 0],
    )

    # Panel 4: custom cmap — demonstrate that hexbin colorbars integrate
    # with the shared legend reactor. Scoping ``pp.legend(axes[1, 1], ...)``
    # to this panel means the band claims only panel 4's stashed entry; the
    # other three panels continue to auto-render their own per-axes colorbars.
    pp.legend(axes[1, 1], side="right")
    pp.hexbinplot(
        data=mixture, x="x", y="y",
        cmap="magma_r",
        title="custom cmap",
        xlabel="x", ylabel="y",
        ax=axes[1, 1],
    )

    pp.savefig("plot_19_hexbin.pdf")
    pp.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
