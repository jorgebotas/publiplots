"""Demo: pp.legend(rows=, cols=) for grid-scoped figure legends.

Renders a 2×3 grid where row 0 shares one legend band on top and
row 1 shares another on bottom, demonstrating how the new grid-scope
kwargs let two independent legend groups coexist on a single figure.
"""
import numpy as np
import pandas as pd

import publiplots as pp


def main(out_path: str = "examples/legend/grid_scope_demo.png") -> None:
    rng = np.random.default_rng(0)
    fig, axes = pp.subplots(nrows=2, ncols=3, axes_size=(50, 30))

    # Pre-register two disjoint scope groups: one band per row.
    pp.legend(rows=0, side="top")
    pp.legend(rows=1, side="bottom")

    # Row 0: scatter, hue = 'cond_a' (lo/hi). Fixed palette so the
    # group merge across panels uses identical handles.
    palette_a = {"lo": "C0", "hi": "C1"}
    for ax in axes[0]:
        df = pd.DataFrame({
            "x": rng.normal(size=80),
            "y": rng.normal(size=80),
            "cond_a": rng.choice(["lo", "hi"], size=80),
        })
        pp.scatterplot(data=df, x="x", y="y", hue="cond_a",
                       palette=palette_a, ax=ax)

    # Row 1: scatter, hue = 'cond_b' (x/y). Different categorical →
    # different group, so the row-1 band shows different entries
    # from the row-0 band.
    palette_b = {"x": "C2", "y": "C3"}
    for ax in axes[1]:
        df = pd.DataFrame({
            "x": rng.normal(size=80),
            "y": rng.normal(size=80),
            "cond_b": rng.choice(["x", "y"], size=80),
        })
        pp.scatterplot(data=df, x="x", y="y", hue="cond_b",
                       palette=palette_b, ax=ax)

    fig.savefig(out_path)


if __name__ == "__main__":
    main()
