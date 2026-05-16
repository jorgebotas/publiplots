"""Cell 2-col figure: PanelAxes scatter + embed_figure of a side lineplot.

The kitchen-sink PR 6b use case: stage a Canvas with one PanelAxes
slot and one EMPTY PanelImage slot (no path), build a separate matplotlib
figure with ``pp.subplots``, and ``canvas.embed_figure(label, fig)``
attaches it. Both panels are vector-preserved on PDF + SVG output;
the embedded figure renders to a deterministic byte buffer at compose
time.

Demonstrates the use case where the side figure is built by an
independent plotting helper that returns a Figure but doesn't know
about the canvas — the user wires them together post-hoc.
"""
from pathlib import Path

import numpy as np
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(0)

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),  # no path → unfilled slot
    )

    # Panel A: scatter on the canvas.
    canvas["A"].ax.scatter(rng.normal(size=200), rng.normal(size=200),
                           s=4, alpha=0.6)
    canvas["A"].ax.set_xlabel("x")
    canvas["A"].ax.set_ylabel("y")

    # Build the side figure independently — could come from any helper
    # that returns a matplotlib Figure. Match the axes_size to Panel B's
    # slot size (70 x 50 mm) so anchor='axes' applies a 1:1 transform —
    # the side figure's xlabel/ylabel/title land exactly in the canvas's
    # surrounding margin allocation.
    side_fig, side_ax = pp.subplots(axes_size=(70, 50))
    x = np.linspace(0, 4 * np.pi, 200)
    side_ax.plot(x, np.sin(x), label=r"$\sin(x)$")
    side_ax.plot(x, np.cos(x), label=r"$\cos(x)$")
    side_ax.set_xlabel("x")
    side_ax.set_ylabel("y")
    side_ax.legend(loc="lower left")

    # Wire the side figure into Panel B with anchor='axes' so the side
    # figure's axes-data box aligns vertically + horizontally with
    # Panel A's matplotlib axes-data box (paper-figure axis alignment).
    canvas.embed_figure("B", side_fig, anchor="axes")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "composer"
    out_dir.mkdir(parents=True, exist_ok=True)
    # save_multiple sugar: write PDF + SVG in one call.
    paths = canvas.save_multiple(
        out_dir / "cell-2col-with-embed",
        formats=["pdf", "svg"],
    )
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
