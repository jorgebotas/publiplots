"""Nature 2-col figure — PanelGrid + PanelText, mathtext caption.

Demonstrates:
- nature-2col preset (183mm wide, abc='lower')
- PanelGrid: a 1×3 sub-grid of small lineplots inside one panel
- PanelText with mathtext caption
- Saves to docs/images/composer/nature-2col-panel-grid.png
"""

from pathlib import Path

import numpy as np
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(2)
    canvas = pp.Canvas("nature-2col")

    # Width budget: 183mm canvas. Panel 'a' = 60mm; PanelGrid cells = 28mm
    # gives 28*3 + 2*2 = 88mm grid width. Decorations: 2 ylabels (~10mm
    # each), 2 right-pads (~2mm each), 2 outer pads (~2mm each), 1 wspace
    # (~3mm) ~= 31mm. Total: 60 + 88 + 31 = 179mm. Fits.
    canvas.add_row(
        pp.PanelAxes(label="a", size=(60, 40)),
        pp.PanelGrid(label="b", shape=(1, 3), axes_size=(28, 40)),
    )
    canvas.add_row(
        pp.PanelText(label="c",
                      text=r"Linear regression on 100 samples ($\alpha = 0.05$)",
                      size=("flex", 8)),
    )

    # Plot in panel 'a'
    x = rng.normal(0, 1, 100)
    y = 0.5 * x + rng.normal(0, 0.5, 100)
    canvas["a"].ax.scatter(x, y, s=4, alpha=0.6)
    canvas["a"].ax.set_xlabel("x"); canvas["a"].ax.set_ylabel("y")

    # Plot in panel 'b' grid (3 small lineplots)
    for i, ax in enumerate(canvas["b"].axes.flat):
        t = np.linspace(0, 10, 50)
        ax.plot(t, np.sin(t * (i + 1)) + rng.normal(0, 0.1, 50), lw=0.8)
        ax.set_xlabel("t"); ax.set_ylabel(f"y{i+1}")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "composer"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.savefig(out_dir / "nature-2col-panel-grid.png")
    print(f"wrote {out_dir / 'nature-2col-panel-grid.png'}")


if __name__ == "__main__":
    main()
