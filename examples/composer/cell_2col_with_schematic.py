"""Cell 2-col figure: PanelImage SVG schematic + PanelAxes scatter → vector PDF.

PR 5's headline use case. Saves to BOTH PDF (vector-preserving) and
PNG (for at-a-glance viewing).
"""
from pathlib import Path

import numpy as np
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(0)
    # Use the same fixture the goldens use so the example is reproducible
    # without bringing extra schematics into the repo.
    fixture = (Path(__file__).resolve().parents[2]
               / "tests" / "composer" / "golden" / "fixtures" / "schematic.svg")

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=fixture, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )

    canvas["B"].ax.scatter(rng.normal(size=200), rng.normal(size=200),
                            s=4, alpha=0.6)
    canvas["B"].ax.set_xlabel("x")
    canvas["B"].ax.set_ylabel("y")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "composer"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.savefig(out_dir / "cell-2col-with-schematic.pdf")
    # Re-create a fresh canvas for PNG to avoid finalize-state issues.
    canvas2 = pp.Canvas("cell-2col")
    canvas2.add_row(
        pp.PanelImage(label="A", path=fixture, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    canvas2["B"].ax.scatter(rng.normal(size=200), rng.normal(size=200),
                             s=4, alpha=0.6)
    canvas2["B"].ax.set_xlabel("x"); canvas2["B"].ax.set_ylabel("y")
    canvas2.savefig(out_dir / "cell-2col-with-schematic.png")
    print(f"wrote {out_dir / 'cell-2col-with-schematic.pdf'}")


if __name__ == "__main__":
    main()
