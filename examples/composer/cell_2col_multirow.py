"""Cell 2-row figure — multi-row support, abc letters spanning rows.

Demonstrates:
- Cell-2col preset (174mm wide, abc='upper')
- Multi-row layout: row 0 has panels A+B, row 1 has panels C+D
- abc sequencing 'A','B','C','D' across rows
- canvas.align(['A', 'C'], edge='left') keeping the column visually crisp
- Saves to docs/images/composer/cell-2col-multirow.png
"""

from pathlib import Path

import numpy as np
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(0)
    canvas = pp.Canvas("cell-2col")

    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 40)),
        pp.PanelAxes(label="B", size=("flex", 40)),
    )
    canvas.add_row(
        pp.PanelAxes(label="C", size=(70, 40)),
        pp.PanelText(label="D", text="n = 1,234\nP < 0.001",
                      size=("flex", 40)),
    )

    # Align before any panel access (align must precede figure finalize).
    canvas.align(["A", "C"], edge="left")

    canvas["A"].ax.scatter(rng.normal(0, 1, 200), rng.normal(0, 1, 200), s=4, alpha=0.6)
    canvas["A"].ax.set_xlabel("x"); canvas["A"].ax.set_ylabel("y")

    t = np.linspace(0, 10, 100)
    canvas["B"].ax.plot(t, np.sin(t) + rng.normal(0, 0.1, 100))
    canvas["B"].ax.set_xlabel("time"); canvas["B"].ax.set_ylabel("signal")

    cats = ["a", "b", "c", "d"]
    canvas["C"].ax.bar(cats, rng.uniform(1, 5, 4))
    canvas["C"].ax.set_xlabel("group"); canvas["C"].ax.set_ylabel("count")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "composer"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.savefig(out_dir / "cell-2col-multirow.png")
    print(f"wrote {out_dir / 'cell-2col-multirow.png'}")


if __name__ == "__main__":
    main()
