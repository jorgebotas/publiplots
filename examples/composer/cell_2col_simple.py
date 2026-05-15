"""Cell 2-column figure with two axes panels — flex sizing, auto abc.

Demonstrates:
- Cell-2col preset (174mm wide, abc='upper', 9pt bold labels)
- Flex sizing: panel B fills the leftover width so the figure is
  exactly 174mm wide
- Auto abc labels (panels labeled 'A' and 'B' from the canvas's abc='upper')
- Saves to docs/images/composer/cell-2col-simple.png

Run from the repo root:

    uv run python examples/composer/cell_2col_simple.py
"""

from pathlib import Path

import numpy as np
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(0)
    df_a = {"x": rng.normal(0, 1, 200), "y": rng.normal(0, 1, 200)}
    df_b = {"x": np.arange(50), "y": np.cumsum(rng.normal(0, 1, 50))}

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label=None, size=(70, 50)),       # A — pinned 70mm
        pp.PanelAxes(label=None, size=("flex", 50)),   # B — fills leftover
    )

    canvas["A"].ax.scatter(df_a["x"], df_a["y"], s=4, alpha=0.6)
    canvas["A"].ax.set_xlabel("measurement A")
    canvas["A"].ax.set_ylabel("measurement B")

    canvas["B"].ax.plot(df_b["x"], df_b["y"], lw=1.0)
    canvas["B"].ax.set_xlabel("time (frames)")
    canvas["B"].ax.set_ylabel("cumulative drift")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "composer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cell-2col-simple.png"
    canvas.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
