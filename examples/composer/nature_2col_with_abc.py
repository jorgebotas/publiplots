"""Nature 2-column figure with three panels — abc lower, mixed labels.

Demonstrates:
- nature-2col preset (183mm wide, abc='lower', 8pt bold labels)
- Three flex panels splitting leftover width equally
- Mix of auto-letter (None → 'a','c') and verbatim ('b.i') labels
- Per-panel label_style override on one panel
- Saves to docs/images/composer/nature-2col-abc.png

Run from the repo root:

    uv run python examples/composer/nature_2col_with_abc.py
"""

from pathlib import Path

import numpy as np
import publiplots as pp


def main() -> None:
    rng = np.random.default_rng(1)
    canvas = pp.Canvas("nature-2col")  # abc='lower' by default

    canvas.add_row(
        pp.PanelAxes(label=None,    size=("flex", 40)),                # → 'a'
        pp.PanelAxes(label="b.i",   size=("flex", 40),
                     label_style={"size": 10}),                        # custom label + size
        pp.PanelAxes(label=None,    size=("flex", 40)),                # → 'c' (b consumed)
    )

    for i, key in enumerate(["a", "b.i", "c"]):
        ax = canvas[key].ax
        x = rng.normal(0, 1, 100)
        y = (i + 1) * 0.3 * x + rng.normal(0, 0.5, 100)
        ax.scatter(x, y, s=4, alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel(f"y{i+1}")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "images" / "composer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "nature-2col-abc.png"
    canvas.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
