"""Produce a publiplots-style canvas with a reserved schematic slot.

Outputs: canvas_with_slot.pdf, canvas_with_slot.svg in spikes/composer/fixtures/.
The canvas is 174 mm x 80 mm (Cell 2-col single row) with two panels:
  - LEFT  (5..85 mm wide, 5..75 mm tall): RESERVED SLOT (axes off, no patch)
  - RIGHT (94..169 mm wide, 5..75 mm tall): a real publiplots scatter

We construct the figure with bare matplotlib (NOT pp.subplots) so the
publiplots SubplotsAutoLayout reactor doesn't resize the figure to fit
decorations after we set its mm dimensions. Plot styling still inherits
publiplots' rcParams because `import publiplots as pp` runs init_rcparams().

The reserved slot's mm-rect is recorded as JSON next to the output files so
the compositors can read it back without re-deriving the geometry.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import publiplots as pp  # noqa: F401  -- side effect: init_rcparams

OUT_DIR = Path(__file__).resolve().parent
CANVAS_W_MM, CANVAS_H_MM = 174.0, 80.0
SLOT_X_MM, SLOT_Y_MM, SLOT_W_MM, SLOT_H_MM = 5.0, 5.0, 80.0, 70.0
RIGHT_X_MM, RIGHT_Y_MM, RIGHT_W_MM, RIGHT_H_MM = 94.0, 5.0, 75.0, 70.0

MM2INCH = 1.0 / 25.4


def main() -> None:
    # Bare-matplotlib figure with mm-precise dims; no auto-layout reactor.
    fig = plt.figure(figsize=(CANVAS_W_MM * MM2INCH, CANVAS_H_MM * MM2INCH))

    # The "real" right axes.
    ax_right = fig.add_axes((
        RIGHT_X_MM / CANVAS_W_MM,
        RIGHT_Y_MM / CANVAS_H_MM,
        RIGHT_W_MM / CANVAS_W_MM,
        RIGHT_H_MM / CANVAS_H_MM,
    ))
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = 0.7 * x + rng.normal(0, 0.5, 200)
    ax_right.scatter(x, y, s=8, alpha=0.6, edgecolor="black", linewidth=0.4)
    ax_right.set_xlabel("x")
    ax_right.set_ylabel("y")
    ax_right.set_title("real publiplots panel")

    # The reserved slot: empty axes, no spines, no ticks.
    ax_slot = fig.add_axes((
        SLOT_X_MM / CANVAS_W_MM,
        SLOT_Y_MM / CANVAS_H_MM,
        SLOT_W_MM / CANVAS_W_MM,
        SLOT_H_MM / CANVAS_H_MM,
    ))
    ax_slot.set_axis_off()
    ax_slot.patch.set_visible(False)

    pdf_path = OUT_DIR / "canvas_with_slot.pdf"
    svg_path = OUT_DIR / "canvas_with_slot.svg"
    # Determinism knobs:
    #   - metadata={...: None} suppresses the wall-clock CreationDate (PDF) /
    #     dc:date (SVG) keys.
    #   - svg.hashsalt fixes the per-session random salt matplotlib uses for
    #     <defs> path / clip-path IDs (default: None → random per process).
    # Both are required for byte-stable output across re-runs; otherwise
    # `git diff` flares after every regen.
    plt.rcParams["svg.hashsalt"] = "publiplots-composer-spike"
    fig.savefig(pdf_path, transparent=True, metadata={"CreationDate": None})
    fig.savefig(svg_path, transparent=True, metadata={"Date": None})
    plt.close(fig)

    geom_path = OUT_DIR / "canvas_with_slot.geom.json"
    geom_path.write_text(json.dumps({
        "canvas_mm": [CANVAS_W_MM, CANVAS_H_MM],
        "slot_mm":   [SLOT_X_MM, SLOT_Y_MM, SLOT_W_MM, SLOT_H_MM],
        "right_panel_mm": [RIGHT_X_MM, RIGHT_Y_MM, RIGHT_W_MM, RIGHT_H_MM],
    }, indent=2))
    print(f"wrote {pdf_path}")
    print(f"wrote {svg_path}")
    print(f"wrote {geom_path}")


if __name__ == "__main__":
    main()
