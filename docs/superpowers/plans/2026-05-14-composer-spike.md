# Composer Vector-Pipeline Derisking Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the three vector-compositing paths described in the Composer spec (pypdf-PDF, cairosvg-SVG, in-tree SVG composer) on real publiplots-style inputs BEFORE PR 1, so the architecture is locked in with evidence rather than hope.

**Architecture:** A separate `feat/composer-spike` branch. Spike code lives under `spikes/composer/` (NOT `src/publiplots/composer/` — this code is throwaway, kept only as a fixture-generation reference). Each path has a runnable script that produces an artefact + a JSON report. A summary `composer-spike.md` doc captures findings, decisions, and any architectural revisions needed before PR 1.

**Tech Stack:** Python ≥ 3.10, matplotlib (existing), pypdf, cairosvg, pillow (existing), pytest. Spike-only deps installed via `uv pip install pypdf cairosvg` inside the editable publiplots venv (NOT added to `pyproject.toml` — that lands in PR 5).

**Spec reference:** `docs/superpowers/specs/2026-05-14-composer-design.md` §"Vector-preserving compositing pipeline" + §"Pre-PR-1 spike".

**Out of scope for this spike:**
- Public `pp.Canvas` API. Spike calls matplotlib + pypdf/cairosvg directly.
- Panel labels, alignment, presets. None of those affect compositing fidelity.
- Production code under `src/publiplots/`. Nothing in this spike ships in a release.

**What "success" means for this spike:** at least the **pypdf path** must work end-to-end (vectors preserved, fonts intact, transparency preserved, opens cleanly in 3 viewers). cairosvg + in-tree SVG paths are nice-to-have for the spike — if they fail, we document the failure and PR 5 falls back to a smaller scope (PDF only, with raster fallback for SVG).

---

## File Structure

```
spikes/composer/                        ← NEW (gitignored except for the scripts + report)
├── README.md                           # how to run the spike, expected outputs
├── fixtures/                           # input PDFs/SVGs/PNGs the spike consumes
│   ├── canvas_with_slot.py             # script: produces canvas_with_slot.pdf + .svg
│   ├── schematic_simple.svg            # commit a hand-rolled minimal SVG (text + shapes)
│   └── schematic_complex.svg           # copy of docs/images/claudecode-color.svg for real-world test
├── path_a_pypdf/
│   ├── compose.py                      # pypdf-based compositor
│   ├── validate.py                     # parse output PDF, assert vectors preserved
│   └── report.json                     # written by validate.py
├── path_b_cairosvg/
│   ├── compose.py                      # cairosvg(SVG→PDF) → pypdf compositor
│   ├── validate.py
│   └── report.json
├── path_c_in_tree_svg/
│   ├── compose.py                      # matplotlib SVG → manual XML insertion
│   ├── validate.py
│   └── report.json
├── outputs/                            # gitignored — generated artefacts
│   ├── path_a_composed.pdf
│   ├── path_b_composed.pdf
│   └── path_c_composed.svg
└── composer-spike.md                   # FINDINGS REPORT (committed)
```

**Gitignore:** add `spikes/composer/outputs/` and `spikes/composer/__pycache__/` to `.gitignore`. The scripts, fixtures, and report ARE committed (they're the artefacts that justify the architecture).

**Branch policy:** all spike work on `feat/composer-spike`. The branch is merged to main as a single PR titled `spike(composer): vector pipeline derisking` so the validation evidence is part of the project history. Spike scripts stay in-tree as a reference for PR 5's compositing implementation.

---

## Task 1 — Branch + workspace setup

**Files:**
- Create: `spikes/composer/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Create the spike branch**

```bash
cd /home/sagemaker-user/publiplots
git checkout -b feat/composer-spike
```

- [ ] **Step 2: Create the spike directory tree**

```bash
mkdir -p spikes/composer/{fixtures,path_a_pypdf,path_b_cairosvg,path_c_in_tree_svg,outputs}
```

- [ ] **Step 3: Add gitignore entries**

Append to `.gitignore`:

```
# Composer spike (outputs are throwaway; scripts are committed)
spikes/composer/outputs/
spikes/composer/**/__pycache__/
spikes/composer/**/*.pyc
```

- [ ] **Step 4: Write spike README**

Create `spikes/composer/README.md` with this exact content:

````markdown
# Composer Vector-Pipeline Spike

Validates three compositing paths from `docs/superpowers/specs/2026-05-14-composer-design.md`
before PR 1 of the Composer rollout.

## Paths

- **path_a_pypdf** — matplotlib PDF + external PDF schematic → composited PDF via pypdf.
- **path_b_cairosvg** — matplotlib PDF + external SVG schematic (cairosvg → PDF) → composited PDF.
- **path_c_in_tree_svg** — matplotlib SVG + external SVG schematic → composited SVG via in-tree XML insertion.

## Run

```bash
uv pip install pypdf cairosvg
cd spikes/composer
uv run python fixtures/canvas_with_slot.py
uv run python path_a_pypdf/compose.py    && uv run python path_a_pypdf/validate.py
uv run python path_b_cairosvg/compose.py && uv run python path_b_cairosvg/validate.py
uv run python path_c_in_tree_svg/compose.py && uv run python path_c_in_tree_svg/validate.py
```

Each `validate.py` writes a `report.json` next to it. See `composer-spike.md` for the
consolidated findings.
````

- [ ] **Step 5: Install spike deps**

```bash
uv pip install pypdf cairosvg
```

Expected: both install without errors. cairosvg may print a system-dep warning if libcairo2 is missing — note it; we'll record it in the report.

- [ ] **Step 6: Sanity-check imports**

```bash
uv run python -c "import pypdf; print(pypdf.__version__)"
uv run python -c "import cairosvg; print(cairosvg.__version__)"
```

Expected: pypdf prints `≥ 4.0`; cairosvg prints `≥ 2.7`. Both ≥ those versions per spec.

- [ ] **Step 7: Commit**

```bash
git add .gitignore spikes/composer/README.md
git commit -m "spike(composer): scaffold derisking branch + README"
```

---

## Task 2 — Fixture: matplotlib canvas with reserved slot (PDF + SVG)

**Files:**
- Create: `spikes/composer/fixtures/canvas_with_slot.py`
- Create: `spikes/composer/fixtures/schematic_simple.svg`
- Create: `spikes/composer/fixtures/schematic_complex.svg`
- Test (manual): produced files open cleanly

**Why this task first:** every path consumes the same canvas. Get the canvas right once.

- [ ] **Step 1: Write `canvas_with_slot.py`**

Create `spikes/composer/fixtures/canvas_with_slot.py` with this exact content:

```python
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
    fig.savefig(pdf_path, transparent=True)
    fig.savefig(svg_path, transparent=True)
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
```

- [ ] **Step 2: Write `schematic_simple.svg`**

Create `spikes/composer/fixtures/schematic_simple.svg` with this exact content:

```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" width="80mm" height="70mm" viewBox="0 0 80 70">
  <rect x="2" y="2" width="76" height="66" fill="none" stroke="black" stroke-width="0.5"/>
  <circle cx="20" cy="20" r="10" fill="#a8e6cf" stroke="black" stroke-width="0.4"/>
  <circle cx="60" cy="20" r="10" fill="#ffd3b6" stroke="black" stroke-width="0.4"/>
  <line x1="30" y1="20" x2="50" y2="20" stroke="black" stroke-width="0.4"/>
  <text x="40" y="50" font-family="Arial" font-size="6" text-anchor="middle">test schematic</text>
  <text x="40" y="58" font-family="Arial" font-size="4" text-anchor="middle">vectors + arial text</text>
</svg>
```

- [ ] **Step 3: Copy a real-world SVG fixture**

```bash
cp /home/sagemaker-user/publiplots/docs/images/claudecode-color.svg \
   /home/sagemaker-user/publiplots/spikes/composer/fixtures/schematic_complex.svg
```

Expected: file copied; ~few KB.

- [ ] **Step 4: Run the canvas script**

```bash
cd /home/sagemaker-user/publiplots
uv run python spikes/composer/fixtures/canvas_with_slot.py
```

Expected: prints three "wrote ..." lines; produces `canvas_with_slot.pdf`, `canvas_with_slot.svg`, `canvas_with_slot.geom.json` in `spikes/composer/fixtures/`.

- [ ] **Step 5: Verify slot is empty in the PDF**

```bash
uv run python -c "
import pypdf
r = pypdf.PdfReader('spikes/composer/fixtures/canvas_with_slot.pdf')
page = r.pages[0]
mb = page.mediabox
print('mediabox pt:', float(mb.width), float(mb.height))
print('mediabox mm:', float(mb.width)/72*25.4, float(mb.height)/72*25.4)
"
```

Expected: mediabox in mm is `~174.0 x ~80.0` (within 0.5 mm of target — matplotlib's PDF backend rounds to integer points).

- [ ] **Step 6: Commit**

```bash
git add spikes/composer/fixtures/
git commit -m "spike(composer): canvas-with-slot fixture + SVG schematics"
```

---

## Task 3 — Path A: pypdf-based PDF compositor

**Files:**
- Create: `spikes/composer/path_a_pypdf/compose.py`
- Create: `spikes/composer/path_a_pypdf/validate.py`

- [ ] **Step 1: Write `compose.py`**

Create `spikes/composer/path_a_pypdf/compose.py` with this exact content:

```python
"""Path A: stamp an external PDF schematic into a reserved slot using pypdf.

Reads:  spikes/composer/fixtures/canvas_with_slot.pdf
        spikes/composer/fixtures/schematic_simple.svg (rendered to PDF via cairosvg
        because pypdf can only stamp PDFs, not SVGs — so this path needs cairosvg
        too for SVG inputs. PDF-source schematics need only pypdf.)
        spikes/composer/fixtures/canvas_with_slot.geom.json

Writes: spikes/composer/outputs/path_a_composed.pdf
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import cairosvg
import pypdf
from pypdf.generic import NameObject, RectangleObject

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MM2PT = 72.0 / 25.4


def main() -> None:
    geom = json.loads((FIX / "canvas_with_slot.geom.json").read_text())
    canvas_w_mm, canvas_h_mm = geom["canvas_mm"]
    slot_x_mm, slot_y_mm, slot_w_mm, slot_h_mm = geom["slot_mm"]

    # Render the SVG schematic to an in-memory single-page PDF.
    svg_bytes = (FIX / "schematic_simple.svg").read_bytes()
    schematic_pdf_bytes = cairosvg.svg2pdf(bytestring=svg_bytes)
    schematic_reader = pypdf.PdfReader(io.BytesIO(schematic_pdf_bytes))
    schematic_page = schematic_reader.pages[0]

    # Open the canvas; we'll mutate its single page in-place.
    canvas_reader = pypdf.PdfReader(FIX / "canvas_with_slot.pdf")
    canvas_page = canvas_reader.pages[0]

    # Compute mm -> pt transform for the schematic page so it fits the slot.
    sch_mb = schematic_page.mediabox
    sch_w_pt = float(sch_mb.width)
    sch_h_pt = float(sch_mb.height)
    target_w_pt = slot_w_mm * MM2PT
    target_h_pt = slot_h_mm * MM2PT
    sx = target_w_pt / sch_w_pt
    sy = target_h_pt / sch_h_pt
    # Slot Y is in *mm from the canvas bottom*. PDF coords also have origin
    # bottom-left, so:
    tx_pt = slot_x_mm * MM2PT
    ty_pt = slot_y_mm * MM2PT

    # Use the published merge_transformed_page API (pypdf >= 4).
    transformation = pypdf.Transformation().scale(sx=sx, sy=sy).translate(tx=tx_pt, ty=ty_pt)
    canvas_page.merge_transformed_page(schematic_page, transformation)

    writer = pypdf.PdfWriter()
    writer.add_page(canvas_page)
    # Pin CreationDate so spike output is byte-stable across runs.
    writer.add_metadata({
        NameObject("/Producer"): "publiplots-composer-spike",
        NameObject("/CreationDate"): "D:20260514000000Z",
    })

    out_path = OUT / "path_a_composed.pdf"
    with open(out_path, "wb") as f:
        writer.write(f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run `compose.py`**

```bash
cd /home/sagemaker-user/publiplots
uv run python spikes/composer/path_a_pypdf/compose.py
```

Expected: prints `wrote .../outputs/path_a_composed.pdf`. No exceptions. File exists and is non-empty.

- [ ] **Step 3: Write `validate.py`**

Create `spikes/composer/path_a_pypdf/validate.py` with this exact content:

```python
"""Validate path A's output: vectors preserved, mediabox correct, schematic present."""

from __future__ import annotations

import json
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIX = ROOT / "fixtures"

MM_TOL = 0.5  # mm tolerance for mediabox dims (matplotlib rounds to integer pt)


def main() -> None:
    geom = json.loads((FIX / "canvas_with_slot.geom.json").read_text())
    expected_w_mm, expected_h_mm = geom["canvas_mm"]

    composed = pypdf.PdfReader(OUT / "path_a_composed.pdf")
    page = composed.pages[0]

    actual_w_mm = float(page.mediabox.width) / 72.0 * 25.4
    actual_h_mm = float(page.mediabox.height) / 72.0 * 25.4

    report = {
        "path": "A_pypdf",
        "expected_mediabox_mm": [expected_w_mm, expected_h_mm],
        "actual_mediabox_mm": [actual_w_mm, actual_h_mm],
        "mediabox_within_tolerance": (
            abs(actual_w_mm - expected_w_mm) <= MM_TOL
            and abs(actual_h_mm - expected_h_mm) <= MM_TOL
        ),
        "n_pages": len(composed.pages),
        "has_xobjects": False,
        "n_xobjects": 0,
        "manual_checks_required": [
            "Open path_a_composed.pdf in Acrobat; click on the schematic shapes — they must be selectable as vector objects.",
            "Open path_a_composed.pdf in Inkscape; ungroup; circles + text must be individually selectable.",
            "Open path_a_composed.pdf in Preview.app (macOS); zoom 800% — text + shapes stay crisp.",
        ],
    }

    # Check the page has XObject resources (the stamped schematic becomes one).
    resources = page.get("/Resources", {})
    if resources:
        xobjects = resources.get("/XObject", {})
        report["has_xobjects"] = bool(xobjects)
        report["n_xobjects"] = len(xobjects) if xobjects else 0

    out = ROOT / "path_a_pypdf" / "report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run `validate.py`**

```bash
cd /home/sagemaker-user/publiplots
uv run python spikes/composer/path_a_pypdf/validate.py
```

Expected: prints a JSON report with `mediabox_within_tolerance: true` and `has_xobjects: true` (the stamped schematic creates an XObject). `n_xobjects` should be ≥ 1.

- [ ] **Step 5: Manual viewer check**

Open `spikes/composer/outputs/path_a_composed.pdf` in at least 2 of: Acrobat, Inkscape, Preview.app, Firefox, Chrome. Verify:
1. The schematic appears in the LEFT half of the canvas (not the right).
2. The schematic's text "test schematic" is selectable as text (proves vectors preserved).
3. The right-hand scatter is unchanged (no overlap, no clipping).
4. No artefacts (overlapping fonts, clipped strokes, mis-rotated content).

If any of these fail, document the failure in `composer-spike.md` (see Task 6) and DO NOT proceed to declare path A successful.

- [ ] **Step 6: Repeat with the complex schematic**

Edit `compose.py` line that reads `schematic_simple.svg` to `schematic_complex.svg`, re-run `compose.py` and `validate.py`, and re-do the manual viewer check. Then revert.

Record any differences in behavior in your notes for Task 6.

- [ ] **Step 7: Commit**

```bash
git add spikes/composer/path_a_pypdf/
git commit -m "spike(composer): path A — pypdf PDF compositor + validation"
```

---

## Task 4 — Path B: cairosvg-based SVG schematic compositing

**Files:**
- Create: `spikes/composer/path_b_cairosvg/compose.py`
- Create: `spikes/composer/path_b_cairosvg/validate.py`

**Note:** Path B is *almost* identical to Path A (since A already uses cairosvg internally for its SVG input), but here we explicitly cover **both** schematic kinds (simple + complex) with **font-handling validation** — the place cairosvg historically struggles.

- [ ] **Step 1: Write `compose.py`**

Create `spikes/composer/path_b_cairosvg/compose.py` with this exact content:

```python
"""Path B: validate cairosvg's SVG-to-PDF conversion preserves fonts + transparency.

Composes BOTH schematic_simple.svg and schematic_complex.svg into separate output
PDFs so we can compare cairosvg's behavior across SVG complexity classes.

Writes: spikes/composer/outputs/path_b_simple.pdf
        spikes/composer/outputs/path_b_complex.pdf
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import cairosvg
import pypdf
from pypdf.generic import NameObject

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MM2PT = 72.0 / 25.4


def compose_one(svg_name: str, out_name: str) -> None:
    geom = json.loads((FIX / "canvas_with_slot.geom.json").read_text())
    slot_x_mm, slot_y_mm, slot_w_mm, slot_h_mm = geom["slot_mm"]

    svg_bytes = (FIX / svg_name).read_bytes()
    schematic_pdf_bytes = cairosvg.svg2pdf(bytestring=svg_bytes)
    schematic_reader = pypdf.PdfReader(io.BytesIO(schematic_pdf_bytes))
    schematic_page = schematic_reader.pages[0]

    canvas_reader = pypdf.PdfReader(FIX / "canvas_with_slot.pdf")
    canvas_page = canvas_reader.pages[0]

    sch_mb = schematic_page.mediabox
    sx = (slot_w_mm * MM2PT) / float(sch_mb.width)
    sy = (slot_h_mm * MM2PT) / float(sch_mb.height)
    tx_pt = slot_x_mm * MM2PT
    ty_pt = slot_y_mm * MM2PT

    transformation = pypdf.Transformation().scale(sx=sx, sy=sy).translate(tx=tx_pt, ty=ty_pt)
    canvas_page.merge_transformed_page(schematic_page, transformation)

    writer = pypdf.PdfWriter()
    writer.add_page(canvas_page)
    writer.add_metadata({
        NameObject("/Producer"): "publiplots-composer-spike-pathB",
        NameObject("/CreationDate"): "D:20260514000000Z",
    })
    out_path = OUT / out_name
    with open(out_path, "wb") as f:
        writer.write(f)
    print(f"wrote {out_path}")


def main() -> None:
    compose_one("schematic_simple.svg", "path_b_simple.pdf")
    compose_one("schematic_complex.svg", "path_b_complex.pdf")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run `compose.py`**

```bash
cd /home/sagemaker-user/publiplots
uv run python spikes/composer/path_b_cairosvg/compose.py
```

Expected: prints two `wrote ...` lines. If cairosvg raises on `schematic_complex.svg` (it can fail on rare SVG features), capture the exception verbatim and skip to Step 5 of this task — that failure mode IS a finding.

- [ ] **Step 3: Write `validate.py`**

Create `spikes/composer/path_b_cairosvg/validate.py` with this exact content:

```python
"""Validate path B: text in the cairosvg-converted schematic survives as text.

cairosvg has historically had two failure modes:
  1. SVG <text> elements get rasterized to paths (text no longer searchable).
  2. SVG <text> elements use embedded fonts that aren't in the host system,
     producing missing-glyph rectangles.

We assert text-as-text by extracting page text via pypdf and looking for our
marker strings. We also note Producer + ProducerCreator metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

MARKER_SIMPLE = "test schematic"   # appears in schematic_simple.svg <text>


def inspect(pdf_path: Path) -> dict:
    r = pypdf.PdfReader(pdf_path)
    page = r.pages[0]
    extracted = page.extract_text() or ""
    return {
        "file": pdf_path.name,
        "n_pages": len(r.pages),
        "extracted_text_length": len(extracted),
        "extracted_text_preview": extracted[:500],
        "marker_simple_text_preserved": MARKER_SIMPLE in extracted,
    }


def main() -> None:
    report = {
        "path": "B_cairosvg",
        "files": [
            inspect(OUT / "path_b_simple.pdf"),
            inspect(OUT / "path_b_complex.pdf"),
        ],
        "manual_checks_required": [
            "Open both path_b_*.pdf in Acrobat. Search for 'test schematic'. Hit must succeed for path_b_simple.pdf.",
            "Inspect path_b_complex.pdf in Inkscape — note any font fallback warnings or missing-glyph indicators.",
            "Compare path_b_simple.pdf to path_a_composed.pdf side-by-side. Should be visually identical (path A already uses cairosvg internally).",
        ],
    }
    out = ROOT / "path_b_cairosvg" / "report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run `validate.py`**

```bash
cd /home/sagemaker-user/publiplots
uv run python spikes/composer/path_b_cairosvg/validate.py
```

Expected: prints a JSON report. `marker_simple_text_preserved: true` for `path_b_simple.pdf` is the load-bearing assertion — if it's false, cairosvg rasterized the text and we have a finding to record.

- [ ] **Step 5: Manual viewer check + finding capture**

For each output PDF, open in Acrobat and Inkscape and check:
1. Text in the schematic is searchable (Cmd-F finds "test schematic").
2. Strokes are not rasterized (zoom 800%, edges stay crisp).
3. For `path_b_complex.pdf`: no font fallback warnings on open.

Record observations directly in your `composer-spike.md` notes (Task 6 will consolidate them). If any fail, document the SVG feature that broke (gradient? clipPath? embedded font? `<foreignObject>`?) — that's the architectural finding PR 5 needs.

- [ ] **Step 6: Commit**

```bash
git add spikes/composer/path_b_cairosvg/
git commit -m "spike(composer): path B — cairosvg SVG-to-PDF + font validation"
```

---

## Task 5 — Path C: in-tree SVG composer

**Files:**
- Create: `spikes/composer/path_c_in_tree_svg/compose.py`
- Create: `spikes/composer/path_c_in_tree_svg/validate.py`

**Why we need this:** if the user calls `canvas.savefig('fig.svg')` we want a vector SVG output, not a rasterized one. Path C tests whether we can compose SVGs without an external tool (svgutils is out of scope per the spec).

- [ ] **Step 1: Write `compose.py`**

Create `spikes/composer/path_c_in_tree_svg/compose.py` with this exact content:

```python
"""Path C: compose two SVGs by inserting one into the other as a <g> subtree.

This is *naive* SVG composition — enough to validate feasibility, not
production-ready. PR 6 will harden it (namespace handling, viewBox math,
def/style merging).

Reads:  fixtures/canvas_with_slot.svg
        fixtures/schematic_simple.svg
        fixtures/canvas_with_slot.geom.json
Writes: outputs/path_c_composed.svg
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def parse_length_to_mm(value: str) -> float:
    """Parse a length string like '174mm' or '493pt' into mm."""
    m = re.match(r"^([\d.]+)\s*(mm|cm|in|pt|px)?$", value.strip())
    if not m:
        raise ValueError(f"cannot parse length {value!r}")
    n = float(m.group(1))
    unit = m.group(2) or "px"
    if unit == "mm":  return n
    if unit == "cm":  return n * 10.0
    if unit == "in":  return n * 25.4
    if unit == "pt":  return n / 72.0 * 25.4
    if unit == "px":  return n / 96.0 * 25.4
    raise ValueError(f"unknown unit {unit!r}")


def main() -> None:
    geom = json.loads((FIX / "canvas_with_slot.geom.json").read_text())
    slot_x_mm, slot_y_mm, slot_w_mm, slot_h_mm = geom["slot_mm"]
    canvas_w_mm, canvas_h_mm = geom["canvas_mm"]

    canvas_tree = ET.parse(FIX / "canvas_with_slot.svg")
    canvas_root = canvas_tree.getroot()
    schematic_tree = ET.parse(FIX / "schematic_simple.svg")
    schematic_root = schematic_tree.getroot()

    # Determine the schematic's natural size in mm.
    sch_w_mm = parse_length_to_mm(schematic_root.attrib.get("width", "80mm"))
    sch_h_mm = parse_length_to_mm(schematic_root.attrib.get("height", "70mm"))

    # Build a <g> wrapper that translates+scales the schematic's contents to
    # the slot. SVG y axis is inverted vs PDF: y=0 is the TOP of the canvas,
    # so we translate by canvas_h_mm - slot_y_mm - slot_h_mm.
    sx = slot_w_mm / sch_w_mm
    sy = slot_h_mm / sch_h_mm
    tx = slot_x_mm
    ty = canvas_h_mm - slot_y_mm - slot_h_mm

    g = ET.Element(f"{{{SVG_NS}}}g", {
        "transform": f"translate({tx} {ty}) scale({sx} {sy})",
    })
    # Copy each child of the schematic root into our wrapper <g>.
    # We strip the schematic's own width/height/viewBox by NOT copying the root.
    for child in list(schematic_root):
        g.append(child)

    canvas_root.append(g)

    out_path = OUT / "path_c_composed.svg"
    canvas_tree.write(out_path, xml_declaration=True, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run `compose.py`**

```bash
cd /home/sagemaker-user/publiplots
uv run python spikes/composer/path_c_in_tree_svg/compose.py
```

Expected: prints `wrote .../outputs/path_c_composed.svg`.

- [ ] **Step 3: Write `validate.py`**

Create `spikes/composer/path_c_in_tree_svg/validate.py` with this exact content:

```python
"""Validate path C: composed SVG opens, namespaces are clean, schematic shapes
appear as a single <g> subtree."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

SVG_NS = "{http://www.w3.org/2000/svg}"


def main() -> None:
    composed = OUT / "path_c_composed.svg"
    tree = ET.parse(composed)
    root = tree.getroot()

    n_groups = sum(1 for _ in root.iter(f"{SVG_NS}g"))
    n_text = sum(1 for _ in root.iter(f"{SVG_NS}text"))
    n_circle = sum(1 for _ in root.iter(f"{SVG_NS}circle"))

    report = {
        "path": "C_in_tree_svg",
        "file_size_bytes": composed.stat().st_size,
        "root_tag": root.tag,
        "n_g_elements": n_groups,
        "n_text_elements": n_text,
        "n_circle_elements": n_circle,
        "manual_checks_required": [
            "Open path_c_composed.svg in Inkscape: schematic appears in left half, scatter on right.",
            "Open path_c_composed.svg in Firefox: same.",
            "Inkscape > Edit > Find: 'test schematic' must hit a text element (vectors preserved).",
            "Save as a different filename in Inkscape; reopen — should round-trip cleanly without warnings.",
        ],
    }
    out = ROOT / "path_c_in_tree_svg" / "report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run `validate.py`**

```bash
cd /home/sagemaker-user/publiplots
uv run python spikes/composer/path_c_in_tree_svg/validate.py
```

Expected: prints a JSON report with `n_text_elements ≥ 2` (the schematic has 2 text elements) and `n_circle_elements ≥ 2`.

- [ ] **Step 5: Manual viewer check**

Open `spikes/composer/outputs/path_c_composed.svg` in:
1. Inkscape — the schematic must appear in the LEFT half. Text "test schematic" must be selectable.
2. Firefox or Chrome — same composition; no rendering errors in the dev-tools console.
3. Try opening AND re-saving in Inkscape — does the round-trip preserve structure or does Inkscape complain?

Record findings for Task 6. The most common failure is namespace pollution (multiple `xmlns` declarations) — note it explicitly if it happens.

- [ ] **Step 6: Commit**

```bash
git add spikes/composer/path_c_in_tree_svg/
git commit -m "spike(composer): path C — in-tree SVG composer + validation"
```

---

## Task 6 — Findings report + decision rubric

**Files:**
- Create: `spikes/composer/composer-spike.md`

This is the single most important deliverable of the spike. Without this, the spike was a code exercise instead of an architectural decision.

- [ ] **Step 1: Re-run all three paths' validate scripts** (capture fresh output)

```bash
cd /home/sagemaker-user/publiplots
uv run python spikes/composer/path_a_pypdf/compose.py
uv run python spikes/composer/path_a_pypdf/validate.py
uv run python spikes/composer/path_b_cairosvg/compose.py
uv run python spikes/composer/path_b_cairosvg/validate.py
uv run python spikes/composer/path_c_in_tree_svg/compose.py
uv run python spikes/composer/path_c_in_tree_svg/validate.py
```

Confirm all three `report.json` files are fresh.

- [ ] **Step 2: Write `composer-spike.md`**

Create `spikes/composer/composer-spike.md` with this exact structure (fill in actual findings — do NOT leave any section as "TBD"):

````markdown
# Composer Vector-Pipeline Spike — Findings

**Branch:** `feat/composer-spike`
**Date:** YYYY-MM-DD (the date the spike was completed)
**Spec reference:** `docs/superpowers/specs/2026-05-14-composer-design.md`

## Summary verdict

One of three outcomes. Pick exactly one and explain in 2-3 sentences:

- **GREEN** — all three paths work; PR 5 implements all three as designed.
- **YELLOW** — path A (pypdf) works; cairosvg or in-tree SVG has caveats. PR 5 ships pypdf for PDF and falls back to raster for SVG with a `UserWarning`. SVG vector path becomes a stretch goal.
- **RED** — path A fails; the architecture needs revision. List the specific failure mode and propose alternatives (e.g. pikepdf instead of pypdf, or rendering-only-to-PNG for v1).

[Write the verdict here.]

## Per-path results

### Path A — pypdf-based PDF compositor

- **Status:** PASS / PARTIAL / FAIL
- **Vectors preserved:** YES / NO (cite report.json + viewer evidence)
- **Fonts preserved:** YES / NO (text searchable in Acrobat?)
- **Transparency preserved:** YES / NO
- **Viewers tested:** [list of viewers]
- **Issues found:** [list any anomalies — overlapping content, mediabox drift, etc.]
- **Recommendation for PR 5:** [adopt as-is / adopt with caveats / replace with X]

### Path B — cairosvg SVG-to-PDF

- **Status:** PASS / PARTIAL / FAIL
- **Simple SVG schematic preserved:** YES / NO
- **Complex SVG schematic preserved:** YES / NO
- **Failure modes observed (if any):** [text rasterization? font fallback? unsupported features?]
- **Viewers tested:** [list]
- **Recommendation for PR 5:** [adopt as default SVG converter / adopt with strict_vectors fallback / use a different lib]

### Path C — in-tree SVG composer

- **Status:** PASS / PARTIAL / FAIL
- **Composition correctness:** [text/circle element counts vs expected]
- **Namespace handling:** [clean / polluted with multiple xmlns]
- **Inkscape round-trip:** [clean / warning-prone / breaks]
- **Recommendation for PR 5:** [ship as designed / harden before shipping / defer SVG output to PR 6+]

## Architectural revisions for PR 5

If any path failed, list the spec changes required:

- [revisions, or "none — spec stands as written"]

## Dependency notes

- pypdf version installed: [X.Y.Z]
- cairosvg version installed: [X.Y.Z]
- libcairo2 system version observed: [version + how detected]
- Anything brittle about the install? [user-facing hint to add to docs]

## Open questions for PR 5

- [Any question whose answer is "we'll find out when we implement PR 5". E.g.: "Does merge_transformed_page preserve PDF/A-1 compliance?" or "Should embedded raster images in source PDFs be re-encoded or pass through?"]

## Files touched in this spike

```
spikes/composer/
├── README.md
├── fixtures/
├── path_a_pypdf/
├── path_b_cairosvg/
├── path_c_in_tree_svg/
└── composer-spike.md
```

Outputs in `spikes/composer/outputs/` are gitignored and regenerable.
````

- [ ] **Step 3: Re-read the report end-to-end**

Open `composer-spike.md` and read every section. Confirm:
- No section says "TBD" or "TODO".
- Every "Status" line is filled in.
- The "Summary verdict" has a clear GREEN / YELLOW / RED.
- The "Architectural revisions" section either lists specific spec edits or says "none".

- [ ] **Step 4: Commit**

```bash
git add spikes/composer/composer-spike.md
git commit -m "spike(composer): findings report + decision rubric"
```

---

## Task 7 — Open the spike PR

**Files:** none (gh CLI only)

- [ ] **Step 1: Push the branch**

```bash
cd /home/sagemaker-user/publiplots
git push -u origin feat/composer-spike
```

- [ ] **Step 2: Confirm PR title + body in your head**

Title: `spike(composer): vector pipeline derisking`

Body template (Step 3 will use this exact text):

```markdown
## Summary

Validates the three vector-compositing paths from the Composer design spec
(`docs/superpowers/specs/2026-05-14-composer-design.md`) on real publiplots
inputs before any production code lands.

## What's in this PR

- `spikes/composer/` — scripts + fixtures + per-path reports.
- `spikes/composer/composer-spike.md` — findings report with summary verdict
  (GREEN / YELLOW / RED), per-path results, and architectural revisions for PR 5.

## Verdict

[Copy the "Summary verdict" line from composer-spike.md here.]

## Test plan

- [x] Path A (pypdf): compose.py runs, validate.py reports vectors preserved, manual viewer check passes.
- [x] Path B (cairosvg): both simple + complex SVG schematics composed; font behavior recorded.
- [x] Path C (in-tree SVG): SVG round-trips through Inkscape + Firefox.
- [x] Findings consolidated in `composer-spike.md`.

## Follow-ups

PR 1 (`feat(composer): canvas + axes panels (single-row)`) starts from the
verdict in `composer-spike.md`. If the verdict is YELLOW or RED, the design
spec gets an amendment commit before PR 1 begins.
```

- [ ] **Step 3: Open the PR**

```bash
gh pr create \
  --title "spike(composer): vector pipeline derisking" \
  --body "$(cat <<'EOF'
## Summary

Validates the three vector-compositing paths from the Composer design spec
(`docs/superpowers/specs/2026-05-14-composer-design.md`) on real publiplots
inputs before any production code lands.

## What's in this PR

- `spikes/composer/` — scripts + fixtures + per-path reports.
- `spikes/composer/composer-spike.md` — findings report with summary verdict
  (GREEN / YELLOW / RED), per-path results, and architectural revisions for PR 5.

## Verdict

See `spikes/composer/composer-spike.md` "Summary verdict" section.

## Test plan

- [x] Path A (pypdf): compose.py runs, validate.py reports vectors preserved, manual viewer check passes.
- [x] Path B (cairosvg): both simple + complex SVG schematics composed; font behavior recorded.
- [x] Path C (in-tree SVG): SVG round-trips through Inkscape + Firefox.
- [x] Findings consolidated in `composer-spike.md`.

## Follow-ups

PR 1 (`feat(composer): canvas + axes panels (single-row)`) starts from the
verdict in `composer-spike.md`. If the verdict is YELLOW or RED, the design
spec gets an amendment commit before PR 1 begins.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: gh prints the new PR URL. Capture the URL for the human.

- [ ] **Step 4: Return the PR URL to the human**

Print the PR URL so the user can review.

---

## Acceptance criteria for this spike

The spike is complete when ALL of:

1. All seven tasks above are checked off.
2. `spikes/composer/composer-spike.md` exists with no "TBD" / "TODO" sections.
3. The "Summary verdict" is one of GREEN / YELLOW / RED with explicit reasoning.
4. Per-path `report.json` files exist with concrete numeric findings.
5. Manual viewer checks were actually performed and notes captured (in the report, not in the agent's head).
6. The spike PR is open against `main`.

If the verdict is YELLOW or RED, the FIRST task of PR 1's plan is "amend the design spec based on spike findings". That amendment must land BEFORE any PR 1 code is written.
