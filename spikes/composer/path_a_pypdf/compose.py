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
