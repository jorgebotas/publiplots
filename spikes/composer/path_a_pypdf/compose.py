"""Path A: stamp an external PDF schematic into a reserved slot using pypdf.

Reads:  spikes/composer/fixtures/canvas_with_slot.pdf
        spikes/composer/fixtures/schematic_simple.svg
        spikes/composer/fixtures/schematic_complex.svg
        (both rendered to PDF via cairosvg because pypdf can only stamp PDFs;
        PDF-source schematics need only pypdf.)
        spikes/composer/fixtures/canvas_with_slot.geom.json

Writes: spikes/composer/outputs/path_a_composed.pdf
        spikes/composer/outputs/path_a_composed_complex.pdf
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
        NameObject("/Producer"): "publiplots-composer-spike",
        NameObject("/CreationDate"): "D:20260514000000Z",
    })
    out_path = OUT / out_name
    with open(out_path, "wb") as f:
        writer.write(f)
    print(f"wrote {out_path}")


def main() -> None:
    compose_one("schematic_simple.svg", "path_a_composed.pdf")
    compose_one("schematic_complex.svg", "path_a_composed_complex.pdf")


if __name__ == "__main__":
    main()
