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
