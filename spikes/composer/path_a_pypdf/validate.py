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
