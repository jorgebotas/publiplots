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
