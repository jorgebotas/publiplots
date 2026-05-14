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
