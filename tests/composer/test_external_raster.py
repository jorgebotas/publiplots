"""Tests for external_raster sidecar PNG output for SVG.

PR 6b adds ``Canvas.savefig(path, *, external_raster=True)`` which,
when the SVG composer encounters a raster source (PNG/JPG/TIFF
PanelImage), writes the source to a sidecar PNG file rather than
inlining a base64 data-URI. Avoids the SVG-bloat hazard for
high-DPI rasters.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import publiplots as pp


lxml_etree = pytest.importorskip("lxml.etree")


@pytest.fixture
def small_png(tmp_path):
    from PIL import Image
    p = tmp_path / "schematic.png"
    Image.new("RGB", (300, 200), color="white").save(p, dpi=(300, 300))
    return p


def _build_canvas_with_png(small_png):
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=small_png, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    return canvas


def test_external_raster_writes_sidecar_png(small_png, tmp_path):
    """``external_raster=True`` writes a sidecar PNG alongside the SVG."""
    canvas = _build_canvas_with_png(small_png)
    out = tmp_path / "out.svg"
    canvas.savefig(out, external_raster=True)
    assert out.exists()
    # Sidecar named ``<stem>-<idx>-<label>.png`` per the PR 6b plan.
    sidecars = list(tmp_path.glob("out-*-A.png"))
    assert len(sidecars) == 1, (
        f"expected one sidecar PNG, got: {list(tmp_path.iterdir())}"
    )


def test_external_raster_emits_relative_href(small_png, tmp_path):
    """The SVG <image> element references the sidecar via a relative
    href — NOT a data: URI."""
    canvas = _build_canvas_with_png(small_png)
    out = tmp_path / "out.svg"
    canvas.savefig(out, external_raster=True)

    tree = lxml_etree.parse(str(out))
    SVG_NS = "http://www.w3.org/2000/svg"
    XLINK_NS = "http://www.w3.org/1999/xlink"
    images = tree.getroot().xpath(
        "//svg:g[starts-with(@id, 'publiplots-panel-image-')]//svg:image",
        namespaces={"svg": SVG_NS},
    )
    assert len(images) == 1
    href = images[0].get(f"{{{XLINK_NS}}}href") or images[0].get("href")
    assert href is not None
    assert not href.startswith("data:"), (
        f"expected relative href to sidecar PNG, got data URI: {href[:60]}"
    )
    # Should be a relative filename ending in .png
    assert href.endswith(".png"), f"expected sidecar PNG href, got {href!r}"
    # And the file it references must exist relative to the SVG.
    sidecar = (out.parent / href).resolve()
    assert sidecar.exists()


def test_external_raster_no_op_for_pdf(small_png, tmp_path):
    """savefig pdf + external_raster=True silently writes PDF (no sidecar)."""
    canvas = _build_canvas_with_png(small_png)
    out = tmp_path / "out.pdf"
    canvas.savefig(out, external_raster=True)
    assert out.exists()
    sidecars = list(tmp_path.glob("out-*.png"))
    assert sidecars == []


def test_default_external_raster_false_uses_inline(small_png, tmp_path):
    """Default (external_raster=False) preserves PR 6a's inline data-URI
    behavior."""
    canvas = _build_canvas_with_png(small_png)
    out = tmp_path / "out.svg"
    canvas.savefig(out)  # default: external_raster=False
    tree = lxml_etree.parse(str(out))
    SVG_NS = "http://www.w3.org/2000/svg"
    XLINK_NS = "http://www.w3.org/1999/xlink"
    images = tree.getroot().xpath(
        "//svg:g[starts-with(@id, 'publiplots-panel-image-')]//svg:image",
        namespaces={"svg": SVG_NS},
    )
    assert len(images) == 1
    href = images[0].get(f"{{{XLINK_NS}}}href") or images[0].get("href")
    assert href.startswith("data:image/png;base64,")
    # And NO sidecar files were written.
    sidecars = list(tmp_path.glob("out-*.png"))
    assert sidecars == []


def test_external_raster_no_raster_sources_no_op(tmp_path):
    """A canvas with no raster PanelImage sources + external_raster=True
    is a silent no-op (no sidecars, just the SVG)."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out = tmp_path / "out.svg"
    canvas.savefig(out, external_raster=True)
    assert out.exists()
    sidecars = list(tmp_path.glob("out-*.png"))
    assert sidecars == []
