"""Tests for compositing/_resources.py — schematic loaders.

PR 6a adds two new helpers alongside the existing PDF-bytes loader:

- :func:`_pillow_to_data_uri` — raster → ``data:image/png;base64,...``
  string for SVG ``<image>`` element embedding.
- :func:`load_schematic_as_svg_element` — multi-extension dispatch
  returning an lxml element + intrinsic mm size suitable for
  inclusion in the canvas SVG tree.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

lxml_etree = pytest.importorskip("lxml.etree")


# ---------------------------------------------------------------------------
# _pillow_to_data_uri
# ---------------------------------------------------------------------------

@pytest.fixture
def png_path(tmp_path):
    from PIL import Image
    p = tmp_path / "frame.png"
    Image.new("RGB", (300, 200), color="white").save(
        p, dpi=(300, 300),
    )
    return p


@pytest.fixture
def svg_path(tmp_path):
    p = tmp_path / "frame.svg"
    p.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="40mm" height="30mm" viewBox="0 0 40 30">'
        '<rect width="40" height="30" fill="blue"/>'
        '</svg>'
    )
    return p


def test_pillow_to_data_uri_returns_png_data_uri(png_path):
    from publiplots.composer.compositing._resources import (
        _pillow_to_data_uri,
    )
    uri = _pillow_to_data_uri(png_path)
    assert isinstance(uri, str)
    assert uri.startswith("data:image/png;base64,")
    payload = uri.split(",", 1)[1]
    decoded = base64.b64decode(payload)
    # Re-saved PNG bytes start with the standard 8-byte PNG header.
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_pillow_to_data_uri_normalises_rgba(tmp_path):
    """RGBA PNGs are re-saved (Pillow's PDF-style conversion is irrelevant
    here — we just need a valid PNG byte stream)."""
    from PIL import Image
    from publiplots.composer.compositing._resources import (
        _pillow_to_data_uri,
    )
    p = tmp_path / "rgba.png"
    Image.new("RGBA", (50, 50), color=(255, 0, 0, 128)).save(
        p, dpi=(300, 300),
    )
    uri = _pillow_to_data_uri(p)
    assert uri.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# load_schematic_as_svg_element
# ---------------------------------------------------------------------------

def test_load_svg_returns_element_and_intrinsic_mm(svg_path):
    from publiplots.composer.compositing._resources import (
        load_schematic_as_svg_element,
    )
    element, (w_mm, h_mm), kind = load_schematic_as_svg_element(svg_path)
    # Returned element is an lxml _Element rooted at <svg>.
    assert element is not None
    assert element.tag.endswith("svg") or element.tag == "svg"
    # mm-Inkscape SVG → 40 × 30 mm intrinsic.
    assert abs(w_mm - 40.0) < 1e-6
    assert abs(h_mm - 30.0) < 1e-6
    assert kind == "vector"


def test_load_svg_corrupt_raises(tmp_path):
    from publiplots.composer.compositing._resources import (
        load_schematic_as_svg_element,
    )
    from publiplots.composer.exceptions import ComposerVectorError
    p = tmp_path / "bad.svg"
    p.write_text("this is not svg <<<")
    with pytest.raises(ComposerVectorError, match=r"svg|parse|lxml"):
        load_schematic_as_svg_element(p)


def test_load_svg_missing_viewbox_raises(tmp_path):
    """An SVG without a viewBox cannot be sized → ComposerVectorError."""
    from publiplots.composer.compositing._resources import (
        load_schematic_as_svg_element,
    )
    from publiplots.composer.exceptions import ComposerVectorError
    p = tmp_path / "no_viewbox.svg"
    p.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="40mm" height="30mm"/>'
    )
    with pytest.raises(ComposerVectorError, match=r"viewBox"):
        load_schematic_as_svg_element(p)


def test_load_png_returns_image_element(png_path):
    """A PNG schematic returns an lxml <image> element with a data URI href."""
    from publiplots.composer.compositing._resources import (
        load_schematic_as_svg_element,
    )
    element, (w_mm, h_mm), kind = load_schematic_as_svg_element(png_path)
    assert kind == "raster"
    # Element is <image> in SVG namespace.
    assert element.tag.endswith("image")
    # xlink:href OR href present + data URI.
    XLINK_NS = "http://www.w3.org/1999/xlink"
    href = element.get(f"{{{XLINK_NS}}}href") or element.get("href")
    assert href is not None
    assert href.startswith("data:image/png;base64,")
    # 300x200 px @ 300 dpi → 25.4mm × 16.93mm.
    assert abs(w_mm - 300.0 * 25.4 / 300.0) < 1e-3
    assert abs(h_mm - 200.0 * 25.4 / 300.0) < 1e-3


def test_load_pdf_in_svg_path_raises(tmp_path):
    """PDF-source schematics in the SVG output path are explicitly rejected."""
    from publiplots.composer.compositing._resources import (
        load_schematic_as_svg_element,
    )
    from publiplots.composer.exceptions import ComposerVectorError
    # Build a minimal valid 1-page PDF via Pillow.
    from PIL import Image
    p = tmp_path / "schematic.pdf"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(p, format="PDF", resolution=300.0)
    with pytest.raises(ComposerVectorError, match=r"PDF.*SVG.*not supported|cannot be embedded"):
        load_schematic_as_svg_element(p, label="A")


def test_load_unsupported_extension_raises(tmp_path):
    """Bogus extensions (e.g., .xyz) → ComposerVectorError."""
    from publiplots.composer.compositing._resources import (
        load_schematic_as_svg_element,
    )
    from publiplots.composer.exceptions import ComposerVectorError
    p = tmp_path / "bogus.xyz"
    p.write_text("nope")
    with pytest.raises(ComposerVectorError, match=r"unsupported|extension|xyz"):
        load_schematic_as_svg_element(p)


def test_load_svg_label_in_pdf_error_message(tmp_path):
    """The PDF-in-SVG error names the panel label and the format."""
    from publiplots.composer.compositing._resources import (
        load_schematic_as_svg_element,
    )
    from publiplots.composer.exceptions import ComposerVectorError
    from PIL import Image
    p = tmp_path / "myschematic.pdf"
    img = Image.new("RGB", (50, 50), color="white")
    img.save(p, format="PDF", resolution=300.0)
    with pytest.raises(ComposerVectorError) as excinfo:
        load_schematic_as_svg_element(p, label="my-label")
    msg = str(excinfo.value)
    assert "my-label" in msg or "my_label" in msg or "my-label" in repr(msg)
    assert "myschematic.pdf" in msg
