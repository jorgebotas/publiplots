"""Tests for PanelImage dataclass — PR 5 promotes from stub to real panel kind.

PanelImage references an external schematic file (PDF/SVG/PNG/JPG/TIFF)
to be vector-stamped (or raster-fallback) into a reserved canvas slot.
Construction-time validation enforces ext support, align/clip vocab,
and path existence.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerError, ComposerVectorError
from publiplots.composer.panels import PanelImage


# ---------------------------------------------------------------------------
# ComposerVectorError exception
# ---------------------------------------------------------------------------

def test_composer_vector_error_subclass_of_composer_error():
    assert issubclass(ComposerVectorError, ComposerError)


def test_composer_vector_error_carries_context():
    err = ComposerVectorError(
        "cairosvg failed",
        panel_label="A",
        path="/tmp/missing.svg",
        source_error="OSError: no such file",
    )
    assert err.panel_label == "A"
    assert err.path == "/tmp/missing.svg"
    assert "cairosvg failed" in str(err)


# ---------------------------------------------------------------------------
# PanelImage dataclass
# ---------------------------------------------------------------------------

def test_panel_image_basic_construction(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='10mm' height='10mm'/>")
    panel = PanelImage(label="A", path=p, size=(40.0, 30.0))
    assert panel.label == "A"
    assert panel.path == p
    assert panel.size == (40.0, 30.0)
    assert panel.align == "center"
    assert panel.clip == "fit"


def test_panel_image_accepts_string_path_and_normalizes(tmp_path):
    """str path → normalized to Path in __post_init__ (architect blocker #7)."""
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    panel = PanelImage(label="A", path=str(p), size=(40.0, 30.0))
    assert isinstance(panel.path, Path)
    assert panel.path == p


def test_panel_image_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "schematic.eps"
    p.write_text("dummy")
    with pytest.raises(ValueError, match=r"unsupported.*\.eps"):
        PanelImage(label="A", path=p, size=(40.0, 30.0))


def test_panel_image_rejects_missing_path(tmp_path):
    p = tmp_path / "does-not-exist.svg"
    with pytest.raises(FileNotFoundError, match=r"does-not-exist\.svg"):
        PanelImage(label="A", path=p, size=(40.0, 30.0))


def test_panel_image_rejects_invalid_align(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    with pytest.raises(ValueError, match=r"align.*'center'.*9"):
        PanelImage(label="A", path=p, size=(40.0, 30.0), align="middle")


def test_panel_image_rejects_invalid_clip(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    with pytest.raises(ValueError, match=r"clip.*'fit'.*'fill'.*'stretch'"):
        PanelImage(label="A", path=p, size=(40.0, 30.0), clip="cover")


def test_panel_image_accepts_all_9_align_values(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    for align in ("top-left", "top", "top-right", "left", "center",
                  "right", "bottom-left", "bottom", "bottom-right"):
        panel = PanelImage(label="A", path=p, size=(40.0, 30.0), align=align)
        assert panel.align == align


def test_panel_image_accepts_all_3_clip_values(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    for clip in ("fit", "fill", "stretch"):
        panel = PanelImage(label="A", path=p, size=(40.0, 30.0), clip=clip)
        assert panel.clip == clip


def test_panel_image_accepts_all_supported_extensions(tmp_path):
    contents = {
        ".svg": "<svg xmlns='http://www.w3.org/2000/svg'/>",
        ".pdf": "%PDF-1.4\n%dummy\n",  # not a valid PDF, but extension check is what we test
        ".png": "\x89PNG\r\n\x1a\n",
        ".jpg": "\xff\xd8\xff\xe0",
        ".jpeg": "\xff\xd8\xff\xe0",
        ".tif": "II*\x00",
        ".tiff": "II*\x00",
    }
    for ext, content in contents.items():
        p = tmp_path / f"schematic{ext}"
        p.write_bytes(content.encode("latin-1") if isinstance(content, str) else content)
        panel = PanelImage(label="A", path=p, size=(40.0, 30.0))
        assert panel.path == p


def test_panel_image_is_frozen_dataclass(tmp_path):
    p = tmp_path / "schematic.svg"
    p.write_text("<svg/>")
    panel = PanelImage(label="A", path=p, size=(40.0, 30.0))
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        panel.label = "B"


def test_panel_image_size_validation_propagates(tmp_path):
    """PR 1's _validate_panel_size shared helper rejects bad sizes."""
    p = tmp_path / "schematic.svg"
    p.write_text("<svg/>")
    with pytest.raises((ValueError, TypeError)):
        PanelImage(label="A", path=p, size=(0.0, 30.0))  # zero width


# ---------------------------------------------------------------------------
# _resources.load_schematic_as_pdf_bytes
# ---------------------------------------------------------------------------

@pytest.fixture
def small_svg(tmp_path):
    """A minimal SVG with a single visible circle + a text marker."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="40mm" height="30mm" viewBox="0 0 40 30">'
        '<circle cx="20" cy="15" r="10" fill="red"/>'
        '<text x="2" y="28" font-size="3">PR5-svg-marker</text>'
        '</svg>'
    )
    p = tmp_path / "schematic.svg"
    p.write_text(svg)
    return p


@pytest.fixture
def small_png(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (200, 200), color=(255, 0, 0))
    p = tmp_path / "schematic.png"
    img.save(p, dpi=(300, 300))
    return p


def test_load_svg_returns_pdf_bytes(small_svg):
    from publiplots.composer.compositing._resources import load_schematic_as_pdf_bytes
    pdf_bytes, kind = load_schematic_as_pdf_bytes(small_svg)
    assert kind == "vector"
    assert pdf_bytes.startswith(b"%PDF-")


def test_load_png_returns_pdf_bytes(small_png):
    from publiplots.composer.compositing._resources import load_schematic_as_pdf_bytes
    pdf_bytes, kind = load_schematic_as_pdf_bytes(small_png)
    assert kind == "raster"
    assert pdf_bytes.startswith(b"%PDF-")


def test_loaded_pdf_has_one_page_and_correct_mediabox(small_svg):
    """A 40mm × 30mm SVG → mediabox ≈ (40 mm × MM2PT, 30 mm × MM2PT) pt."""
    import io
    import pypdf
    from publiplots.composer.compositing._geometry import MM2PT
    from publiplots.composer.compositing._resources import load_schematic_as_pdf_bytes
    pdf_bytes, _ = load_schematic_as_pdf_bytes(small_svg)
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 1
    mb = reader.pages[0].mediabox
    assert abs(float(mb.width) - 40.0 * MM2PT) < 1.0  # tolerance: cairosvg
    assert abs(float(mb.height) - 30.0 * MM2PT) < 1.0


def test_load_pdf_passthrough(tmp_path):
    """A real .pdf input is read directly, not re-rendered through cairosvg."""
    import io
    import pypdf

    # Create a tiny one-page PDF with pypdf.
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72.0, height=72.0)
    p = tmp_path / "schematic.pdf"
    with open(p, "wb") as f:
        writer.write(f)

    from publiplots.composer.compositing._resources import load_schematic_as_pdf_bytes
    pdf_bytes, kind = load_schematic_as_pdf_bytes(p)
    assert kind == "vector"
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 1


def test_load_corrupt_svg_raises_composer_vector_error(tmp_path):
    p = tmp_path / "corrupt.svg"
    p.write_text("this is not svg")
    from publiplots.composer.exceptions import ComposerVectorError
    from publiplots.composer.compositing._resources import load_schematic_as_pdf_bytes
    with pytest.raises(ComposerVectorError, match=r"corrupt|svg|cairosvg"):
        load_schematic_as_pdf_bytes(p)
