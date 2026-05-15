"""Tests for the extracted _pillow_to_pdf_bytes helper.

The helper centralizes the Pillow→PDF conversion logic that was
previously duplicated between
:func:`publiplots.composer.compositing._resources.load_schematic_as_pdf_bytes`
(raster branch) and
:func:`publiplots.composer.compositing.pdf._raster_fallback`. PR 6a
extracts it; both call sites now delegate.
"""
from __future__ import annotations

import io
from pathlib import Path

import pypdf
import pytest


def _read_pdf_mediabox(pdf_bytes: bytes):
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    mb = reader.pages[0].mediabox
    return float(mb.width), float(mb.height)


@pytest.fixture
def png_path(tmp_path):
    from PIL import Image
    p = tmp_path / "frame.png"
    Image.new("RGB", (300, 200), color="white").save(
        p, dpi=(300, 300),
    )
    return p


@pytest.fixture
def jpg_path(tmp_path):
    from PIL import Image
    p = tmp_path / "frame.jpg"
    Image.new("RGB", (300, 200), color="white").save(
        p, format="JPEG", dpi=(300, 300),
    )
    return p


@pytest.fixture
def tif_path(tmp_path):
    from PIL import Image
    p = tmp_path / "frame.tif"
    Image.new("RGB", (300, 200), color="white").save(
        p, format="TIFF", dpi=(300, 300),
    )
    return p


def test_pillow_to_pdf_bytes_returns_pdf_signature(png_path):
    from publiplots.composer.compositing._resources import (
        _pillow_to_pdf_bytes,
    )
    pdf_bytes = _pillow_to_pdf_bytes(png_path)
    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert pdf_bytes[:5] == b"%PDF-"


def test_pillow_to_pdf_bytes_jpg(jpg_path):
    from publiplots.composer.compositing._resources import (
        _pillow_to_pdf_bytes,
    )
    pdf_bytes = _pillow_to_pdf_bytes(jpg_path)
    assert pdf_bytes[:5] == b"%PDF-"


def test_pillow_to_pdf_bytes_tiff(tif_path):
    from publiplots.composer.compositing._resources import (
        _pillow_to_pdf_bytes,
    )
    pdf_bytes = _pillow_to_pdf_bytes(tif_path)
    assert pdf_bytes[:5] == b"%PDF-"


def test_pillow_to_pdf_bytes_mediabox_300dpi(png_path):
    """A 300x200-px image at 300 dpi → mediabox of 1in × 0.667in = 72 × 48 pt."""
    from publiplots.composer.compositing._resources import (
        _pillow_to_pdf_bytes,
    )
    pdf_bytes = _pillow_to_pdf_bytes(png_path)
    w_pt, h_pt = _read_pdf_mediabox(pdf_bytes)
    # 300px / 300dpi = 1in = 72pt; 200px / 300dpi = 0.667in ≈ 48pt
    assert abs(w_pt - 72.0) < 1.0
    assert abs(h_pt - 48.0) < 1.0


def test_pillow_to_pdf_bytes_explicit_dpi_override(tmp_path):
    """Passing an explicit dpi override resizes the mediabox accordingly."""
    from PIL import Image
    from publiplots.composer.compositing._resources import (
        _pillow_to_pdf_bytes,
    )
    p = tmp_path / "no_dpi.png"
    # Save a PNG WITHOUT dpi info; 300x200 px image.
    Image.new("RGB", (300, 200), color="white").save(p)
    pdf_bytes = _pillow_to_pdf_bytes(p, dpi=150.0)
    w_pt, h_pt = _read_pdf_mediabox(pdf_bytes)
    # 300 px / 150 dpi = 2 in = 144 pt; 200 px / 150 dpi = 1.333 in ≈ 96 pt
    assert abs(w_pt - 144.0) < 1.0
    assert abs(h_pt - 96.0) < 1.0


def test_pillow_to_pdf_bytes_rgba_converted(tmp_path):
    """Pillow PDF save needs RGB or L; RGBA images are converted."""
    from PIL import Image
    from publiplots.composer.compositing._resources import (
        _pillow_to_pdf_bytes,
    )
    p = tmp_path / "rgba.png"
    Image.new("RGBA", (100, 100), color=(255, 0, 0, 128)).save(
        p, dpi=(300, 300),
    )
    pdf_bytes = _pillow_to_pdf_bytes(p)
    assert pdf_bytes[:5] == b"%PDF-"
