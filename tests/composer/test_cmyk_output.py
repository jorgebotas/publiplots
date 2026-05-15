"""Tests for CMYK + TIFF compression on raster savefig.

PR 6b extends ``Canvas.savefig`` with two raster-only kwargs:

- ``cmyk: bool = False`` — convert RGB→CMYK on raster output for
  journal submission. Valid only for ``.tif/.tiff/.jpg/.jpeg``.
  ``cmyk=True`` paired with ``.pdf`` / ``.svg`` / ``.png`` raises with
  a helpful message.
- ``tiff_compression: str = "tiff_lzw"`` — TIFF compression knob.
  Defaults to LZW (matches matplotlib's default). Other values
  (``'tiff_deflate'``, ``'raw'``, etc) flow through to Pillow.
"""
from __future__ import annotations

import pytest

import publiplots as pp


@pytest.fixture
def simple_canvas():
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    return canvas


# ---------------------------------------------------------------------------
# Successful CMYK round-trips
# ---------------------------------------------------------------------------

def test_savefig_tiff_cmyk_round_trip(simple_canvas, tmp_path):
    """Canvas → CMYK TIFF → re-open → mode == 'CMYK'."""
    from PIL import Image
    out = tmp_path / "fig.tif"
    simple_canvas.savefig(out, cmyk=True)
    assert out.exists()
    with Image.open(out) as img:
        assert img.mode == "CMYK"


def test_savefig_jpeg_cmyk_round_trip(simple_canvas, tmp_path):
    """Canvas → CMYK JPEG → re-open → mode == 'CMYK'."""
    from PIL import Image
    out = tmp_path / "fig.jpg"
    simple_canvas.savefig(out, cmyk=True)
    assert out.exists()
    with Image.open(out) as img:
        assert img.mode == "CMYK"


def test_savefig_tiff_default_rgb(simple_canvas, tmp_path):
    """Default ``cmyk=False`` writes an RGB TIFF (mode 'RGBA' or 'RGB')."""
    from PIL import Image
    out = tmp_path / "fig.tif"
    simple_canvas.savefig(out)
    with Image.open(out) as img:
        assert img.mode != "CMYK"


def test_savefig_jpeg_default_rgb(simple_canvas, tmp_path):
    """Default ``cmyk=False`` writes an RGB JPEG."""
    from PIL import Image
    out = tmp_path / "fig.jpg"
    simple_canvas.savefig(out)
    with Image.open(out) as img:
        assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# CMYK rejection on non-raster / unsupported raster exts
# ---------------------------------------------------------------------------

def test_savefig_pdf_cmyk_raises(simple_canvas, tmp_path):
    """``cmyk=True`` + .pdf → ValueError pointing at raster outputs."""
    out = tmp_path / "fig.pdf"
    with pytest.raises(ValueError, match=r"cmyk.*raster|cmyk.*PDF"):
        simple_canvas.savefig(out, cmyk=True)
    assert not out.exists()


def test_savefig_svg_cmyk_raises(simple_canvas, tmp_path):
    """``cmyk=True`` + .svg → ValueError."""
    out = tmp_path / "fig.svg"
    with pytest.raises(ValueError, match=r"cmyk.*raster|cmyk.*SVG"):
        simple_canvas.savefig(out, cmyk=True)
    assert not out.exists()


def test_savefig_png_cmyk_raises(simple_canvas, tmp_path):
    """``cmyk=True`` + .png → ValueError (PNG doesn't support CMYK)."""
    out = tmp_path / "fig.png"
    with pytest.raises(ValueError, match=r"PNG.*CMYK|cmyk.*PNG"):
        simple_canvas.savefig(out, cmyk=True)
    assert not out.exists()


# ---------------------------------------------------------------------------
# TIFF compression knob
# ---------------------------------------------------------------------------

def test_savefig_tiff_compression_default_is_lzw(simple_canvas, tmp_path):
    """Bare ``canvas.savefig('out.tif')`` produces LZW-compressed output.

    Reviewer-required regression: matplotlib's TIFF backend defaults to
    RAW (uncompressed); without explicit thread-through of the
    ``tiff_compression='tiff_lzw'`` default kwarg via
    ``pil_kwargs={"compression": ...}``, the documented default would
    be silently ignored — producing 10–20× larger files than promised.
    """
    from PIL import Image
    out = tmp_path / "fig.tif"
    # No kwargs → API default tiff_compression='tiff_lzw' MUST take effect.
    simple_canvas.savefig(out)
    with Image.open(out) as img:
        compression = img.info.get("compression")
        assert compression in ("tiff_lzw", "lzw"), (
            f"bare-default canvas.savefig('out.tif') should produce LZW; "
            f"got {compression!r}"
        )


def test_savefig_tiff_compression_explicit_lzw(simple_canvas, tmp_path):
    """Explicit ``tiff_compression='tiff_lzw'`` is also LZW (parity)."""
    from PIL import Image
    out = tmp_path / "fig.tif"
    simple_canvas.savefig(out, tiff_compression="tiff_lzw")
    with Image.open(out) as img:
        assert img.info.get("compression") in ("tiff_lzw", "lzw")


def test_savefig_tiff_compression_raw_flow_through(simple_canvas, tmp_path):
    """``tiff_compression='raw'`` → uncompressed TIFF (no LZW applied)."""
    from PIL import Image
    out = tmp_path / "fig.tif"
    simple_canvas.savefig(out, tiff_compression="raw")
    with Image.open(out) as img:
        compression = img.info.get("compression")
        assert compression in ("raw", None), (
            f"expected raw / no compression, got {compression!r}"
        )


def test_savefig_tiff_cmyk_with_compression(simple_canvas, tmp_path):
    """``cmyk=True`` AND custom tiff_compression flow through together.

    Pillow normalizes ``tiff_deflate`` → ``tiff_adobe_deflate`` on
    read; we accept both spellings.
    """
    from PIL import Image
    out = tmp_path / "fig.tif"
    simple_canvas.savefig(out, cmyk=True, tiff_compression="tiff_deflate")
    with Image.open(out) as img:
        assert img.mode == "CMYK"
        compression = img.info.get("compression")
        assert compression in (
            "tiff_deflate", "tiff_adobe_deflate", "deflate",
        ), (
            f"expected tiff_deflate / tiff_adobe_deflate, got {compression!r}"
        )


# ---------------------------------------------------------------------------
# save_multiple + cmyk integration (closes the Task 7 gap)
# ---------------------------------------------------------------------------

def test_save_multiple_cmyk_only_raster_succeeds(simple_canvas, tmp_path):
    """``cmyk=True + formats=['tif','jpg']`` end-to-end writes both files."""
    from PIL import Image
    stem = tmp_path / "figure"
    paths = simple_canvas.save_multiple(stem, formats=["tif", "jpg"], cmyk=True)
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        with Image.open(p) as img:
            assert img.mode == "CMYK"
