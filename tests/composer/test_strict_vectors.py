"""Tests for Canvas(strict_vectors=...) flag + raster fallback semantics.

PR 5 introduces the strict_vectors gate: True raises on any schematic
load failure; False (default) falls back to a raster re-render and
emits a UserWarning.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerVectorError


@pytest.fixture
def corrupt_svg(tmp_path):
    p = tmp_path / "corrupt.svg"
    p.write_text("this is not svg data")
    return p


def test_strict_vectors_true_raises_on_corrupt_svg(corrupt_svg, tmp_path):
    canvas = pp.Canvas("cell-2col", strict_vectors=True)
    canvas.add_row(
        pp.PanelImage(label="A", path=corrupt_svg, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out = tmp_path / "out.pdf"
    with pytest.raises(ComposerVectorError, match=r"corrupt|svg|cairosvg"):
        canvas.savefig(out)


def test_strict_vectors_false_happy_png_no_warning(tmp_path):
    """A valid PNG schematic with strict_vectors=False writes PDF cleanly.

    The PNG goes through the raster path of _resources directly — NOT
    a fallback. No warning should fire.
    """
    from PIL import Image
    png_path = tmp_path / "fallback.png"
    Image.new("RGB", (200, 200), color="red").save(png_path, dpi=(300, 300))

    canvas = pp.Canvas("cell-2col", strict_vectors=False)
    canvas.add_row(
        pp.PanelImage(label="A", path=png_path, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out = tmp_path / "out.pdf"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        canvas.savefig(out)
    # No vector-fallback warning should have fired (raster source took
    # the raster path, not a fallback).
    fallback_warnings = [w for w in caught
                         if "fallback" in str(w.message).lower()]
    assert not fallback_warnings, (
        f"unexpected fallback warning on raster source: {fallback_warnings}"
    )
    assert out.exists()


def test_strict_vectors_false_corrupt_svg_warns_and_falls_back(
    monkeypatch, tmp_path,
):
    """A corrupt SVG with strict_vectors=False emits UserWarning + falls back.

    The corrupt SVG can't be rasterized either (Pillow can't read SVG),
    so we mock the resources loader's SVG branch to raise a controlled
    error and verify the fallback path is exercised. The actual fallback
    target uses a valid PNG byte stream supplied by the mock.
    """
    from publiplots.composer.compositing import pdf as pdf_mod
    from publiplots.composer.exceptions import ComposerVectorError
    from PIL import Image

    # Build a valid PNG schematic.
    png_path = tmp_path / "real.png"
    Image.new("RGB", (200, 200), color="blue").save(png_path, dpi=(300, 300))

    # Use a "schematic.svg" pointer for the PanelImage but route the
    # resource loader through a mock that ALWAYS raises ComposerVectorError
    # on first call (simulating cairosvg failing). The PR 5 fallback then
    # rasterizes via Pillow against the same path — but since the path is
    # actually the PNG above, Pillow CAN open it and fallback succeeds.
    svg_path = tmp_path / "fake.svg"
    svg_path.write_text("<svg/>")  # minimal valid stub for path-exists check

    real_loader = pdf_mod.load_schematic_as_pdf_bytes
    call_count = {"n": 0}

    def fake_loader(path):
        call_count["n"] += 1
        # First call: raise as if cairosvg failed.
        # The fallback in pdf.py's _raster_fallback takes the path
        # directly; so to test the fallback path landing, we point
        # PanelImage at the real PNG instead and only mock the loader.
        raise ComposerVectorError(
            "simulated cairosvg failure",
            panel_label="A",
            path=str(path),
            source_error="forced",
        )

    monkeypatch.setattr(pdf_mod, "load_schematic_as_pdf_bytes", fake_loader)

    canvas = pp.Canvas("cell-2col", strict_vectors=False)
    canvas.add_row(
        pp.PanelImage(label="A", path=png_path, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out = tmp_path / "out.pdf"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        canvas.savefig(out)

    # The fallback warning must fire (panel label A, "vector load" + "raster").
    fallback_warnings = [w for w in caught
                         if "vector load" in str(w.message)
                         and "raster" in str(w.message)]
    assert fallback_warnings, (
        f"expected vector-fallback UserWarning, got {[str(w.message) for w in caught]}"
    )
    assert out.exists()
