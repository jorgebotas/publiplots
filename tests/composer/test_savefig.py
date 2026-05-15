"""Tests for canvas.savefig — raster pipeline only in PR 1."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pytest

import publiplots as pp


def test_savefig_png_works(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.scatter([1, 2, 3], [1, 4, 2])

    out = tmp_path / "fig.png"
    canvas.savefig(out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_savefig_jpg_works(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.scatter([1, 2], [1, 2])

    out = tmp_path / "fig.jpg"
    canvas.savefig(out)
    assert out.exists()


def test_savefig_tiff_works(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.scatter([1, 2], [1, 2])

    out = tmp_path / "fig.tiff"
    canvas.savefig(out)
    assert out.exists()


def test_savefig_pdf_writes_pdf(tmp_path):
    """PR 5 lands the vector PDF compositing pipeline.

    Replaces the prior NotImplementedError contract — saving to .pdf
    now produces a valid PDF (axes-only canvases stamp no schematics).
    """
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))

    out = tmp_path / "fig.pdf"
    canvas.savefig(out)
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")


def test_savefig_svg_writes_svg(tmp_path):
    """PR 6a lands the in-tree SVG composer.

    Replaces the prior NotImplementedError contract — saving to .svg
    now produces a valid SVG (axes-only canvases stamp no schematics).
    """
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))

    out = tmp_path / "fig.svg"
    canvas.savefig(out)
    assert out.exists()
    text = out.read_bytes()
    assert b"<svg" in text


def test_savefig_unknown_extension_raises(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))

    out = tmp_path / "fig.xyz"
    with pytest.raises(ValueError, match="unknown.*extension"):
        canvas.savefig(out)


def test_savefig_before_add_row_raises():
    """Saving an empty canvas — no figure to write."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(RuntimeError, match="add_row"):
        canvas.savefig("/tmp/never-written.png")


def test_savefig_accepts_str_and_path(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))

    out_str = str(tmp_path / "via_str.png")
    out_path = tmp_path / "via_path.png"

    canvas.savefig(out_str)
    canvas.savefig(out_path)

    assert Path(out_str).exists()
    assert out_path.exists()
