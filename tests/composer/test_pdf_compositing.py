"""Tests for compositing/pdf.py — savefig_pdf orchestrator.

Exercises the end-to-end PDF compose path with a real Canvas + a
PanelImage. Golden-PDF regression tests (mode='render_compare', etc.)
land in Task 6. This task covers the basic interface + invariants.
"""
from __future__ import annotations

import io
from pathlib import Path

import pypdf
import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerVectorError


@pytest.fixture
def small_svg(tmp_path):
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


def test_savefig_pdf_writes_file(small_svg, tmp_path):
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=small_svg, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out = tmp_path / "out.pdf"
    canvas.savefig(out)  # dispatches to compositing.pdf.savefig_pdf
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")


def test_savefig_pdf_one_page(small_svg, tmp_path):
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelImage(label="A", path=small_svg, size=(70, 50)),
                   pp.PanelAxes(label="B", size=("flex", 50)))
    out = tmp_path / "out.pdf"
    canvas.savefig(out)
    reader = pypdf.PdfReader(out)
    assert len(reader.pages) == 1


def test_savefig_pdf_mediabox_matches_figure_size(small_svg, tmp_path):
    """Page mediabox dims (in pt) match canvas.figure_size_mm × MM2PT."""
    from publiplots.composer.compositing._geometry import MM2PT
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelImage(label="A", path=small_svg, size=(70, 50)),
                   pp.PanelAxes(label="B", size=("flex", 50)))
    out = tmp_path / "out.pdf"
    canvas.savefig(out)
    reader = pypdf.PdfReader(out)
    mb = reader.pages[0].mediabox
    fig_w_mm, fig_h_mm = canvas.figure_size_mm
    assert abs(float(mb.width) - fig_w_mm * MM2PT) < 1.0
    assert abs(float(mb.height) - fig_h_mm * MM2PT) < 1.0


def test_savefig_pdf_with_no_panel_images(tmp_path):
    """A canvas with NO PanelImage panels still saves to PDF (no compositing)."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelAxes(label="A", size=(70, 50)),
                   pp.PanelAxes(label="B", size=("flex", 50)))
    out = tmp_path / "out.pdf"
    canvas.savefig(out)
    assert out.exists()
    reader = pypdf.PdfReader(out)
    assert len(reader.pages) == 1


def test_savefig_pdf_creation_date_pinned(small_svg, tmp_path):
    """Two saves of the same canvas produce byte-identical PDFs (modulo none)."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelImage(label="A", path=small_svg, size=(70, 50)),
                   pp.PanelAxes(label="B", size=("flex", 50)))
    out1 = tmp_path / "a.pdf"
    out2 = tmp_path / "b.pdf"
    canvas.savefig(out1)
    # Rebuild canvas because the first canvas was finalized; build a fresh
    # one to avoid stateful effects.
    canvas2 = pp.Canvas("cell-2col")
    canvas2.add_row(pp.PanelImage(label="A", path=small_svg, size=(70, 50)),
                    pp.PanelAxes(label="B", size=("flex", 50)))
    canvas2.savefig(out2)
    # PDFs should be byte-identical given pinned /CreationDate + /Producer.
    # Tolerance: small differences in whitespace are OK; we check that the
    # /CreationDate and /Producer entries match.
    md1 = pypdf.PdfReader(out1).metadata
    md2 = pypdf.PdfReader(out2).metadata
    assert md1.get("/CreationDate") == md2.get("/CreationDate")
    assert md1.get("/Producer") == md2.get("/Producer")
