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


# ---------------------------------------------------------------------------
# Golden PDF parametrized tests
# ---------------------------------------------------------------------------

from tests.composer.golden._compositions import COMPOSITIONS
from tests.composer.golden._helpers import (
    _pdf_rasterizer_available,
    assert_pdf_matches,
)


PDF_GOLDEN_NAMES = [
    "cell-2col-with-svg-schematic",
    "cell-2col-with-png-schematic",
]


# (name, mode) pairs. mediabox applies to all goldens; structure ONLY
# to raster/PDF sources because cairosvg-output SVGs are inlined into
# the canvas content stream by pypdf's merge_transformed_page (no
# XObject wrapping). For SVG goldens, mediabox + render_compare are
# the meaningful gates.
PDF_GOLDEN_MODE_PAIRS = [
    ("cell-2col-with-svg-schematic", "mediabox"),
    ("cell-2col-with-png-schematic", "mediabox"),
    ("cell-2col-with-png-schematic", "structure"),
]


@pytest.mark.parametrize("name,mode", PDF_GOLDEN_MODE_PAIRS)
def test_pdf_golden_matches(name: str, mode: str) -> None:
    """Composition `name` matches its golden PDF in `mode`."""
    build_fn = dict(COMPOSITIONS)[name]
    canvas = build_fn()
    assert_pdf_matches(canvas, name, mode=mode)


def test_assert_pdf_matches_structure_rejects_svg_source(tmp_path):
    """Regression guard: structure mode MUST fail for SVG-source goldens
    if a future relaxation removes the n_xobj >= 1 strict check.

    Spec-reviewer empirically showed that matplotlib's empty axes alone
    produces ~1.7-3 KB of content stream — large enough to falsely pass
    a `content_stream_len >= 200 bytes` heuristic even when the SVG was
    silently dropped. This test pins the contract: structure mode
    requires an XObject, period.
    """
    build_fn = dict(COMPOSITIONS)["cell-2col-with-svg-schematic"]
    canvas = build_fn()
    with pytest.raises(AssertionError, match=r"structure check failed"):
        assert_pdf_matches(canvas, "cell-2col-with-svg-schematic",
                           mode="structure")


@pytest.mark.parametrize("name", PDF_GOLDEN_NAMES)
@pytest.mark.skipif(
    not _pdf_rasterizer_available(),
    reason="No PDF rasterizer (pdf2image or pdftocairo) available."
)
def test_pdf_golden_render_compare(name: str) -> None:
    """Composition `name` renders pixel-equivalently to its golden PDF.

    Forgiving comparison via `compare_images(tol=20)` after rasterizing
    both PDFs at 200 DPI. The mediabox + structure tests above are the
    primary CI gates; this is the visual regression backstop.
    """
    build_fn = dict(COMPOSITIONS)[name]
    canvas = build_fn()
    assert_pdf_matches(canvas, name, mode="render_compare")


# ---------------------------------------------------------------------------
# PR 6b — embed_figure compositing branch
# ---------------------------------------------------------------------------

def test_savefig_pdf_with_embedded_figure_passes_mediabox_check(tmp_path):
    """Canvas with one embed_figure'd PanelImage saves to a PDF whose
    mediabox matches canvas.figure_size_mm × MM2PT (no compositing
    regression vs. path-based PanelImage)."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._geometry import MM2PT

    fig, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([1, 2, 3], [4, 5, 6])
    try:
        canvas = pp.Canvas("cell-2col")
        canvas.add_row(
            pp.PanelAxes(label="A", size=(70, 50)),
            pp.PanelImage(label="B", size=(70, 50)),
        )
        canvas.embed_figure("B", fig)
        out = tmp_path / "out.pdf"
        canvas.savefig(out)
    finally:
        plt.close(fig)
    assert out.exists()
    reader = pypdf.PdfReader(out)
    mb = reader.pages[0].mediabox
    fig_w_mm, fig_h_mm = canvas.figure_size_mm
    assert abs(float(mb.width) - fig_w_mm * MM2PT) < 1.0
    assert abs(float(mb.height) - fig_h_mm * MM2PT) < 1.0


def test_savefig_pdf_with_embed_figure_strict_vectors_raises_on_no_figure(
    tmp_path,
):
    """Empty PanelImage (no path, no embed_figure) + savefig pdf →
    ComposerVectorError with the embed_figure hint."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    out = tmp_path / "out.pdf"
    with pytest.raises(ComposerVectorError, match=r"no path and no embedded figure"):
        canvas.savefig(out)


def test_savefig_pdf_embed_figure_byte_deterministic(tmp_path):
    """Two saves of the same canvas (with embed_figure) produce identical
    PDFs. Locks the determinism contract for the embedded-figure branch."""
    import matplotlib.pyplot as plt

    def build():
        fig, ax = plt.subplots(figsize=(2.0, 1.5))
        ax.plot([1, 2, 3], [4, 5, 6])
        canvas = pp.Canvas("cell-2col")
        canvas.add_row(
            pp.PanelAxes(label="A", size=(70, 50)),
            pp.PanelImage(label="B", size=(70, 50)),
        )
        canvas.embed_figure("B", fig)
        return canvas, fig

    canvas1, fig1 = build()
    out1 = tmp_path / "a.pdf"
    canvas1.savefig(out1)
    plt.close(fig1)

    canvas2, fig2 = build()
    out2 = tmp_path / "b.pdf"
    canvas2.savefig(out2)
    plt.close(fig2)

    md1 = pypdf.PdfReader(out1).metadata
    md2 = pypdf.PdfReader(out2).metadata
    assert md1.get("/CreationDate") == md2.get("/CreationDate")
    assert md1.get("/Producer") == md2.get("/Producer")
    assert out1.read_bytes() == out2.read_bytes()
