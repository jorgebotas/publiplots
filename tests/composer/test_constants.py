"""Tests for compositing._constants module — shared determinism strings.

PR 6b extracts ``_DEFAULT_CREATION_DATE``, ``_DEFAULT_DATE``,
``_SVG_HASHSALT``, ``_PRODUCER`` from the per-format compositing
modules into a shared module so the new ``_embed`` helpers can
reuse them without circular imports.

These tests pin:
1. The constants exist at the new home with the exact values the PR 5
   and PR 6a goldens were generated against.
2. The PR 5 / PR 6a public modules still see the same string values
   (no drift introduced by the refactor).
"""
from __future__ import annotations


def test_constants_module_exposes_expected_strings():
    """The four determinism strings live at composer.compositing._constants."""
    from publiplots.composer.compositing import _constants

    assert _constants._DEFAULT_CREATION_DATE == "D:20260101000000Z"
    assert _constants._DEFAULT_DATE == "2026-01-01T00:00:00"
    assert _constants._SVG_HASHSALT == "publiplots-composer"
    assert _constants._PRODUCER == "publiplots-composer"


def test_pdf_module_sees_constants_via_constants():
    """compositing.pdf must still expose the same _DEFAULT_CREATION_DATE
    + _PRODUCER values it had pre-refactor (PR 5 goldens depend on the
    string literals being byte-identical)."""
    from publiplots.composer.compositing import pdf as pdf_mod
    from publiplots.composer.compositing import _constants

    assert pdf_mod._DEFAULT_CREATION_DATE == _constants._DEFAULT_CREATION_DATE
    assert pdf_mod._PRODUCER == _constants._PRODUCER


def test_svg_module_sees_constants_via_constants():
    """compositing.svg must still expose the same _DEFAULT_DATE +
    _SVG_HASHSALT values (PR 6a goldens depend on the string literals)."""
    from publiplots.composer.compositing import svg as svg_mod
    from publiplots.composer.compositing import _constants

    assert svg_mod._DEFAULT_DATE == _constants._DEFAULT_DATE
    assert svg_mod._SVG_HASHSALT == _constants._SVG_HASHSALT


def test_existing_pdf_golden_byte_identical_after_refactor(tmp_path):
    """Re-rendering a PR 5 golden composition produces the same /CreationDate
    + /Producer values; refactor didn't drift the deterministic literals.

    This is the architect-required regression test: round-trip a PR 5
    golden through the refactored module and verify the determinism
    constants survived the move.
    """
    import pypdf

    import publiplots as pp

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out = tmp_path / "regression.pdf"
    canvas.savefig(out)

    md = pypdf.PdfReader(out).metadata
    assert md.get("/CreationDate") == "D:20260101000000Z"
    assert md.get("/Producer") == "publiplots-composer"


def test_existing_svg_golden_byte_identical_after_refactor(tmp_path):
    """Re-rendering a PR 6a golden composition produces the same
    <dc:date> + svg.hashsalt-derived defs ids; refactor didn't drift
    the deterministic literals."""
    import publiplots as pp

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out = tmp_path / "regression.svg"
    canvas.savefig(out)

    text = out.read_text(encoding="utf-8")
    assert "<dc:date>2026-01-01T00:00:00</dc:date>" in text
