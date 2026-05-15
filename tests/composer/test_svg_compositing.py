"""Tests for compositing/svg.py — savefig_svg orchestrator.

Mirrors test_pdf_compositing.py's structure but for the SVG path:
basic interface + invariants in this module; golden-SVG regression
tests live alongside.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerVectorError


# ---------------------------------------------------------------------------
# Task 3 — lxml ImportError → install hint
# ---------------------------------------------------------------------------

def test_no_lxml_raises_install_hint(monkeypatch, tmp_path):
    """Patching `lxml` to fail import → savefig_svg raises with install hint.

    We use ``importlib.reload`` because the module-level imports happen
    once at first import; a later monkeypatch of ``sys.modules['lxml']``
    has no effect on already-imported references unless we reload.
    """
    # Force the `import lxml.etree` inside savefig_svg to raise.
    monkeypatch.setitem(sys.modules, "lxml", None)
    monkeypatch.setitem(sys.modules, "lxml.etree", None)

    # Re-import the module so our patched ImportError actually bites.
    import publiplots.composer.compositing.svg as svg_mod
    importlib.reload(svg_mod)

    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out = tmp_path / "out.svg"
    with pytest.raises(ComposerVectorError, match=r"lxml.*publiplots\[composer\]"):
        svg_mod.savefig_svg(canvas.figure, out, panels=list(canvas._panels_list))

    # Restore the module so subsequent tests in this run don't see the
    # broken state.
    monkeypatch.undo()
    importlib.reload(svg_mod)
