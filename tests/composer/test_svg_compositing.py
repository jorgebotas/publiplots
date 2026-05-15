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

lxml_etree = pytest.importorskip("lxml.etree")


@pytest.fixture
def small_svg(tmp_path):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="40mm" height="30mm" viewBox="0 0 40 30">'
        '<circle cx="20" cy="15" r="10" fill="red"/>'
        '<text x="2" y="28" font-size="3">PR6a-svg-marker</text>'
        '</svg>'
    )
    p = tmp_path / "schematic.svg"
    p.write_text(svg)
    return p


@pytest.fixture
def small_png(tmp_path):
    from PIL import Image
    p = tmp_path / "schematic.png"
    Image.new("RGB", (300, 200), color="white").save(
        p, dpi=(300, 300),
    )
    return p


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


# ---------------------------------------------------------------------------
# Task 5 — savefig_svg orchestrator
# ---------------------------------------------------------------------------

def _build_canvas_with_panels(small_svg):
    """Helper: build + finalize a 2-col canvas with one PanelImage."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=small_svg, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    _ = canvas.figure  # force finalization
    return canvas


def _save_svg_directly(canvas, out, **kwargs):
    """Call savefig_svg directly, bypassing dispatch (Task 6 wires it)."""
    from publiplots.composer.compositing.svg import savefig_svg
    savefig_svg(
        canvas.figure, out,
        panels=list(canvas._panels_list),
        strict_vectors=getattr(canvas, "_strict_vectors", False),
        **kwargs,
    )


def test_savefig_svg_writes_file(small_svg, tmp_path):
    canvas = _build_canvas_with_panels(small_svg)
    out = tmp_path / "out.svg"
    _save_svg_directly(canvas, out)
    assert out.exists()
    text = out.read_bytes()
    assert b"<svg" in text


def test_savefig_svg_root_has_viewbox(small_svg, tmp_path):
    """Output SVG root carries a viewBox in matplotlib's pt-based user units."""
    canvas = _build_canvas_with_panels(small_svg)
    out = tmp_path / "out.svg"
    _save_svg_directly(canvas, out)
    tree = lxml_etree.parse(str(out))
    root = tree.getroot()
    vb = root.get("viewBox")
    assert vb is not None
    parts = [float(s) for s in vb.replace(",", " ").split()]
    assert len(parts) == 4
    assert parts[2] > 0
    assert parts[3] > 0


def test_savefig_svg_no_panel_images(tmp_path):
    """A canvas with NO PanelImage panels still saves to SVG (no compositing)."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    _ = canvas.figure
    out = tmp_path / "out.svg"
    _save_svg_directly(canvas, out)
    assert out.exists()
    tree = lxml_etree.parse(str(out))
    SVG_NS = "http://www.w3.org/2000/svg"
    groups = tree.getroot().xpath(
        "//svg:g[starts-with(@id, 'publiplots-panel-image-')]",
        namespaces={"svg": SVG_NS},
    )
    assert groups == []


def test_savefig_svg_panel_image_wraps_in_g_with_id(small_svg, tmp_path):
    """Each PanelImage produces a wrapper <g id='publiplots-panel-image-...'>."""
    canvas = _build_canvas_with_panels(small_svg)
    out = tmp_path / "out.svg"
    _save_svg_directly(canvas, out)
    tree = lxml_etree.parse(str(out))
    SVG_NS = "http://www.w3.org/2000/svg"
    groups = tree.getroot().xpath(
        "//svg:g[starts-with(@id, 'publiplots-panel-image-')]",
        namespaces={"svg": SVG_NS},
    )
    assert len(groups) == 1
    assert groups[0].get("id") == "publiplots-panel-image-0-A"
    transform = groups[0].get("transform")
    assert "translate(" in transform
    assert "scale(" in transform


def test_savefig_svg_panel_image_unlabeled_id(small_svg, tmp_path):
    """PanelImage with label=False gets the 'unlabeled' suffix."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label=False, path=small_svg, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    _ = canvas.figure
    out = tmp_path / "out.svg"
    _save_svg_directly(canvas, out)
    tree = lxml_etree.parse(str(out))
    SVG_NS = "http://www.w3.org/2000/svg"
    groups = tree.getroot().xpath(
        "//svg:g[starts-with(@id, 'publiplots-panel-image-')]",
        namespaces={"svg": SVG_NS},
    )
    assert len(groups) == 1
    assert groups[0].get("id") == "publiplots-panel-image-0-unlabeled"


def test_savefig_svg_png_panel_uses_image_data_uri(small_png, tmp_path):
    """PNG-source PanelImage produces an embedded <image> data URI."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=small_png, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    _ = canvas.figure
    out = tmp_path / "out.svg"
    _save_svg_directly(canvas, out)
    tree = lxml_etree.parse(str(out))
    SVG_NS = "http://www.w3.org/2000/svg"
    XLINK_NS = "http://www.w3.org/1999/xlink"
    images = tree.getroot().xpath(
        "//svg:g[starts-with(@id, 'publiplots-panel-image-')]//svg:image",
        namespaces={"svg": SVG_NS},
    )
    assert len(images) == 1
    href = images[0].get(f"{{{XLINK_NS}}}href") or images[0].get("href")
    assert href is not None
    assert href.startswith("data:image/png;base64,")


def test_savefig_svg_strict_vectors_raises_on_corrupt_svg(tmp_path):
    """strict_vectors=True + corrupt SVG → ComposerVectorError."""
    p = tmp_path / "corrupt.svg"
    p.write_text("not svg <<<")
    canvas = pp.Canvas("cell-2col", strict_vectors=True)
    canvas.add_row(
        pp.PanelImage(label="A", path=p, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    _ = canvas.figure
    from publiplots.composer.compositing.svg import savefig_svg
    out = tmp_path / "out.svg"
    with pytest.raises(ComposerVectorError):
        savefig_svg(
            canvas.figure, out,
            panels=list(canvas._panels_list),
            strict_vectors=True,
        )


def test_savefig_svg_strict_vectors_false_falls_back_with_warning(
    monkeypatch, tmp_path,
):
    """strict_vectors=False + simulated SVG loader failure → UserWarning +
    raster <image> fallback. The caller's PNG path is the actual fallback
    target (Pillow can open it)."""
    import warnings as _w
    from PIL import Image
    from publiplots.composer.compositing import _resources as res_mod
    from publiplots.composer.exceptions import ComposerVectorError

    png_path = tmp_path / "real.png"
    Image.new("RGB", (200, 200), color="green").save(
        png_path, dpi=(300, 300),
    )

    def fake_loader(path, *, label=None):
        raise ComposerVectorError(
            "simulated svg load failure",
            panel_label=str(label) if label not in (None, False) else None,
            path=str(path),
            source_error="forced",
        )

    monkeypatch.setattr(res_mod, "load_schematic_as_svg_element", fake_loader)

    canvas = pp.Canvas("cell-2col", strict_vectors=False)
    canvas.add_row(
        pp.PanelImage(label="A", path=png_path, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    _ = canvas.figure
    out = tmp_path / "out.svg"
    from publiplots.composer.compositing.svg import savefig_svg
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        savefig_svg(
            canvas.figure, out,
            panels=list(canvas._panels_list),
            strict_vectors=False,
        )
    assert out.exists()
    fallback_warnings = [
        w for w in caught
        if "vector load" in str(w.message)
        and "raster" in str(w.message).lower()
    ]
    assert fallback_warnings, (
        f"expected vector-fallback UserWarning; got: "
        f"{[str(w.message) for w in caught]}"
    )


# ---------------------------------------------------------------------------
# Task 8 — golden-SVG parametrized regression tests
# ---------------------------------------------------------------------------

from tests.composer.golden._compositions import COMPOSITIONS
from tests.composer.golden._helpers import (
    _svg_renderer_available,
    assert_svg_matches,
)


SVG_GOLDEN_NAMES = [
    "cell-2col-with-svg-schematic",
    "cell-2col-with-png-schematic",
]


# (name, mode) pairs. viewbox + structure are meaningful for both
# schematic types. render_compare adds a visual gate when cairosvg is
# available.
SVG_GOLDEN_MODE_PAIRS = [
    (name, mode)
    for name in SVG_GOLDEN_NAMES
    for mode in ("viewbox", "structure")
]


@pytest.mark.parametrize("name,mode", SVG_GOLDEN_MODE_PAIRS)
def test_svg_golden_matches(name: str, mode: str) -> None:
    """Composition `name` matches its golden SVG in `mode`."""
    build_fn = dict(COMPOSITIONS)[name]
    canvas = build_fn()
    assert_svg_matches(canvas, name, mode=mode)


@pytest.mark.parametrize("name", SVG_GOLDEN_NAMES)
@pytest.mark.skipif(
    not _svg_renderer_available(),
    reason="cairosvg/Pillow unavailable for SVG render_compare.",
)
def test_svg_golden_render_compare(name: str) -> None:
    """Composition `name` renders pixel-equivalently to its golden SVG."""
    build_fn = dict(COMPOSITIONS)[name]
    canvas = build_fn()
    assert_svg_matches(canvas, name, mode="render_compare")


# ---------------------------------------------------------------------------
# Task 9 — byte-determinism regression
# ---------------------------------------------------------------------------

def test_savefig_svg_byte_deterministic(small_svg, tmp_path):
    """Two saves of the same canvas produce byte-identical SVGs.

    Spike Finding 2: matplotlib's SVG writer emits `<defs>` IDs whose
    suffixes are derived from the rcParam `svg.hashsalt` AND a
    `<dc:date>` driven from `metadata['Date']`. With both pinned, the
    output bytes are reproducible. This test guards the contract from
    future regressions (e.g., a new matplotlib release adding a third
    randomness source).
    """
    canvas1 = pp.Canvas("cell-2col")
    canvas1.add_row(
        pp.PanelImage(label="A", path=small_svg, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out1 = tmp_path / "a.svg"
    canvas1.savefig(out1)

    canvas2 = pp.Canvas("cell-2col")
    canvas2.add_row(
        pp.PanelImage(label="A", path=small_svg, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out2 = tmp_path / "b.svg"
    canvas2.savefig(out2)

    assert out1.read_bytes() == out2.read_bytes(), (
        "Two saves of the same canvas produced different SVG bytes; "
        "byte-determinism contract broken (svg.hashsalt + metadata={'Date'}"
        " were expected to suffice)."
    )


def test_metadata_date_omit_strips_dc_date(small_svg, tmp_path):
    """`metadata_date='omit'` removes <dc:date> from the output entirely.

    Distinct from the default (`metadata_date=None` → `_DEFAULT_DATE`),
    which writes a deterministic timestamp; `'omit'` writes nothing.
    """
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=small_svg, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out = tmp_path / "omit.svg"
    canvas.savefig(out, metadata_date="omit")

    text = out.read_text(encoding="utf-8")
    assert "<dc:date>" not in text, (
        "metadata_date='omit' should strip the <dc:date> element "
        "entirely; matplotlib treats Date=None as the suppress sentinel."
    )


def test_metadata_date_literal_writes_supplied_value(small_svg, tmp_path):
    """A literal `metadata_date` string flows through to <dc:date>."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=small_svg, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    out = tmp_path / "literal.svg"
    canvas.savefig(out, metadata_date="2024-12-31T23:59:59")

    text = out.read_text(encoding="utf-8")
    assert "<dc:date>2024-12-31T23:59:59</dc:date>" in text, (
        "metadata_date literal should write through to <dc:date>."
    )
