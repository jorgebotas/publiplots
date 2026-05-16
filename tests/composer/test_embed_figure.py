"""Tests for Canvas.embed_figure + Panel.embedded_figure plumbing.

PR 6b adds a post-staging mutation API: stage a PanelImage with no
path, then embed a matplotlib Figure into the slot. Compose-time
dispatches the embedded-figure branch in the PDF / SVG composers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import publiplots as pp


# ---------------------------------------------------------------------------
# Task 3 — Panel finalization for unfilled PanelImage
# ---------------------------------------------------------------------------

def test_panel_image_no_path_finalizes_to_unfilled():
    """Finalizing an unfilled PanelImage (path=None) leaves
    ``image_path=None`` AND ``embedded_figure=None`` on the Panel record."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    panel_b = canvas["B"]
    assert panel_b.kind == "image"
    assert panel_b.image_path is None
    assert panel_b.embedded_figure is None


def test_panel_image_with_path_finalizes_with_path(tmp_path):
    """Existing PR 5 contract: path-bearing PanelImage finalizes with a
    Path-typed ``image_path`` and ``embedded_figure=None``."""
    p = tmp_path / "schematic.svg"
    p.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="40mm" height="30mm" viewBox="0 0 40 30"/>'
    )
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", path=p, size=(70, 50)),
        pp.PanelAxes(label="B", size=("flex", 50)),
    )
    panel_a = canvas["A"]
    assert panel_a.kind == "image"
    assert isinstance(panel_a.image_path, Path)
    assert panel_a.embedded_figure is None


# ---------------------------------------------------------------------------
# Task 4 — Canvas.embed_figure
# ---------------------------------------------------------------------------

@pytest.fixture
def side_figure():
    """A small standalone matplotlib Figure to attach via embed_figure."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([1, 2, 3], [4, 5, 6])
    yield fig
    plt.close(fig)


def test_embed_figure_attaches_figure(side_figure):
    """Happy path: stage PanelImage(no path), embed_figure, panel sees it."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    canvas.embed_figure("B", side_figure)
    panel_b = canvas["B"]
    assert panel_b.embedded_figure is side_figure


def test_embed_figure_accepts_int_index(side_figure):
    """``embed_figure(idx, fig)`` resolves int via insertion-order lookup."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    canvas.embed_figure(1, side_figure)
    assert canvas["B"].embedded_figure is side_figure


def test_embed_figure_raises_on_axes_panel(side_figure):
    """embed_figure on a PanelAxes target raises TypeError with the panel
    label in the message."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    with pytest.raises(TypeError, match=r"embed_figure.*'A'.*kind='axes'"):
        canvas.embed_figure("A", side_figure)


def test_embed_figure_raises_on_text_panel(side_figure):
    """embed_figure on a PanelText also raises TypeError."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelText(label="A", text="caption", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    with pytest.raises(TypeError, match=r"embed_figure.*kind='text'"):
        canvas.embed_figure("A", side_figure)


def test_embed_figure_raises_on_double_embed(side_figure):
    """Calling embed_figure twice on the same panel raises RuntimeError."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    canvas.embed_figure("B", side_figure)
    with pytest.raises(RuntimeError, match=r"already has an embedded figure"):
        canvas.embed_figure("B", side_figure)


def test_embed_figure_raises_on_empty_canvas(side_figure):
    """embed_figure before any add_row raises (Canvas has no panels)."""
    canvas = pp.Canvas("cell-2col")
    with pytest.raises(RuntimeError, match=r"no rows yet|no panels yet"):
        canvas.embed_figure("B", side_figure)


def test_embed_figure_raises_on_unknown_label(side_figure):
    """embed_figure with unknown label raises KeyError."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    with pytest.raises(KeyError, match=r"no panel with label"):
        canvas.embed_figure("Z", side_figure)


def test_embed_figure_index_out_of_range(side_figure):
    """embed_figure with int index out of range raises KeyError."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    with pytest.raises(KeyError, match=r"out of range"):
        canvas.embed_figure(5, side_figure)


# ---------------------------------------------------------------------------
# Task 5 — render_figure_to_{pdf,svg}_bytes helpers
# ---------------------------------------------------------------------------

def _build_simple_figure(figsize=(2.0, 1.5)):
    """Helper: a deterministic side figure (no rng, no time, no env)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([1, 2, 3], [4, 5, 6])
    return fig


def test_render_figure_to_pdf_bytes_starts_with_pdf_marker():
    """The PDF bytes start with ``%PDF-`` (magic header)."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import (
        render_figure_to_pdf_bytes,
    )

    fig = _build_simple_figure()
    try:
        pdf_bytes = render_figure_to_pdf_bytes(fig)
    finally:
        plt.close(fig)
    assert pdf_bytes.startswith(b"%PDF-")


def test_render_figure_to_pdf_bytes_deterministic():
    """Two renders of the same Figure produce byte-identical PDFs."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import (
        render_figure_to_pdf_bytes,
    )

    fig1 = _build_simple_figure()
    fig2 = _build_simple_figure()
    try:
        b1 = render_figure_to_pdf_bytes(fig1)
        b2 = render_figure_to_pdf_bytes(fig2)
    finally:
        plt.close(fig1)
        plt.close(fig2)
    assert b1 == b2


def test_render_figure_to_pdf_bytes_pinned_metadata():
    """The PDF bytes carry the pinned ``/CreationDate`` + ``/Producer``
    literals (``D:20260101000000Z`` / ``publiplots-composer``)."""
    import io
    import matplotlib.pyplot as plt
    import pypdf
    from publiplots.composer.compositing._embed import (
        render_figure_to_pdf_bytes,
    )

    fig = _build_simple_figure()
    try:
        pdf_bytes = render_figure_to_pdf_bytes(fig)
    finally:
        plt.close(fig)
    md = pypdf.PdfReader(io.BytesIO(pdf_bytes)).metadata
    assert md.get("/CreationDate") == "D:20260101000000Z"
    assert md.get("/Producer") == "publiplots-composer"


def test_render_figure_to_svg_bytes_parses_via_lxml():
    """The SVG bytes parse cleanly via lxml.etree.fromstring."""
    import matplotlib.pyplot as plt
    from lxml import etree as _lxml_etree
    from publiplots.composer.compositing._embed import (
        render_figure_to_svg_bytes,
    )

    fig = _build_simple_figure()
    try:
        svg_bytes = render_figure_to_svg_bytes(fig)
    finally:
        plt.close(fig)
    root = _lxml_etree.fromstring(svg_bytes)
    assert root.tag.endswith("svg")
    assert root.get("viewBox") is not None


def test_render_figure_to_svg_bytes_deterministic():
    """Two renders of the same Figure produce byte-identical SVGs."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import (
        render_figure_to_svg_bytes,
    )

    fig1 = _build_simple_figure()
    fig2 = _build_simple_figure()
    try:
        b1 = render_figure_to_svg_bytes(fig1)
        b2 = render_figure_to_svg_bytes(fig2)
    finally:
        plt.close(fig1)
        plt.close(fig2)
    assert b1 == b2


def test_render_figure_to_svg_bytes_carries_default_dc_date():
    """The SVG bytes contain ``<dc:date>2026-01-01T00:00:00</dc:date>``."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import (
        render_figure_to_svg_bytes,
    )

    fig = _build_simple_figure()
    try:
        svg_bytes = render_figure_to_svg_bytes(fig)
    finally:
        plt.close(fig)
    text = svg_bytes.decode("utf-8")
    assert "<dc:date>2026-01-01T00:00:00</dc:date>" in text


def test_render_figure_to_pdf_bytes_pp_subplots_deterministic_100x():
    """Architect-required: 100× round-trip a ``pp.subplots``-built figure
    through render_figure_to_pdf_bytes; all 100 outputs byte-identical.

    The SubplotsAutoLayout reactor is allowed to run AT MOST ONCE per
    render. If the reactor measured differently each time (e.g. due to
    font cache state), 100 successive renders of the same figure would
    drift and this test would fail.
    """
    import matplotlib.pyplot as plt
    import publiplots as pp
    from publiplots.composer.compositing._embed import (
        render_figure_to_pdf_bytes,
    )

    rendered = []
    figs = []
    try:
        for _ in range(100):
            fig, ax = pp.subplots(axes_size=(40, 30))
            ax.plot([1, 2, 3], [4, 5, 6])
            figs.append(fig)
            rendered.append(render_figure_to_pdf_bytes(fig))
    finally:
        for f in figs:
            plt.close(f)

    first = rendered[0]
    for i, b in enumerate(rendered[1:], start=1):
        assert b == first, (
            f"render_figure_to_pdf_bytes is non-deterministic for "
            f"pp.subplots-built figures: render #{i} drifted from #0."
        )


def test_render_figure_to_svg_bytes_pp_subplots_deterministic_100x():
    """Same as PDF test, but for the SVG path."""
    import matplotlib.pyplot as plt
    import publiplots as pp
    from publiplots.composer.compositing._embed import (
        render_figure_to_svg_bytes,
    )

    rendered = []
    figs = []
    try:
        for _ in range(100):
            fig, ax = pp.subplots(axes_size=(40, 30))
            ax.plot([1, 2, 3], [4, 5, 6])
            figs.append(fig)
            rendered.append(render_figure_to_svg_bytes(fig))
    finally:
        for f in figs:
            plt.close(f)

    first = rendered[0]
    for i, b in enumerate(rendered[1:], start=1):
        assert b == first, (
            f"render_figure_to_svg_bytes is non-deterministic for "
            f"pp.subplots-built figures: render #{i} drifted from #0."
        )


# ---------------------------------------------------------------------------
# PR 6c Task 1 — Panel.embedded_figure_anchor + Canvas._panel_decoration_budget_mm
# ---------------------------------------------------------------------------

def test_panel_embedded_figure_anchor_default_is_figure():
    """Panel result dataclass defaults ``embedded_figure_anchor='figure'``
    so PR 6b's existing behavior is preserved when the new field is unset.
    """
    from publiplots.composer.panels import Panel
    panel = Panel(
        label="X", kind="image", ax=None,
        size_mm=(70.0, 50.0), bbox_mm=(0.0, 0.0, 70.0, 50.0),
    )
    assert panel.embedded_figure_anchor == "figure"


def test_panel_embedded_figure_anchor_axes_value():
    """The Panel field accepts ``'axes'`` as a value (frozen dataclass)."""
    from publiplots.composer.panels import Panel
    panel = Panel(
        label="X", kind="image", ax=None,
        size_mm=(70.0, 50.0), bbox_mm=(0.0, 0.0, 70.0, 50.0),
        embedded_figure_anchor="axes",
    )
    assert panel.embedded_figure_anchor == "axes"


def test_panel_decoration_budget_mm_panel_a_in_cell_2col():
    """For Panel A (left column, cell-2col), the budget on each side
    matches the canvas's reservation around its bbox.

    cell-2col: outer_pad=2, hpad=3, ylabel_space=10, right=2,
    title_space=5, xlabel_space=8.
    Panel A bbox=(12, 10, 70, 50); canvas=(171, 67); first row → vpad=0.
      - left:   x_left=12 (canvas edge at 0) → 12 mm = outer_pad+ylabel_space.
      - right:  panel B starts at x=97 (gap=15 between A's right edge 82
                and B's left edge 97).
      - top:    canvas top - panel top edge = 67 - 60 = 7 mm.
      - bottom: panel bottom - canvas bottom = 10 - 0 = 10 mm.
    """
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    panel_a = canvas["A"]
    budget = canvas._panel_decoration_budget_mm(panel_a)
    assert set(budget.keys()) == {"left", "right", "top", "bottom"}
    assert budget["left"] == pytest.approx(12.0, abs=1e-6)
    assert budget["right"] == pytest.approx(15.0, abs=1e-6)
    assert budget["top"] == pytest.approx(7.0, abs=1e-6)
    assert budget["bottom"] == pytest.approx(10.0, abs=1e-6)


def test_panel_decoration_budget_mm_panel_b_in_cell_2col():
    """Panel B (right column, cell-2col): budget mirrors A on left but
    sees the canvas right edge on the right.
      - left:   gap to A's right edge (15 mm).
      - right:  canvas right edge (4 mm = right + outer_pad).
      - top/bottom: same as A.
    """
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    panel_b = canvas["B"]
    budget = canvas._panel_decoration_budget_mm(panel_b)
    assert budget["left"] == pytest.approx(15.0, abs=1e-6)
    assert budget["right"] == pytest.approx(4.0, abs=1e-6)
    assert budget["top"] == pytest.approx(7.0, abs=1e-6)
    assert budget["bottom"] == pytest.approx(10.0, abs=1e-6)


def test_panel_decoration_budget_mm_two_panel_row_meets_at_hpad():
    """A's right budget + B's left budget == 2× the gap between them
    (each panel sees the full geometric gap as its budget; PR 6c is
    intentionally permissive — overlapping decorations would be
    diagnosed at render time, not pre-empted by halving the budget)."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    a_budget = canvas._panel_decoration_budget_mm(canvas["A"])
    b_budget = canvas._panel_decoration_budget_mm(canvas["B"])
    # Both panels see the same 15 mm gap on the side facing each other.
    assert a_budget["right"] == pytest.approx(b_budget["left"], abs=1e-6)


def test_panel_decoration_budget_mm_panel_image_kind_returns_same_shape():
    """PanelImage and PanelAxes get the same budget dict shape; for
    matched bboxes the values match too (the helper is panel-kind-
    agnostic — it only reads bbox_mm + canvas geometry)."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelImage(label="A", size=(70, 50)),
        pp.PanelAxes(label="B", size=(70, 50)),
    )
    a_image_budget = canvas._panel_decoration_budget_mm(canvas["A"])

    canvas2 = pp.Canvas("cell-2col")
    canvas2.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelAxes(label="B", size=(70, 50)),
    )
    a_axes_budget = canvas2._panel_decoration_budget_mm(canvas2["A"])

    # Same slot position → same budget regardless of panel kind.
    assert a_image_budget == a_axes_budget


# ---------------------------------------------------------------------------
# PR 6c Task 2 — Canvas.embed_figure anchor= kwarg
# ---------------------------------------------------------------------------

def test_embed_figure_anchor_default_is_figure(side_figure):
    """Calling embed_figure with no anchor= preserves PR 6b's behavior:
    Panel.embedded_figure_anchor=='figure'."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    canvas.embed_figure("B", side_figure)
    assert canvas["B"].embedded_figure_anchor == "figure"


def test_embed_figure_anchor_explicit_figure(side_figure):
    """Explicit anchor='figure' matches the default."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    canvas.embed_figure("B", side_figure, anchor="figure")
    assert canvas["B"].embedded_figure_anchor == "figure"


def test_embed_figure_anchor_axes_stored(side_figure):
    """anchor='axes' stores correctly on the Panel record."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    canvas.embed_figure("B", side_figure, anchor="axes")
    assert canvas["B"].embedded_figure_anchor == "axes"


def test_embed_figure_anchor_invalid_raises(side_figure):
    """anchor='invalid' raises ValueError naming the valid options."""
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70, 50)),
        pp.PanelImage(label="B", size=(70, 50)),
    )
    with pytest.raises(ValueError, match=r"anchor=.*invalid.*'figure'.*'axes'"):
        canvas.embed_figure("B", side_figure, anchor="invalid")


# ---------------------------------------------------------------------------
# PR 6c Task 3 — extract_side_axes_bbox + check_decoration_overflow helpers
# ---------------------------------------------------------------------------

def test_extract_side_axes_bbox_single_axes():
    """Happy path: single primary axes; returned bbox is positive,
    nested inside the figure's mediabox in pt."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import (
        extract_side_axes_bbox,
        _settle_subplots_auto_layout,
    )

    fig, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([1, 2, 3], [4, 5, 6])
    try:
        _settle_subplots_auto_layout(fig)
        left_pt, bottom_pt, w_pt, h_pt = extract_side_axes_bbox(fig)
    finally:
        plt.close(fig)
    fig_w_pt = 2.0 * 72.0
    fig_h_pt = 1.5 * 72.0
    assert 0 < left_pt < fig_w_pt
    assert 0 < bottom_pt < fig_h_pt
    assert 0 < w_pt < fig_w_pt
    assert 0 < h_pt < fig_h_pt
    # The axes-data rect cannot exceed the mediabox.
    assert left_pt + w_pt <= fig_w_pt + 1e-6
    assert bottom_pt + h_pt <= fig_h_pt + 1e-6


def test_extract_side_axes_bbox_returns_full_mediabox_when_no_axes():
    """Empty figure: degenerate fallback returns the full mediabox."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import extract_side_axes_bbox

    fig = plt.figure(figsize=(2.0, 1.5))
    try:
        left_pt, bottom_pt, w_pt, h_pt = extract_side_axes_bbox(fig)
    finally:
        plt.close(fig)
    assert left_pt == 0.0
    assert bottom_pt == 0.0
    assert w_pt == pytest.approx(2.0 * 72.0, abs=1e-6)
    assert h_pt == pytest.approx(1.5 * 72.0, abs=1e-6)


def test_extract_side_axes_bbox_multi_axes_uses_union_for_subplots_sharex():
    """``plt.subplots(nrows=2, sharex=True)``: the returned bbox unions
    BOTH rows. CRITICAL: this is the architect-flagged twin-vs-shared
    case where ``get_shared_x_axes`` would have collapsed both rows
    incorrectly. The position-rect dedup keeps disjoint shared-x rects
    separate."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import (
        extract_side_axes_bbox,
        _settle_subplots_auto_layout,
    )

    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(2.0, 1.5))
    axes[0].plot([1, 2, 3], [4, 5, 6])
    axes[1].plot([1, 2, 3], [3, 2, 1])
    try:
        _settle_subplots_auto_layout(fig)
        left_pt, bottom_pt, w_pt, h_pt = extract_side_axes_bbox(fig)
        # Single-axes baseline for comparison: just one of the two
        # axes' bounds.
        ax0_pos = axes[0].get_position()
        ax1_pos = axes[1].get_position()
    finally:
        plt.close(fig)
    fig_w_pt = 2.0 * 72.0
    fig_h_pt = 1.5 * 72.0
    expected_h_uu = (ax0_pos.y0 + ax0_pos.height) - ax1_pos.y0
    expected_h_pt = expected_h_uu * fig_h_pt
    # Union height should approximately equal the span from row-0-top
    # down to row-1-bottom (i.e. union covers both rows, not just one).
    assert h_pt == pytest.approx(expected_h_pt, abs=0.5)


def test_extract_side_axes_bbox_skips_twin_axes():
    """``ax.twinx()`` makes a NEW axes overlaying the primary axes (same
    position rect). The position-rect dedup collapses them; bbox area
    is NOT double-counted."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import (
        extract_side_axes_bbox,
        _settle_subplots_auto_layout,
    )

    fig, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([1, 2, 3], [4, 5, 6])
    twin = ax.twinx()
    twin.plot([1, 2, 3], [10, 20, 30])
    try:
        _settle_subplots_auto_layout(fig)
        bbox_with_twin = extract_side_axes_bbox(fig)
    finally:
        plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(2.0, 1.5))
    ax2.plot([1, 2, 3], [4, 5, 6])
    try:
        _settle_subplots_auto_layout(fig2)
        bbox_without_twin = extract_side_axes_bbox(fig2)
    finally:
        plt.close(fig2)
    # Twin overlays the same position rect → bbox dimensions match
    # the no-twin baseline (within float tolerance).
    for w, wo in zip(bbox_with_twin, bbox_without_twin):
        assert w == pytest.approx(wo, abs=0.5)


def test_extract_side_axes_bbox_disjoint_subplots_keeps_all():
    """``plt.subplots(nrows=2, ncols=2)``: 4 disjoint position rects
    → union covers all 4. Width union ≈ 2 column widths + gap; height
    union ≈ 2 row heights + gap."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import (
        extract_side_axes_bbox,
        _settle_subplots_auto_layout,
    )

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(3.0, 2.0))
    for ax in axes.ravel():
        ax.plot([1, 2, 3], [4, 5, 6])
    try:
        _settle_subplots_auto_layout(fig)
        left_pt, bottom_pt, w_pt, h_pt = extract_side_axes_bbox(fig)
    finally:
        plt.close(fig)
    fig_w_pt = 3.0 * 72.0
    fig_h_pt = 2.0 * 72.0
    # Union must cover most of the figure (disjoint 2x2 grid).
    assert w_pt > fig_w_pt * 0.5
    assert h_pt > fig_h_pt * 0.5


def test_check_decoration_overflow_passes_when_within_canvas_reservation():
    """Side fig with tiny decoration extents and roomy canvas budget:
    no exception."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import (
        check_decoration_overflow,
        extract_side_axes_bbox,
        _settle_subplots_auto_layout,
    )

    fig, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([1, 2, 3], [4, 5, 6])
    try:
        _settle_subplots_auto_layout(fig)
        axes_bbox_pt = extract_side_axes_bbox(fig)
        # Roomy budget: 100 mm on every side. No decoration would
        # overflow.
        check_decoration_overflow(
            fig, axes_bbox_pt,
            decoration_budget_mm={
                "left": 100.0, "right": 100.0,
                "top": 100.0, "bottom": 100.0,
            },
            panel_label="B",
            slot_size_mm=(70.0, 50.0),
        )
    finally:
        plt.close(fig)


def test_check_decoration_overflow_raises_on_oversized_ylabel():
    """Side fig with a deliberately large ylabel + zero left budget
    raises ComposerVectorError naming the 'left' side."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import (
        check_decoration_overflow,
        extract_side_axes_bbox,
        _settle_subplots_auto_layout,
    )
    from publiplots.composer.exceptions import ComposerVectorError

    fig, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([1, 2, 3], [4, 5, 6])
    ax.set_ylabel("y" * 50, fontsize=20)  # absurdly wide ylabel
    try:
        _settle_subplots_auto_layout(fig)
        axes_bbox_pt = extract_side_axes_bbox(fig)
        with pytest.raises(ComposerVectorError, match=r"'left'"):
            check_decoration_overflow(
                fig, axes_bbox_pt,
                decoration_budget_mm={
                    "left": 0.0, "right": 100.0,
                    "top": 100.0, "bottom": 100.0,
                },
                panel_label="B",
                slot_size_mm=(70.0, 50.0),
            )
    finally:
        plt.close(fig)


def test_check_decoration_overflow_message_contains_actionable_hints():
    """The error message names the panel label, the side, mm overflow,
    AND the actionable hint substrings (anchor='figure', PR 6d)."""
    import matplotlib.pyplot as plt
    from publiplots.composer.compositing._embed import (
        check_decoration_overflow,
        extract_side_axes_bbox,
        _settle_subplots_auto_layout,
    )
    from publiplots.composer.exceptions import ComposerVectorError

    fig, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([1, 2, 3], [4, 5, 6])
    ax.set_ylabel("y" * 50, fontsize=20)
    try:
        _settle_subplots_auto_layout(fig)
        axes_bbox_pt = extract_side_axes_bbox(fig)
        with pytest.raises(ComposerVectorError) as exc_info:
            check_decoration_overflow(
                fig, axes_bbox_pt,
                decoration_budget_mm={
                    "left": 0.0, "right": 100.0,
                    "top": 100.0, "bottom": 100.0,
                },
                panel_label="B",
                slot_size_mm=(70.0, 50.0),
            )
    finally:
        plt.close(fig)
    msg = str(exc_info.value)
    assert "'B'" in msg
    assert "left" in msg
    assert "anchor='figure'" in msg
    assert "PR 6d" in msg
