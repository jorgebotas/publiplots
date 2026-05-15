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
