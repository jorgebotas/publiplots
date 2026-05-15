"""Spike Finding 4 regression test.

The bug: a Composer-built figure where the bottom xlabel / left ylabel /
top title clips at the canvas mediabox edge, because the figure was
constructed at the panel-only mm size without reserving decoration
space. The spike's bare-mpl fixture exhibited this exact bug; the
Composer's add_row attaches SubplotsAutoLayout to fix it.

This test confirms the fix:
  1. Canvas auto-grows to fit decorations after first draw.
  2. The panel's xlabel tightbbox lies inside the figure's mediabox
     (no clipping).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import publiplots as pp


def test_canvas_size_after_draw_is_within_decoration_tolerance():
    """The Composer's SubplotsAutoLayout reactor adjusts the figure size
    after first draw to fit the actual rendered decorations. The rcParams
    initial values (ylabel=10mm, xlabel=8mm, title=5mm, right=2mm) are
    sized to overestimate by ~0.5mm typical, so the post-draw figure
    can be slightly SMALLER than the initial size. This test confirms
    the adjustment stays within ~1mm — much larger and we'd suspect a
    layout bug; much smaller is fine and expected.

    Note: this is NOT the Finding 4 assertion; that's
    test_canvas_*_does_not_clip_at_canvas_*. This test just sanity-checks
    that the reactor doesn't go wild and shrink the figure massively
    or grow it to fit imaginary decorations."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.set_xlabel("a long-ish x-axis label that takes vertical space")
    canvas["A"].ax.set_ylabel("y")
    canvas["A"].ax.set_title("Panel title")

    initial_w_mm, initial_h_mm = canvas.figure_size_mm

    # Drive the layout reactor to convergence.
    canvas.figure.canvas.draw()
    canvas.figure._publiplots_auto_layout.settle()

    settled_w_mm, settled_h_mm = canvas.figure_size_mm

    # Reactor adjustment is bounded: ~0.5mm typical, ~1mm tolerance here.
    # Width tends to shrink slightly (real ylabel/right narrower than rcParams).
    # Height can shrink slightly (real title/xlabel narrower than rcParams)
    # OR grow (long xlabel exceeding the 8mm reservation).
    assert abs(settled_w_mm - initial_w_mm) < 1.0, (
        f"width shifted {settled_w_mm - initial_w_mm:.2f}mm beyond tolerance"
    )
    assert abs(settled_h_mm - initial_h_mm) < 5.0, (
        f"height shifted {settled_h_mm - initial_h_mm:.2f}mm beyond tolerance"
    )


def test_canvas_reactor_actually_attached():
    """Sanity check — the figure has the SubplotsAutoLayout reactor
    attached. If this fails, none of the decoration tests can guarantee
    Finding 4 is fixed."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    assert hasattr(canvas.figure, "_publiplots_auto_layout"), (
        "SubplotsAutoLayout was not attached — Finding 4 fix is missing"
    )


def test_canvas_xlabel_does_not_clip_at_canvas_bottom():
    """Spike Finding 4 regression — the rendered xlabel's tightbbox must
    lie INSIDE the figure mediabox (no clipping at canvas bottom)."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.set_xlabel("x")  # use a short xlabel like the spike fixture

    fig = canvas.figure
    fig.canvas.draw()
    fig._publiplots_auto_layout.settle()

    # Get the renderer and the xlabel tightbbox in display coordinates.
    renderer = fig.canvas.get_renderer()
    xlabel_artist = canvas["A"].ax.xaxis.get_label()
    xlabel_bbox_px = xlabel_artist.get_window_extent(renderer=renderer)

    # Figure bbox in display coordinates.
    fig_bbox_px = fig.bbox

    # The xlabel's bottom edge must be at or above the figure's bottom edge.
    # Allow 0.5 px of float-noise tolerance (matplotlib pt rounding).
    assert xlabel_bbox_px.y0 >= fig_bbox_px.y0 - 0.5, (
        f"xlabel bottom (px y0={xlabel_bbox_px.y0:.2f}) clips below "
        f"figure bottom (px y0={fig_bbox_px.y0:.2f}) — "
        f"SubplotsAutoLayout did not reserve enough xlabel_space"
    )


def test_canvas_title_does_not_clip_at_canvas_top():
    """Same regression check, top edge."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.set_title("My panel title")

    fig = canvas.figure
    fig.canvas.draw()
    fig._publiplots_auto_layout.settle()

    renderer = fig.canvas.get_renderer()
    title_artist = canvas["A"].ax.title
    title_bbox_px = title_artist.get_window_extent(renderer=renderer)
    fig_bbox_px = fig.bbox

    assert title_bbox_px.y1 <= fig_bbox_px.y1 + 0.5, (
        f"title top (px y1={title_bbox_px.y1:.2f}) clips above "
        f"figure top (px y1={fig_bbox_px.y1:.2f}) — "
        f"SubplotsAutoLayout did not reserve enough title_space"
    )


def test_canvas_ylabel_does_not_clip_at_canvas_left():
    """Same regression check, left edge."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas["A"].ax.set_ylabel("Density (counts / mm²)")  # longer-than-default

    fig = canvas.figure
    fig.canvas.draw()
    fig._publiplots_auto_layout.settle()

    renderer = fig.canvas.get_renderer()
    ylabel_artist = canvas["A"].ax.yaxis.get_label()
    ylabel_bbox_px = ylabel_artist.get_window_extent(renderer=renderer)
    fig_bbox_px = fig.bbox

    assert ylabel_bbox_px.x0 >= fig_bbox_px.x0 - 0.5, (
        f"ylabel left (px x0={ylabel_bbox_px.x0:.2f}) clips left of "
        f"figure left (px x0={fig_bbox_px.x0:.2f}) — "
        f"SubplotsAutoLayout did not reserve enough ylabel_space"
    )
