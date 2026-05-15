"""Tests for multi-row canvas geometry — PR 3 introduction."""

import pytest

import publiplots as pp
from publiplots.composer._layout import compute_canvas_geometry


# ---------------------------------------------------------------------------
# compute_canvas_geometry — pure math, no matplotlib
# ---------------------------------------------------------------------------

def test_single_row_geometry_matches_pr1():
    """Single-row case must match what add_row produced in PR 1+2."""
    g = compute_canvas_geometry(
        rows=[{
            "panel_widths_mm": (70.0, 70.0),
            "row_height_mm": 40.0,
            "vpad_mm": 0.0,
        }],
        canvas_width_mm=174.0,
        outer_pad=2.0,
        ylabel_space=10.0,
        right=2.0,
        wspace=3.0,
        title_space=5.0,
        xlabel_space=8.0,
    )
    # Width: 2+10+70+2+3+10+70+2+2 = 171
    # Height: 2+5+40+8+2 = 57
    assert abs(g.canvas_width_mm - 171.0) < 0.01
    assert abs(g.canvas_height_mm - 57.0) < 0.01
    # Panel positions
    assert len(g.row_axes_rects_mm) == 1
    assert len(g.row_axes_rects_mm[0]) == 2  # 2 panels in row 0


def test_two_row_geometry_stacks_correctly():
    """Two rows: heights sum + vpad on the second row."""
    g = compute_canvas_geometry(
        rows=[
            {"panel_widths_mm": (70.0, 70.0), "row_height_mm": 40.0, "vpad_mm": 0.0},
            {"panel_widths_mm": (70.0, 70.0), "row_height_mm": 30.0, "vpad_mm": 4.0},
        ],
        canvas_width_mm=174.0,
        outer_pad=2.0,
        ylabel_space=10.0,
        right=2.0,
        wspace=3.0,
        title_space=5.0,
        xlabel_space=8.0,
    )
    # Width same as single-row: 171mm
    assert abs(g.canvas_width_mm - 171.0) < 0.01
    # Height: 2 (outer top)
    #         + 5 (title row 0) + 40 (row 0) + 8 (xlabel row 0)
    #         + 4 (vpad row 1)
    #         + 5 (title row 1) + 30 (row 1) + 8 (xlabel row 1)
    #         + 2 (outer bottom)
    #       = 104
    assert abs(g.canvas_height_mm - 104.0) < 0.01


def test_heterogeneous_columns_per_row():
    """Row 0 has 2 panels of width 70; row 1 has 1 panel of width 150."""
    g = compute_canvas_geometry(
        rows=[
            {"panel_widths_mm": (70.0, 70.0), "row_height_mm": 40.0, "vpad_mm": 0.0},
            {"panel_widths_mm": (150.0,), "row_height_mm": 30.0, "vpad_mm": 4.0},
        ],
        canvas_width_mm=174.0,
        outer_pad=2.0,
        ylabel_space=10.0,
        right=2.0,
        wspace=3.0,
        title_space=5.0,
        xlabel_space=8.0,
    )
    # Width is the MAX of any single row's width:
    #   row 0: 2+10+70+2+3+10+70+2+2 = 171
    #   row 1: 2+10+150+2+2 = 166
    # canvas width = max(171, 166) = 171
    assert abs(g.canvas_width_mm - 171.0) < 0.01

    # Row 1's single panel is centered in the canvas (or left-justified —
    # we'll lock the convention here): per spec, justify='start' is default,
    # so the panel sits at the LEFT (after outer_pad + ylabel_space).
    row1_rects = g.row_axes_rects_mm[1]
    assert len(row1_rects) == 1
    x_mm, y_mm, w_mm, h_mm = row1_rects[0]
    # Left edge should be 2 (outer_pad) + 10 (ylabel) = 12 mm
    assert abs(x_mm - 12.0) < 0.01
    assert abs(w_mm - 150.0) < 0.01


def test_panel_positions_have_correct_y_coords_top_down():
    """y_mm uses bottom-left origin (matplotlib convention). Row 0 (top)
    has the LARGEST y; row 1 (bottom) has the SMALLEST y."""
    g = compute_canvas_geometry(
        rows=[
            {"panel_widths_mm": (70.0, 70.0), "row_height_mm": 40.0, "vpad_mm": 0.0},
            {"panel_widths_mm": (70.0, 70.0), "row_height_mm": 30.0, "vpad_mm": 4.0},
        ],
        canvas_width_mm=174.0,
        outer_pad=2.0,
        ylabel_space=10.0,
        right=2.0,
        wspace=3.0,
        title_space=5.0,
        xlabel_space=8.0,
    )
    row0_y = g.row_axes_rects_mm[0][0][1]
    row1_y = g.row_axes_rects_mm[1][0][1]
    # Row 0 is on top → its y is larger.
    assert row0_y > row1_y


def test_zero_rows_raises():
    with pytest.raises(ValueError, match="at least one row"):
        compute_canvas_geometry(
            rows=[],
            canvas_width_mm=174.0,
            outer_pad=2.0, ylabel_space=10.0, right=2.0, wspace=3.0,
            title_space=5.0, xlabel_space=8.0,
        )


# ---------------------------------------------------------------------------
# PR 3: multi-row Canvas integration
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MM_TOL = 0.01


def test_add_row_called_twice_succeeds():
    """PR 3 lifts PR 1's NotImplementedError on multi-call add_row."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    # Triggers finalization via panel access:
    a = canvas["A"]
    b = canvas["B"]
    assert a.label == "A" and b.label == "B"


def test_two_row_canvas_figure_height_grows():
    """Two rows of 40mm panels — figure height should be roughly
    2 * (panel + xlabel + title) + outer_pad×2 + vpad."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    w_mm, h_mm = canvas.figure_size_mm
    # Per the geometry helper:
    # height = 2 (outer top)
    #        + 5 (title) + 40 (row 0) + 8 (xlabel)
    #        + vpad (default 4) + 5 + 40 + 8
    #        + 2 (outer bottom)
    # Total: 114
    assert h_mm > 100  # rough lower bound


def test_canvas_indexing_int_spans_rows():
    """canvas[0] is the FIRST panel by insertion order (across rows).
    canvas[2] is the third panel even if it's in a different row."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    canvas.add_row(pp.PanelAxes(label="C", size=(140.0, 30.0)))
    assert canvas[0].label == "A"
    assert canvas[1].label == "B"
    assert canvas[2].label == "C"


def test_canvas_abc_sequencing_continues_across_rows():
    """abc='upper' produces 'A','B','C',... across all rows in
    insertion order (row 0 first, then row 1, etc.)."""
    canvas = pp.Canvas("cell-2col", abc="upper")
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    canvas.add_row(
        pp.PanelAxes(label=None, size=(60.0, 40.0)),
    )
    assert canvas[0].label == "A"
    assert canvas[1].label == "B"
    assert canvas[2].label == "C"


def test_canvas_two_rows_panels_have_correct_y_ordering():
    """Row 0 panel sits HIGHER on the canvas than row 1 panel.
    bbox_mm uses bottom-left origin, so row 0's y > row 1's y."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    a_y = canvas["A"].bbox_mm[1]
    b_y = canvas["B"].bbox_mm[1]
    assert a_y > b_y


def test_canvas_heterogeneous_row_widths():
    """Row 0: 2 panels of 60mm; Row 1: 1 panel of 130mm.
    No errors, geometry sane."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(60.0, 40.0)),
        pp.PanelAxes(label="B", size=(60.0, 40.0)),
    )
    canvas.add_row(pp.PanelAxes(label="C", size=(130.0, 30.0)))
    assert canvas["A"].size_mm == (60.0, 40.0)
    assert canvas["C"].size_mm == (130.0, 30.0)
