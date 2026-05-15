"""Tests for canvas.add_row — geometry, figure creation, layout reactor."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerOverflowError


# ---------------------------------------------------------------------------
# add_row creates the figure + populates panels
# ---------------------------------------------------------------------------

def test_add_row_creates_figure():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    assert canvas.figure is not None


def test_add_row_populates_two_panels():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    a = canvas["A"]
    b = canvas["B"]
    assert a.label == "A" and a.kind == "axes"
    assert b.label == "B" and b.kind == "axes"


def test_add_row_panels_each_get_a_real_axes():
    """canvas[label].ax is a matplotlib.axes.Axes — not None."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    from matplotlib.axes import Axes
    assert isinstance(canvas["A"].ax, Axes)
    assert isinstance(canvas["B"].ax, Axes)
    assert canvas["A"].ax is not canvas["B"].ax


def test_add_row_single_panel():
    """A single panel still works — width = panel width + outer_pad×2 +
    ylabel + right reservations from rcParams."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(140.0, 50.0)))
    assert canvas["A"].size_mm == (140.0, 50.0)


def test_add_row_three_panels():
    canvas = pp.Canvas("custom", width=174.0)
    # 3 panels of width 50 + 2 hpads of 4mm + outer/ylabel/right reservations
    canvas.add_row(
        pp.PanelAxes(label="A", size=(40.0, 30.0)),
        pp.PanelAxes(label="B", size=(40.0, 30.0)),
        pp.PanelAxes(label="C", size=(40.0, 30.0)),
    )
    assert set(canvas._panels.keys()) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_add_row_zero_panels_raises():
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(ValueError, match="at least one panel"):
        canvas.add_row()


def test_add_row_duplicate_labels_raises():
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(ValueError, match="duplicate.*label"):
        canvas.add_row(
            pp.PanelAxes(label="A", size=(40.0, 30.0)),
            pp.PanelAxes(label="A", size=(40.0, 30.0)),
        )


def test_add_row_rejects_non_panelaxes():
    """PR 3 adds PanelGrid/PanelText; PR 1 only accepts PanelAxes."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(TypeError, match="PanelAxes"):
        canvas.add_row("not a panel")


def test_add_row_width_overflow_raises_with_dims():
    """Two 100mm panels in a 174mm canvas overflow; the error carries
    requested + available dims."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(ComposerOverflowError) as exc_info:
        canvas.add_row(
            pp.PanelAxes(label="A", size=(100.0, 40.0)),
            pp.PanelAxes(label="B", size=(100.0, 40.0)),
        )
    err = exc_info.value
    assert err.requested_mm > 174.0
    assert err.available_mm <= 174.0


# ---------------------------------------------------------------------------
# Geometry — mm precision
# ---------------------------------------------------------------------------

MM_TOL = 0.01  # 0.01 mm tolerance for figure-size assertions


def test_add_row_figure_width_equals_panels_plus_decorations():
    """In PR 1 the figure width = sum(panel widths) + decoration
    reservations, NOT the canvas width budget. (PR 2 adds 'flex'
    sizing to absorb the slack so the figure equals canvas width
    exactly.) With 2×70mm panels + rcParams defaults
    (outer_pad=2, ylabel=10, right=2, wspace=3):
    figure_width = 2+10+70+2+3+10+70+2+2 = 171 mm."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    w_mm, _ = canvas.figure_size_mm
    assert abs(w_mm - 171.0) < MM_TOL


def test_add_row_figure_width_never_exceeds_canvas_width():
    """The figure width must NEVER exceed the canvas budget — that's
    what overflow validation guarantees. With panels sized to barely
    fit (2×71mm = 142 + 31 decorations = 173 ≤ 174), the figure is
    173 mm wide, just inside the budget."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(71.0, 40.0)),
        pp.PanelAxes(label="B", size=(71.0, 40.0)),
    )
    w_mm, _ = canvas.figure_size_mm
    assert w_mm <= 174.0 + MM_TOL
    assert abs(w_mm - 173.0) < MM_TOL


def test_add_row_figure_height_grows_to_fit_panels_plus_decorations():
    """Height = panel max height + xlabel_space + title_space + outer_pad×2.
    With rcParams defaults: 40 + 8 (xlabel) + 5 (title) + 4 (2×outer_pad)
    = 57 mm, give or take. Decoration auto-measurement may inflate it
    further on first draw (we don't draw here, so this asserts the
    INITIAL figure size from rcParams defaults)."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    _, h_mm = canvas.figure_size_mm
    # Panel height + initial decoration reservations
    expected_initial = 40.0 + 8.0 + 5.0 + 2.0 + 2.0  # = 57 mm
    assert abs(h_mm - expected_initial) < MM_TOL


def test_add_row_panel_bbox_has_correct_width():
    """Each panel's bbox_mm width equals its declared size width."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(50.0, 40.0)),
    )
    a_bbox = canvas["A"].bbox_mm
    b_bbox = canvas["B"].bbox_mm
    assert abs(a_bbox[2] - 70.0) < MM_TOL  # bbox is (x, y, w, h)
    assert abs(b_bbox[2] - 50.0) < MM_TOL


def test_add_row_panel_a_is_left_of_panel_b():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelAxes(label="B", size=(70.0, 40.0)),
    )
    a_x = canvas["A"].bbox_mm[0]
    b_x = canvas["B"].bbox_mm[0]
    assert a_x < b_x


def test_add_row_panels_preserve_ordering_in_dict():
    """canvas._panels iterates in add_row order — important for PR 2's
    auto-letter sequencing."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="C", size=(40.0, 30.0)),
        pp.PanelAxes(label="A", size=(40.0, 30.0)),
        pp.PanelAxes(label="B", size=(40.0, 30.0)),
    )
    assert list(canvas._panels.keys()) == ["C", "A", "B"]


# ---------------------------------------------------------------------------
# PR 2: flex sizing geometry
# ---------------------------------------------------------------------------

def test_add_row_with_flex_panel_fills_canvas_width_exactly():
    """When ≥1 flex panel exists, figure_width equals canvas.width_mm
    exactly (modulo float noise) — the slack is absorbed."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(60.0, 40.0)),
        pp.PanelAxes(label="B", size=("flex", 40.0)),
    )
    w_mm, _ = canvas.figure_size_mm
    assert abs(w_mm - 174.0) < MM_TOL


def test_add_row_with_flex_panel_resolves_to_leftover_width():
    """One pinned 60mm panel + one flex panel in a 174mm canvas.
    Decorations: 2 + 10 + 60 + 2 + 3 + 10 + flex + 2 + 2 = 91 + flex
    Setting equal to 174: flex = 83mm."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(60.0, 40.0)),
        pp.PanelAxes(label="B", size=("flex", 40.0)),
    )
    a_w = canvas["A"].size_mm[0]
    b_w = canvas["B"].size_mm[0]
    assert abs(a_w - 60.0) < MM_TOL
    assert abs(b_w - 83.0) < MM_TOL


def test_add_row_two_flex_panels_split_leftover_equally():
    """Two flex panels in a 174mm canvas with no pinned panels.
    Decorations: 2 + 10 + flex + 2 + 3 + 10 + flex + 2 + 2 = 31 + 2*flex
    Setting equal to 174: each flex = 71.5mm."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=("flex", 40.0)),
        pp.PanelAxes(label="B", size=("flex", 40.0)),
    )
    a_w = canvas["A"].size_mm[0]
    b_w = canvas["B"].size_mm[0]
    assert abs(a_w - 71.5) < MM_TOL
    assert abs(b_w - 71.5) < MM_TOL


def test_add_row_flex_with_pinned_overflow_raises():
    """If pinned panels alone overflow the canvas, the flex resolver
    refuses to make flex panels go to ≤0 mm."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises((ComposerOverflowError, ValueError)) as exc_info:
        canvas.add_row(
            pp.PanelAxes(label="A", size=(150.0, 40.0)),
            pp.PanelAxes(label="B", size=("flex", 40.0)),
        )
    # Either error type is acceptable; the message should mention flex
    # or non-positive.
    msg = str(exc_info.value).lower()
    assert "flex" in msg or "non-positive" in msg or "exceed" in msg


def test_add_row_pinned_only_still_uses_pr1_overflow_path():
    """If no flex panels, the overflow check + advisor from Task 1 fires."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(ComposerOverflowError) as exc_info:
        canvas.add_row(
            pp.PanelAxes(label="A", size=(100.0, 40.0)),
            pp.PanelAxes(label="B", size=(100.0, 40.0)),
        )
    msg = str(exc_info.value)
    # Task 1 advisor mentioned scaling — verify it's still in the message.
    assert "multiply" in msg.lower() or "0.7" in msg
