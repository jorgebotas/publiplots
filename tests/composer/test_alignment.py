"""Tests for canvas.align() — explicit alignment overrides."""

import matplotlib
matplotlib.use("Agg")
import pytest

import publiplots as pp
from publiplots.composer.exceptions import ComposerAlignmentError


MM_TOL = 0.01


# ---------------------------------------------------------------------------
# Construction + recording
# ---------------------------------------------------------------------------

def test_canvas_align_method_exists():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    canvas.align(["A", "B"], edge="left")
    assert len(canvas._alignments) == 1


def test_canvas_align_unknown_edge_raises():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    with pytest.raises(ValueError, match="edge"):
        canvas.align(["A"], edge="invalid")


def test_canvas_align_unknown_mode_raises():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    with pytest.raises(ValueError, match="mode"):
        canvas.align(["A"], edge="left", mode="weird")


def test_canvas_align_unknown_panel_raises():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    with pytest.raises(KeyError, match="panel"):
        canvas.align(["A", "Z"], edge="left")


def test_canvas_align_after_finalize_raises():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.finalize()
    with pytest.raises(RuntimeError, match="finalize"):
        canvas.align(["A"], edge="left")


# ---------------------------------------------------------------------------
# Alignment math — left edge
# ---------------------------------------------------------------------------

def test_align_left_makes_panels_share_left_edge():
    """Two panels in different rows already share their left edges by
    topology (both rows start at outer_pad + ylabel = 12mm). The explicit
    align('left') call should be a no-op shift (within MM_TOL)."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    canvas.align(["A", "B"], edge="left")
    canvas.finalize()
    a_x = canvas["A"].bbox_mm[0]
    b_x = canvas["B"].bbox_mm[0]
    assert abs(a_x - b_x) < MM_TOL


# ---------------------------------------------------------------------------
# Anchor — non-default
# ---------------------------------------------------------------------------

def test_align_anchor_specifies_which_edge_wins():
    """anchor='B' makes B's edge the reference; A shifts to match.
    In this trivial test both panels start with the same x, so the
    behavior is observably equivalent to default — the test just
    verifies the kwarg is accepted and the result still satisfies the
    alignment."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    canvas.align(["A", "B"], edge="left", anchor="B")
    canvas.finalize()
    a_x = canvas["A"].bbox_mm[0]
    b_x = canvas["B"].bbox_mm[0]
    assert abs(a_x - b_x) < MM_TOL


def test_align_anchor_must_be_one_of_panels_in_request():
    """anchor must be one of the panels passed to align()."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    with pytest.raises(ValueError, match="anchor"):
        canvas.align(["A", "B"], edge="left", anchor="C")


# ---------------------------------------------------------------------------
# Slot-boundary check — alignment can't push panels outside their slots
# ---------------------------------------------------------------------------

def test_align_shift_outside_slot_raises():
    """If aligning would require shifting panel B's content beyond what
    its slot can absorb, raise ComposerAlignmentError.

    Construct a configuration where panel B's slot has very limited
    horizontal slack: row 1 has a single 140mm panel B in a 174mm canvas
    (slot fully consumes the row's available width minus decorations).
    Aligning B's right edge to A's right edge in row 0 (which is at
    x=82mm) would require shifting B 60mm left from its current right
    edge (at x=152mm) — far outside its 12mm-of-slack slot.
    """
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(140.0, 40.0)))
    canvas.align(["A", "B"], edge="right", anchor="A")
    with pytest.raises(ComposerAlignmentError) as exc_info:
        canvas.finalize()
    err = exc_info.value
    # Either A or B should be in the offending panels list (the side
    # that needs to shift).
    assert "B" in err.panels or "A" in err.panels


# ---------------------------------------------------------------------------
# Mode — axes vs tight (PR 3 ships axes mode primarily)
# ---------------------------------------------------------------------------

def test_align_mode_axes_default():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    canvas.align(["A", "B"], edge="left")  # mode='axes' is default
    canvas.finalize()
    assert abs(canvas["A"].bbox_mm[0] - canvas["B"].bbox_mm[0]) < MM_TOL


def test_align_mode_tight_accepted():
    """mode='tight' is accepted in PR 3 (uses axes-bbox path internally;
    full tightbbox measurement lands in PR 4.5)."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.align(["A"], edge="left", mode="tight")
    canvas.finalize()
    # No exception is the primary assertion.
    assert canvas["A"] is not None


# ---------------------------------------------------------------------------
# Multiple alignment calls — accumulate
# ---------------------------------------------------------------------------

def test_multiple_align_calls_all_recorded():
    """Multiple canvas.align() calls accumulate; each is applied at
    finalize time."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))
    canvas.align(["A", "B"], edge="left")
    canvas.align(["A", "B"], edge="right")
    assert len(canvas._alignments) == 2
    canvas.finalize()
    # Both panels share both edges => they have the SAME width too.
    a_x, a_y, a_w, a_h = canvas["A"].bbox_mm
    b_x, b_y, b_w, b_h = canvas["B"].bbox_mm
    assert abs(a_x - b_x) < MM_TOL
    assert abs(a_w - b_w) < MM_TOL  # widths follow trivially when both edges share
