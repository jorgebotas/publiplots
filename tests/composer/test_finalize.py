"""Tests for lazy figure finalization in PR 3 multi-row Canvas."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import publiplots as pp


def test_canvas_figure_is_none_before_access():
    """After add_row but before any figure access, _figure is still None."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    # PR 3 makes figure creation LAZY — _figure is None until access.
    assert canvas._figure is None


def test_canvas_figure_property_triggers_finalization():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    fig = canvas.figure  # accessor triggers
    assert fig is not None
    assert canvas._figure is fig


def test_canvas_indexing_triggers_finalization():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    p = canvas["A"]  # indexing triggers
    assert canvas._figure is not None
    assert p.ax is not None


def test_canvas_savefig_triggers_finalization(tmp_path):
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.savefig(tmp_path / "fig.png")  # savefig triggers
    assert canvas._figure is not None
    assert (tmp_path / "fig.png").exists()


def test_canvas_finalize_explicit_call():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.finalize()
    assert canvas._figure is not None


def test_canvas_finalize_idempotent():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.finalize()
    fig1 = canvas._figure
    canvas.finalize()  # second call must be a no-op
    assert canvas._figure is fig1


def test_canvas_add_row_after_finalize_raises():
    """Once finalized, add_row is no longer accepted."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.finalize()
    with pytest.raises(RuntimeError, match="finalize"):
        canvas.add_row(pp.PanelAxes(label="B", size=(70.0, 40.0)))


def test_canvas_savefig_before_add_row_still_raises():
    """An empty canvas (no rows) can't be saved."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(RuntimeError, match="add_row"):
        canvas.savefig("/tmp/nope.png")


def test_canvas_figure_size_mm_is_none_before_any_row():
    """figure_size_mm returns None on a fresh canvas with no rows
    staged. Once a row is staged, accessing figure_size_mm triggers
    lazy finalization (so it has a value).
    """
    canvas = pp.Canvas("custom", width=174.0)
    # No rows yet — figure_size_mm returns None.
    assert canvas.figure_size_mm is None
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    # Accessing figure_size_mm triggers lazy finalization.
    assert canvas.figure_size_mm is not None
