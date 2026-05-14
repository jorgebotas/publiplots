"""Tests for canvas[label] indexing."""

import pytest

import publiplots as pp


def test_canvas_indexing_returns_panel_for_known_label():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    p = canvas["A"]
    assert p.label == "A"


def test_canvas_indexing_raises_keyerror_for_unknown_label():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    with pytest.raises(KeyError, match="no panel with label"):
        canvas["B"]


def test_canvas_indexing_raises_keyerror_before_add_row():
    """Indexing on a not-yet-finalized canvas — no panels yet."""
    canvas = pp.Canvas("custom", width=174.0)
    with pytest.raises(KeyError):
        canvas["A"]


def test_canvas_indexing_keyerror_lists_known_labels():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(40.0, 30.0)),
        pp.PanelAxes(label="C", size=(40.0, 30.0)),
    )
    with pytest.raises(KeyError, match=r"\['A', 'C'\]"):
        canvas["B"]
