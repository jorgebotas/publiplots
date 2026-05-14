"""Tests for PanelAxes dataclass + the Panel result type."""

import pytest

from publiplots.composer.panels import Panel, PanelAxes


# ---------------------------------------------------------------------------
# PanelAxes — input dataclass passed to canvas.add_row(*panels)
# ---------------------------------------------------------------------------

def test_panel_axes_basic_construction():
    p = PanelAxes(label="A", size=(70.0, 40.0))
    assert p.label == "A"
    assert p.size == (70.0, 40.0)


def test_panel_axes_label_required_in_pr1():
    """PR 2 will introduce auto-letter sequencing (label=None → 'A','B',...).
    PR 1 requires an explicit label string. This test will be UPDATED in
    PR 2 to allow label=None; pin to required-string for now."""
    with pytest.raises(TypeError):
        PanelAxes(size=(70.0, 40.0))  # no label


def test_panel_axes_label_must_be_str():
    with pytest.raises(TypeError, match="label must be a string"):
        PanelAxes(label=123, size=(70.0, 40.0))


def test_panel_axes_size_required():
    with pytest.raises(TypeError):
        PanelAxes(label="A")  # no size


def test_panel_axes_size_must_be_2_tuple():
    with pytest.raises(ValueError, match="size must be a 2-tuple"):
        PanelAxes(label="A", size=80.0)
    with pytest.raises(ValueError, match="size must be a 2-tuple"):
        PanelAxes(label="A", size=(80.0,))
    with pytest.raises(ValueError, match="size must be a 2-tuple"):
        PanelAxes(label="A", size=(80.0, 40.0, 10.0))


def test_panel_axes_size_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        PanelAxes(label="A", size=(0.0, 40.0))
    with pytest.raises(ValueError, match="positive"):
        PanelAxes(label="A", size=(80.0, -1.0))


def test_panel_axes_size_must_be_numeric():
    with pytest.raises(ValueError, match="numeric"):
        PanelAxes(label="A", size=("flex", 40.0))


def test_panel_axes_size_coerces_to_floats():
    """Integers are accepted and stored as floats (mm precision matters)."""
    p = PanelAxes(label="A", size=(70, 40))
    assert p.size == (70.0, 40.0)
    assert isinstance(p.size[0], float)
    assert isinstance(p.size[1], float)


# ---------------------------------------------------------------------------
# Panel — result type returned by canvas[label]
# ---------------------------------------------------------------------------

def test_panel_exposes_label_and_kind():
    """Panel is a frozen dataclass-like view; label/kind are read-only."""
    # Use a sentinel ax (None) for unit-test purposes; integration tests
    # in test_indexing.py use a real Axes via canvas[label].
    p = Panel(
        label="A",
        kind="axes",
        ax=None,
        size_mm=(70.0, 40.0),
        bbox_mm=(5.0, 5.0, 70.0, 40.0),
    )
    assert p.label == "A"
    assert p.kind == "axes"
    assert p.size_mm == (70.0, 40.0)
    assert p.bbox_mm == (5.0, 5.0, 70.0, 40.0)


def test_panel_axes_attribute_accessible_for_axes_kind():
    """For kind='axes', .ax returns whatever the canvas stored (a real
    Axes in production; None in this unit test)."""
    p = Panel(label="A", kind="axes", ax=None,
              size_mm=(70.0, 40.0), bbox_mm=(5.0, 5.0, 70.0, 40.0))
    # .ax is just an attribute — no special accessor logic in PR 1.
    assert p.ax is None  # The integration tests check the real-Axes case.


def test_panel_is_immutable():
    """Panel is a frozen dataclass — users shouldn't mutate it."""
    p = Panel(label="A", kind="axes", ax=None,
              size_mm=(70.0, 40.0), bbox_mm=(5.0, 5.0, 70.0, 40.0))
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError or AttributeError
        p.label = "B"
