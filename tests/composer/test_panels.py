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


def test_panel_axes_label_must_be_str():
    with pytest.raises(TypeError, match="label must be"):
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


# ---------------------------------------------------------------------------
# PR 2: flex sizing — size=('flex', h_mm) is now valid
# ---------------------------------------------------------------------------

def test_panel_axes_size_accepts_flex_width():
    p = PanelAxes(label="A", size=("flex", 40.0))
    assert p.size == ("flex", 40.0)


def test_panel_axes_flex_width_height_must_still_be_positive_numeric():
    with pytest.raises(ValueError, match="positive"):
        PanelAxes(label="A", size=("flex", 0.0))
    with pytest.raises(ValueError, match="positive"):
        PanelAxes(label="A", size=("flex", -1.0))


def test_panel_axes_flex_height_must_be_numeric():
    with pytest.raises(ValueError, match="numeric"):
        PanelAxes(label="A", size=("flex", "flex"))


def test_panel_axes_flex_width_only_keyword_accepted():
    """The ONLY string accepted in the width slot is 'flex'. Anything else
    is still rejected as non-numeric."""
    with pytest.raises(ValueError, match="numeric|flex"):
        PanelAxes(label="A", size=("auto", 40.0))


def test_panel_axes_size_height_cannot_be_flex():
    """PR 2 does NOT add flex height (PR 3 might — for vfill policies).
    Flex is width-only in PR 2."""
    with pytest.raises(ValueError, match="numeric|flex"):
        PanelAxes(label="A", size=(70.0, "flex"))


# ---------------------------------------------------------------------------
# PR 2: label modes — None (auto), False (no label), str (verbatim)
# ---------------------------------------------------------------------------

def test_panel_axes_label_none_is_auto_slot():
    """label=None reserves an auto-letter slot; resolution happens in
    canvas.add_row() based on the canvas's abc=. PR 1 required str;
    PR 2 makes None valid."""
    p = PanelAxes(label=None, size=(70.0, 40.0))
    assert p.label is None


def test_panel_axes_label_false_is_no_label():
    """label=False explicitly suppresses any label rendering and is
    skipped from the auto-letter sequence."""
    p = PanelAxes(label=False, size=(70.0, 40.0))
    assert p.label is False


def test_panel_axes_label_str_unchanged():
    """The PR 1 contract for str labels is preserved verbatim."""
    p = PanelAxes(label="B.i", size=(70.0, 40.0))
    assert p.label == "B.i"


def test_panel_axes_label_other_types_rejected():
    """label must be None, False, or a str. int/float/list/etc. raise."""
    with pytest.raises(TypeError, match="label must be"):
        PanelAxes(label=123, size=(70.0, 40.0))
    with pytest.raises(TypeError, match="label must be"):
        PanelAxes(label=["A", "B"], size=(70.0, 40.0))


def test_panel_axes_label_true_rejected():
    """label=True isn't a meaningful state and we don't want users
    confusing it with abc=True (which lives on the Canvas, not the panel)."""
    with pytest.raises(TypeError, match="label must be"):
        PanelAxes(label=True, size=(70.0, 40.0))


# ---------------------------------------------------------------------------
# PR 2: label_style override at panel construction
# ---------------------------------------------------------------------------

def test_panel_axes_accepts_label_style_dict():
    p = PanelAxes(label="A", size=(70.0, 40.0), label_style={"loc": "ur", "size": 11})
    assert p.label_style == {"loc": "ur", "size": 11}


def test_panel_axes_label_style_defaults_to_none():
    p = PanelAxes(label="A", size=(70.0, 40.0))
    assert p.label_style is None


def test_panel_axes_label_style_must_be_mapping_or_none():
    with pytest.raises(TypeError, match="label_style"):
        PanelAxes(label="A", size=(70.0, 40.0), label_style="not-a-dict")
