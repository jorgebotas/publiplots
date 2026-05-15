"""Tests for PanelText dataclass — PR 3 introduction."""

import pytest

import publiplots as pp
from publiplots.composer.panels import PanelText


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------

def test_panel_text_basic_construction():
    p = PanelText(label="A", text="hello", size=(40.0, 20.0))
    assert p.label == "A"
    assert p.text == "hello"
    assert p.size == (40.0, 20.0)
    assert p.text_kw == {}
    assert p.label_style is None


def test_panel_text_text_required():
    with pytest.raises(TypeError):
        PanelText(label="A", size=(40.0, 20.0))


def test_panel_text_text_must_be_str():
    with pytest.raises(TypeError, match="text must be a str"):
        PanelText(label="A", text=123, size=(40.0, 20.0))


def test_panel_text_label_modes_inherit_panelaxes_contract():
    """label=None / False / str — same contract as PanelAxes."""
    p_none = PanelText(label=None, text="x", size=(40.0, 20.0))
    p_false = PanelText(label=False, text="x", size=(40.0, 20.0))
    p_str = PanelText(label="A", text="x", size=(40.0, 20.0))
    assert p_none.label is None
    assert p_false.label is False
    assert p_str.label == "A"


def test_panel_text_label_true_rejected():
    """Same bool-True trap as PanelAxes."""
    with pytest.raises(TypeError, match="label must be"):
        PanelText(label=True, text="x", size=(40.0, 20.0))


def test_panel_text_size_must_be_2_tuple_of_positive_numerics():
    with pytest.raises(ValueError, match="size must be a 2-tuple"):
        PanelText(label="A", text="x", size=20.0)
    with pytest.raises(ValueError, match="positive"):
        PanelText(label="A", text="x", size=(0.0, 20.0))


def test_panel_text_size_flex_width_accepted():
    """PanelText supports flex sizing the same way PanelAxes does."""
    p = PanelText(label="A", text="x", size=("flex", 20.0))
    assert p.size == ("flex", 20.0)


def test_panel_text_text_kw_must_be_mapping_or_none():
    p = PanelText(label="A", text="x", size=(40.0, 20.0), text_kw={"fontsize": 12})
    assert p.text_kw == {"fontsize": 12}
    with pytest.raises(TypeError, match="text_kw"):
        PanelText(label="A", text="x", size=(40.0, 20.0), text_kw="not-a-dict")


def test_panel_text_supports_label_style():
    """Same per-panel label_style override as PanelAxes."""
    p = PanelText(label="A", text="x", size=(40.0, 20.0),
                  label_style={"loc": "ur"})
    assert p.label_style == {"loc": "ur"}


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------

def test_panel_text_exported_at_top_level():
    assert hasattr(pp, "PanelText")
    p = pp.PanelText(label="A", text="x", size=(40.0, 20.0))
    assert isinstance(p, PanelText)
