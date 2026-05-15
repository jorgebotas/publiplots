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


# ---------------------------------------------------------------------------
# Canvas integration — text panel renders a hidden axes with text
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")


def test_panel_text_canvas_kind_is_text():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelText(label="E", text="hello", size=(60.0, 20.0)))
    assert canvas["E"].kind == "text"


def test_panel_text_canvas_renders_text_artist():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelText(label="E", text="n = 1,234", size=(60.0, 20.0)))
    ax = canvas["E"].ax
    text_artists = [t for t in ax.texts if t.get_text() == "n = 1,234"]
    assert len(text_artists) == 1


def test_panel_text_axes_has_no_visible_spines():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelText(label="E", text="x", size=(60.0, 20.0)))
    ax = canvas["E"].ax
    # PanelText hides spines + ticks via set_axis_off
    for spine in ax.spines.values():
        assert not spine.get_visible()


def test_panel_text_supports_mathtext():
    """Mathtext (e.g., r"$\\alpha$") is rendered by matplotlib's text engine.
    Smoke test: ensure rendering doesn't raise."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelText(label="E", text=r"$\alpha = 0.05$", size=(60.0, 20.0)))
    canvas.figure.canvas.draw()  # forces text rendering
    # If mathtext failed, draw would have raised.
    text_artists = [t for t in canvas["E"].ax.texts if "alpha" in t.get_text()]
    assert len(text_artists) == 1


def test_panel_text_text_kw_overrides_apply():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelText(label="E", text="x", size=(60.0, 20.0),
                                 text_kw={"fontsize": 14, "color": "red"}))
    text_artists = [t for t in canvas["E"].ax.texts if t.get_text() == "x"]
    assert len(text_artists) == 1
    artist = text_artists[0]
    assert artist.get_fontsize() == 14
    assert artist.get_color() == "red"


def test_panel_text_in_multi_row():
    """Mix PanelAxes and PanelText across rows."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(140.0, 40.0)))
    canvas.add_row(pp.PanelText(label="E", text="caption", size=(140.0, 15.0)))
    assert canvas["A"].kind == "axes"
    assert canvas["E"].kind == "text"


def test_panel_text_in_row_with_panel_axes():
    """PanelAxes + PanelText in the same row, side-by-side."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(
        pp.PanelAxes(label="A", size=(70.0, 40.0)),
        pp.PanelText(label="E", text="caption", size=("flex", 40.0)),
    )
    assert canvas["A"].kind == "axes"
    assert canvas["E"].kind == "text"
