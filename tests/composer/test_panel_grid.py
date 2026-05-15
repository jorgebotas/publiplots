"""Tests for PanelGrid dataclass — PR 3 introduction.

Tests the input record only. Actual axes-grid construction is exercised
by tests/composer/test_multi_row.py + test_add_row.py after Task 6 lands
the canvas refactor.
"""

import pytest

import publiplots as pp
from publiplots.composer.panels import PanelGrid


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------

def test_panel_grid_basic_construction():
    p = PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0))
    assert p.label == "C"
    assert p.shape == (2, 3)
    assert p.axes_size == (40.0, 30.0)
    assert p.sharex is False
    assert p.sharey is False
    assert p.hspace == 2.0
    assert p.wspace == 2.0


def test_panel_grid_shape_must_be_2_tuple_of_positive_ints():
    with pytest.raises(ValueError, match="shape must be"):
        PanelGrid(label="C", shape=(2,), axes_size=(40.0, 30.0))
    with pytest.raises(ValueError, match="shape must be"):
        PanelGrid(label="C", shape=(0, 3), axes_size=(40.0, 30.0))
    with pytest.raises(ValueError, match="shape must be"):
        PanelGrid(label="C", shape=(2, -1), axes_size=(40.0, 30.0))


def test_panel_grid_axes_size_must_be_2_tuple_of_positive_floats():
    """axes_size in PanelGrid is per-CELL; flex is NOT supported here
    (would require resolving inside the panel's slot)."""
    with pytest.raises(ValueError, match="axes_size|numeric"):
        PanelGrid(label="C", shape=(2, 3), axes_size=("flex", 30.0))
    with pytest.raises(ValueError, match="positive"):
        PanelGrid(label="C", shape=(2, 3), axes_size=(0.0, 30.0))


def test_panel_grid_sharex_sharey_validated():
    """sharex/sharey accept bool or 'all'/'row'/'col'/'none' — same as
    pp.subplots."""
    PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), sharex=True)
    PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), sharex="row")
    PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), sharey="col")
    with pytest.raises(ValueError, match="share"):
        PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), sharex="invalid")


def test_panel_grid_hspace_wspace_must_be_non_negative():
    with pytest.raises(ValueError, match="hspace"):
        PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), hspace=-1.0)
    with pytest.raises(ValueError, match="wspace"):
        PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0), wspace=-1.0)


def test_panel_grid_label_modes_match_panelaxes_contract():
    PanelGrid(label=None, shape=(2, 3), axes_size=(40.0, 30.0))
    PanelGrid(label=False, shape=(2, 3), axes_size=(40.0, 30.0))
    with pytest.raises(TypeError, match="label must be"):
        PanelGrid(label=True, shape=(2, 3), axes_size=(40.0, 30.0))


def test_panel_grid_size_mm_property_computed_from_shape_and_axes_size():
    """The panel's outer mm rect = (cols*w + (cols-1)*wspace,
                                   rows*h + (rows-1)*hspace)."""
    p = PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0),
                  hspace=2.0, wspace=2.0)
    # 3 cols of 40mm + 2 wspaces of 2mm = 124mm
    # 2 rows of 30mm + 1 hspace of 2mm = 62mm
    assert p.size_mm == (124.0, 62.0)


def test_panel_grid_size_mm_with_default_spacing():
    p = PanelGrid(label="C", shape=(1, 3), axes_size=(50.0, 30.0))
    # default wspace=2.0, hspace=2.0; 3 cells of 50 + 2 wspaces = 154mm
    assert p.size_mm == (154.0, 30.0)


def test_panel_grid_supports_label_style():
    p = PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0),
                  label_style={"loc": "ur"})
    assert p.label_style == {"loc": "ur"}


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------

def test_panel_grid_exported_at_top_level():
    assert hasattr(pp, "PanelGrid")
    p = pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0))
    assert isinstance(p, PanelGrid)
