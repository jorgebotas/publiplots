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


# ---------------------------------------------------------------------------
# Regression: PanelGrid accepted at add_row, rejected at finalize
# ---------------------------------------------------------------------------

def test_panel_grid_accepted_at_add_row_and_finalizes():
    """PR3 Task 7 contract: PanelGrid is a valid input to add_row AND
    finalization succeeds (lifting the Task 6 NotImplementedError).

    Regression test for the AttributeError that was happening when
    add_row's eager flex/overflow check tried to read PanelGrid.size
    (which doesn't exist — PanelGrid has size_mm property)."""
    canvas = pp.Canvas("custom", width=174.0)
    # Should NOT raise here — PanelGrid is a valid input.
    canvas.add_row(pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0)))
    # Finalization should also succeed (Task 7 lifted the boundary).
    canvas.finalize()
    assert canvas["C"].kind == "axesgrid"


def test_panel_grid_in_multirow_with_panel_axes():
    """Regression: a row mixing PanelAxes + PanelGrid should also pass
    add_row's validation. (Originally written when Task 7 hadn't landed;
    kept as a staging-only check that the eager-path bug stays fixed.)"""
    canvas = pp.Canvas("custom", width=174.0)
    # add_row should NOT raise — both panel types are valid inputs.
    canvas.add_row(pp.PanelAxes(label="A", size=(70.0, 40.0)))
    canvas.add_row(pp.PanelGrid(label="C", shape=(1, 3), axes_size=(40.0, 30.0)))
    # Verify both rows are staged.
    assert len(canvas._rows) == 2


def test_panel_grid_overflow_in_add_row():
    """A PanelGrid wider than the canvas should raise ComposerOverflowError
    at add_row time (not finalize), same as PanelAxes does."""
    from publiplots.composer.exceptions import ComposerOverflowError
    canvas = pp.Canvas("custom", width=174.0)
    # PanelGrid: 5 cols of 50mm + 4 wspaces of 2mm = 258mm — overflows 174mm.
    with pytest.raises(ComposerOverflowError):
        canvas.add_row(
            pp.PanelGrid(label="C", shape=(2, 5), axes_size=(50.0, 30.0))
        )


# ---------------------------------------------------------------------------
# Canvas integration — the panel actually creates a sub-grid of axes
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")


def test_panel_grid_canvas_axes_attribute_returns_2d_array():
    """canvas[label].axes returns a 2D numpy array of Axes for a grid panel."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0)))
    panel = canvas["C"]
    import numpy as np
    from matplotlib.axes import Axes
    assert isinstance(panel.axes, np.ndarray)
    assert panel.axes.shape == (2, 3)
    for row in panel.axes:
        for ax in row:
            assert isinstance(ax, Axes)


def test_panel_grid_kind_is_axesgrid():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0)))
    assert canvas["C"].kind == "axesgrid"


def test_panel_grid_ax_is_none_use_axes_instead():
    """For PanelGrid, .ax is None — use .axes (2D array) instead."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0)))
    panel = canvas["C"]
    assert panel.ax is None
    assert panel.axes is not None


def test_panel_grid_outer_size_matches_size_mm_property():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(1, 3), axes_size=(50.0, 30.0)))
    panel = canvas["C"]
    # Panel.size_mm reflects the OUTER mm rect of the grid panel
    # (not the per-cell size).
    # 3 cols of 50mm + 2 wspaces of 2mm = 154mm wide; 1 row of 30mm = 30mm tall.
    assert panel.size_mm == (154.0, 30.0)


def test_panel_grid_inner_axes_have_correct_per_cell_size():
    """Each inner axes occupies axes_size mm; verify by reading the
    per-axes bbox in figure-fraction and converting back to mm."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(1, 3), axes_size=(50.0, 30.0)))
    canvas.finalize()
    fig_w_in, fig_h_in = canvas.figure.get_size_inches()
    for ax in canvas["C"].axes.flat:
        bbox = ax.get_position()
        ax_w_mm = bbox.width * fig_w_in * 25.4
        ax_h_mm = bbox.height * fig_h_in * 25.4
        assert abs(ax_w_mm - 50.0) < 0.5  # mm precision (publiplots default)
        assert abs(ax_h_mm - 30.0) < 0.5


def test_panel_grid_sharex_propagates_to_inner_axes():
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelGrid(label="C", shape=(2, 3), axes_size=(40.0, 30.0),
                                 sharex=True))
    axes = canvas["C"].axes
    # sharex=True: all 6 axes share the same xaxis with axes[0,0].
    primary = axes[0, 0].get_shared_x_axes()
    for r in range(2):
        for c in range(3):
            if r == 0 and c == 0:
                continue
            assert primary.joined(axes[0, 0], axes[r, c])


def test_panel_grid_in_multi_row():
    """Mix PanelAxes and PanelGrid across rows."""
    canvas = pp.Canvas("custom", width=174.0)
    canvas.add_row(pp.PanelAxes(label="A", size=(140.0, 40.0)))
    canvas.add_row(pp.PanelGrid(label="C", shape=(1, 3), axes_size=(40.0, 30.0)))
    assert canvas["A"].kind == "axes"
    assert canvas["C"].kind == "axesgrid"
