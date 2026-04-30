"""Tests for pp.subplots() and its supporting components."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

SUBPLOT_KEYS = [
    "subplots.title_space",
    "subplots.xlabel_space",
    "subplots.ylabel_space",
    "subplots.right",
    "subplots.hspace",
    "subplots.wspace",
    "subplots.outer_pad",
]


def test_subplots_rcparams_keys_exist():
    for key in SUBPLOT_KEYS:
        assert key in pp.rcParams, f"missing rcParam: {key}"


def test_subplots_rcparams_publication_defaults():
    pp.set_publication_style()
    try:
        assert pp.rcParams["subplots.title_space"] == 5
        assert pp.rcParams["subplots.xlabel_space"] == 8
        assert pp.rcParams["subplots.ylabel_space"] == 10
        assert pp.rcParams["subplots.right"] == 2
        assert pp.rcParams["subplots.hspace"] == 8
        assert pp.rcParams["subplots.wspace"] == 10
        assert pp.rcParams["subplots.outer_pad"] == 2
    finally:
        pp.reset_style()


def test_subplots_rcparams_notebook_defaults():
    pp.set_notebook_style()
    try:
        assert pp.rcParams["subplots.title_space"] == 8
        assert pp.rcParams["subplots.xlabel_space"] == 12
        assert pp.rcParams["subplots.ylabel_space"] == 14
        assert pp.rcParams["subplots.right"] == 2
        assert pp.rcParams["subplots.hspace"] == 12
        assert pp.rcParams["subplots.wspace"] == 14
        assert pp.rcParams["subplots.outer_pad"] == 3
    finally:
        pp.reset_style()


# ---------------------------------------------------------------------------
# FigureLayout — pure geometry
# ---------------------------------------------------------------------------
from publiplots.layout.figure_layout import FigureLayout


def _make_layout(nrows=1, ncols=1, **overrides):
    defaults = dict(
        nrows=nrows, ncols=ncols,
        axes_size=(50.0, 30.0),
        title_space=5.0, xlabel_space=8.0, ylabel_space=10.0, right=2.0,
        hspace=8.0, wspace=10.0, outer_pad=2.0, legend_column=0.0,
    )
    defaults.update(overrides)
    return FigureLayout(**defaults)


def test_figure_layout_single_cell_size():
    layout = _make_layout()
    W, H = layout.figure_size()
    # W = 2 + (10 + 50 + 2) + 0 + 2 = 66
    # H = 2 + (5 + 30 + 8) + 2 = 47
    assert W == pytest.approx(66.0)
    assert H == pytest.approx(47.0)


def test_figure_layout_2x3_size_matches_formula():
    layout = _make_layout(nrows=2, ncols=3)
    W, H = layout.figure_size()
    # W = 2 + 3*(10+50+2) + 2*10 + 0 + 2 = 2 + 186 + 20 + 2 = 210
    # H = 2 + 2*(5+30+8) + 1*8 + 2 = 2 + 86 + 8 + 2 = 98
    assert W == pytest.approx(210.0)
    assert H == pytest.approx(98.0)


def test_figure_layout_legend_column_adds_width_only():
    base = _make_layout(nrows=2, ncols=3)
    wide = _make_layout(nrows=2, ncols=3, legend_column=30.0)
    W0, H0 = base.figure_size()
    W1, H1 = wide.figure_size()
    assert W1 == pytest.approx(W0 + 30.0)
    assert H1 == pytest.approx(H0)


def test_figure_layout_axes_position_is_deterministic():
    layout = _make_layout(nrows=2, ncols=3)
    p_first = layout.axes_position(0, 0)
    p_again = layout.axes_position(0, 0)
    assert p_first == p_again


def test_figure_layout_axes_positions_dont_overlap():
    layout = _make_layout(nrows=2, ncols=3)
    rects = [layout.axes_position(r, c) for r in range(2) for c in range(3)]
    # Check pairwise non-overlap (rectangles as (x0, y0, w, h) in figure fractions)
    for i, (x0a, y0a, wa, ha) in enumerate(rects):
        for j, (x0b, y0b, wb, hb) in enumerate(rects):
            if i == j:
                continue
            x1a, y1a = x0a + wa, y0a + ha
            x1b, y1b = x0b + wb, y0b + hb
            overlaps = not (x1a <= x0b or x1b <= x0a or y1a <= y0b or y1b <= y0a)
            assert not overlaps, f"cells {i} and {j} overlap"


def test_figure_layout_with_updated_reservations_preserves_axes_size():
    layout = _make_layout()
    updated = layout.with_updated_reservations(title_space=20.0, xlabel_space=15.0)
    assert updated.axes_size == layout.axes_size
    assert updated.title_space == 20.0
    assert updated.xlabel_space == 15.0
    # Untouched fields unchanged
    assert updated.ylabel_space == layout.ylabel_space


def test_figure_layout_row_zero_is_top():
    """Row 0 should have the HIGHEST y0 in figure fractions (matplotlib convention)."""
    layout = _make_layout(nrows=3, ncols=1)
    y0_top = layout.axes_position(0, 0)[1]
    y0_mid = layout.axes_position(1, 0)[1]
    y0_bot = layout.axes_position(2, 0)[1]
    assert y0_top > y0_mid > y0_bot