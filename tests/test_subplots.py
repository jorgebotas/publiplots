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


# ---------------------------------------------------------------------------
# SubplotsAutoLayout — draw-event hook
# ---------------------------------------------------------------------------
from publiplots.layout.auto_layout import SubplotsAutoLayout

MM2INCH = 1 / 25.4


def _make_fig_with_layout(layout, ncells=None):
    """Build a matplotlib figure with axes placed per the layout. Returns (fig, axes_matrix)."""
    W, H = layout.figure_size()
    fig = plt.figure(figsize=(W * MM2INCH, H * MM2INCH), layout=None)
    axes = []
    for r in range(layout.nrows):
        row = []
        for c in range(layout.ncols):
            ax = fig.add_axes(layout.axes_position(r, c))
            row.append(ax)
        axes.append(row)
    return fig, axes


def test_auto_layout_grows_title_space_for_title():
    layout = _make_layout(nrows=1, ncols=1, title_space=1.0)  # deliberately too small
    fig, axes = _make_fig_with_layout(layout)
    ax = axes[0][0]
    ax.set_title("A title that needs more vertical room than 1 mm")
    reactor = SubplotsAutoLayout(fig, layout, locked=set())
    fig.canvas.draw()
    # With bidirectional updates, the figure-height proxy is noisy (other
    # sides may shrink), so assert directly on the reservation we care about.
    assert reactor._layout.title_space > 1.0, (
        f"title_space should have grown past 1.0 mm, got {reactor._layout.title_space:.2f}"
    )


def test_auto_layout_preserves_axes_size_after_resize():
    declared_w, declared_h = 50.0, 30.0
    layout = _make_layout(axes_size=(declared_w, declared_h))
    fig, axes = _make_fig_with_layout(layout)
    ax = axes[0][0]
    ax.set_title("A title")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    SubplotsAutoLayout(fig, layout, locked=set())
    fig.canvas.draw()

    # Actual axes bbox in mm
    pos = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    ax_w_mm = pos.width * fig_w_in / MM2INCH
    ax_h_mm = pos.height * fig_h_in / MM2INCH
    assert ax_w_mm == pytest.approx(declared_w, abs=0.5)
    assert ax_h_mm == pytest.approx(declared_h, abs=0.5)


def test_auto_layout_locked_side_not_remeasured():
    layout = _make_layout(title_space=25.0)  # locked oversize reservation
    fig, axes = _make_fig_with_layout(layout)
    axes[0][0].set_title("tiny")  # way less than 25 mm
    reactor = SubplotsAutoLayout(fig, layout, locked={"title_space"})
    fig.canvas.draw()
    # Locked reservation must stay pinned regardless of measured decoration size.
    assert reactor._layout.title_space == 25.0, (
        f"locked title_space should remain 25.0 mm, got {reactor._layout.title_space:.2f}"
    )


def test_auto_layout_no_hook_when_all_sides_locked():
    layout = _make_layout()
    fig, _ = _make_fig_with_layout(layout)
    all_sides = {"title_space", "xlabel_space", "ylabel_space", "right"}
    reactor = SubplotsAutoLayout(fig, layout, locked=all_sides)
    # No draw-event callback should be connected
    assert reactor._cid is None


def test_auto_layout_second_draw_no_change_within_threshold():
    layout = _make_layout()
    fig, axes = _make_fig_with_layout(layout)
    axes[0][0].set_title("stable")
    SubplotsAutoLayout(fig, layout, locked=set())
    fig.canvas.draw()
    size_after_first = fig.get_size_inches().copy()
    fig.canvas.draw()
    size_after_second = fig.get_size_inches()
    assert np.allclose(size_after_first, size_after_second, atol=1e-4)


def test_auto_layout_attaches_layout_to_figure():
    layout = _make_layout()
    fig, _ = _make_fig_with_layout(layout)
    SubplotsAutoLayout(fig, layout, locked=set())
    assert getattr(fig, "_publiplots_layout", None) is layout