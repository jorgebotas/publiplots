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


def test_subplots_rcparams_defaults():
    """publiplots is publication-first; subplots defaults are baked into
    PUBLIPLOTS_RCPARAMS with no separate style mode."""
    assert pp.rcParams["subplots.title_space"] == 5
    assert pp.rcParams["subplots.xlabel_space"] == 8
    assert pp.rcParams["subplots.ylabel_space"] == 10
    assert pp.rcParams["subplots.right"] == 2
    assert pp.rcParams["subplots.hspace"] == 3
    assert pp.rcParams["subplots.wspace"] == 3
    assert pp.rcParams["subplots.outer_pad"] == 2


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
    # Broadcast scalar reservations to tuples for FigureLayout
    for side in ("title_space", "xlabel_space"):
        v = defaults[side]
        if not isinstance(v, tuple):
            defaults[side] = (float(v),) * defaults["nrows"]
    for side in ("ylabel_space", "right"):
        v = defaults[side]
        if not isinstance(v, tuple):
            defaults[side] = (float(v),) * defaults["ncols"]
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
    updated = layout.with_updated_reservations(title_space=(20.0,), xlabel_space=(15.0,))
    assert updated.axes_size == layout.axes_size
    assert updated.title_space == (20.0,)
    assert updated.xlabel_space == (15.0,)
    # Untouched fields unchanged
    assert updated.ylabel_space == layout.ylabel_space


def test_figure_layout_row_zero_is_top():
    """Row 0 should have the HIGHEST y0 in figure fractions (matplotlib convention)."""
    layout = _make_layout(nrows=3, ncols=1)
    y0_top = layout.axes_position(0, 0)[1]
    y0_mid = layout.axes_position(1, 0)[1]
    y0_bot = layout.axes_position(2, 0)[1]
    assert y0_top > y0_mid > y0_bot


def _make_tuple_layout(nrows=1, ncols=1, **overrides):
    """Like _make_layout but accepts tuple reservations directly."""
    defaults = dict(
        nrows=nrows, ncols=ncols,
        axes_size=(50.0, 30.0),
        title_space=tuple([5.0] * nrows),
        xlabel_space=tuple([8.0] * nrows),
        ylabel_space=tuple([10.0] * ncols),
        right=tuple([2.0] * ncols),
        hspace=4.0, wspace=4.0, outer_pad=2.0, legend_column=0.0,
    )
    defaults.update(overrides)
    return FigureLayout(**defaults)


def test_figure_layout_per_row_title_space_changes_position():
    small = _make_tuple_layout(nrows=2, ncols=1,
                                title_space=(5.0, 5.0),
                                xlabel_space=(8.0, 8.0))
    big = _make_tuple_layout(nrows=2, ncols=1,
                              title_space=(20.0, 5.0),
                              xlabel_space=(8.0, 8.0))
    y0_top_small = small.axes_position(0, 0)[1]
    y0_top_big = big.axes_position(0, 0)[1]
    assert y0_top_big < y0_top_small


def test_figure_layout_per_col_right_is_cumulative():
    uniform = _make_tuple_layout(nrows=1, ncols=3, right=(2.0, 2.0, 2.0))
    asymmetric = _make_tuple_layout(nrows=1, ncols=3, right=(2.0, 2.0, 30.0))
    w_uniform, _ = uniform.figure_size()
    w_asym, _ = asymmetric.figure_size()
    assert w_asym == pytest.approx(w_uniform + 28.0)


def test_figure_layout_with_updated_reservations_accepts_tuples():
    layout = _make_tuple_layout(nrows=2, ncols=1)
    updated = layout.with_updated_reservations(title_space=(12.0, 6.0))
    assert updated.title_space == (12.0, 6.0)
    assert updated.xlabel_space == layout.xlabel_space
    assert updated.axes_size == layout.axes_size


def test_figure_layout_wrong_length_title_space_rejected():
    with pytest.raises(ValueError, match="title_space"):
        FigureLayout(
            nrows=2, ncols=1,
            axes_size=(50.0, 30.0),
            title_space=(5.0, 5.0, 5.0),
            xlabel_space=(8.0, 8.0),
            ylabel_space=(10.0,), right=(2.0,),
            hspace=4.0, wspace=4.0, outer_pad=2.0, legend_column=0.0,
        )


def test_figure_layout_wrong_length_ylabel_space_rejected():
    with pytest.raises(ValueError, match="ylabel_space"):
        FigureLayout(
            nrows=1, ncols=2,
            axes_size=(50.0, 30.0),
            title_space=(5.0,), xlabel_space=(8.0,),
            ylabel_space=(10.0, 10.0, 10.0),
            right=(2.0, 2.0),
            hspace=4.0, wspace=4.0, outer_pad=2.0, legend_column=0.0,
        )


def test_figure_layout_rejects_scalar_reservations():
    with pytest.raises((ValueError, TypeError)):
        FigureLayout(
            nrows=2, ncols=1,
            axes_size=(50.0, 30.0),
            title_space=5.0,
            xlabel_space=(8.0, 8.0),
            ylabel_space=(10.0,), right=(2.0,),
            hspace=4.0, wspace=4.0, outer_pad=2.0, legend_column=0.0,
        )


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
    assert reactor._layout.title_space[0] > 1.0, (
        f"title_space[0] should have grown past 1.0 mm, got {reactor._layout.title_space[0]:.2f}"
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
    assert reactor._layout.title_space == (25.0,), (
        f"locked title_space should remain (25.0,), got {reactor._layout.title_space}"
    )


def test_auto_layout_no_hook_when_all_sides_locked():
    layout = _make_layout()
    fig, _ = _make_fig_with_layout(layout)
    all_sides = {"title_space", "xlabel_space", "ylabel_space", "right", "legend_column"}
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


def test_auto_layout_right_is_per_column():
    """Adding a wide artist to one column should only grow that column's right."""
    layout = _make_layout(nrows=1, ncols=3, right=2.0)
    fig, axes = _make_fig_with_layout(layout)
    # Attach a wide text artist to the rightmost axes only — it inflates
    # that axes' tightbbox beyond the spine.
    axes[0][2].text(
        1.3, 0.5, "hanging text", transform=axes[0][2].transAxes,
    )
    reactor = SubplotsAutoLayout(fig, layout, locked=set())
    fig.canvas.draw()
    assert reactor._layout.right[2] > reactor._layout.right[0] + 5.0, (
        f"right[2] should exceed right[0] by > 5 mm after the text, got "
        f"right[0]={reactor._layout.right[0]:.1f}, right[2]={reactor._layout.right[2]:.1f}"
    )


def test_auto_layout_title_space_is_per_row():
    """Set a title on only one row; only that row's title_space grows."""
    layout = _make_layout(nrows=2, ncols=1)
    fig, axes = _make_fig_with_layout(layout)
    axes[0][0].set_title("Top row title")
    # axes[1][0] has no title.
    reactor = SubplotsAutoLayout(fig, layout, locked=set())
    fig.canvas.draw()
    assert reactor._layout.title_space[0] > reactor._layout.title_space[1] + 2.0, (
        f"title_space[0] should exceed title_space[1] by > 2 mm, got "
        f"{reactor._layout.title_space[0]:.1f} vs {reactor._layout.title_space[1]:.1f}"
    )


# ---------------------------------------------------------------------------
# pp.subplots() — public API
# ---------------------------------------------------------------------------
from matplotlib.axes import Axes as _Axes
from matplotlib.figure import Figure as _Figure


def test_subplots_scalar_axes_size_coerced_to_tuple():
    fig, ax = pp.subplots(axes_size=30)
    assert fig._publiplots_layout.axes_size == (30.0, 30.0)


def test_subplots_axes_size_none_uses_rcparams_default():
    fig, _ = pp.subplots()
    assert fig._publiplots_layout.axes_size == (70.0, 50.0)


def test_subplots_rejects_figsize_kwarg():
    with pytest.raises(TypeError, match="axes_size"):
        pp.subplots(axes_size=(50, 30), figsize=(5, 3))


def test_subplots_warns_on_layout_engine_kwarg():
    with pytest.warns(UserWarning, match="publiplots manages layout"):
        fig, ax = pp.subplots(axes_size=(50, 30), constrained_layout=True)
    assert fig.get_layout_engine() is None


def test_subplots_disables_layout_engine():
    fig, ax = pp.subplots(axes_size=(50, 30))
    assert fig.get_layout_engine() is None


def test_subplots_squeeze_returns_scalar_for_1x1():
    fig, ax = pp.subplots(axes_size=(50, 30))
    assert isinstance(ax, _Axes)


def test_subplots_returns_1d_array_for_single_row():
    fig, axes = pp.subplots(1, 3, axes_size=(50, 30))
    assert axes.shape == (3,)


def test_subplots_returns_1d_array_for_single_col():
    fig, axes = pp.subplots(3, 1, axes_size=(50, 30))
    assert axes.shape == (3,)


def test_subplots_returns_2d_array_for_grid():
    fig, axes = pp.subplots(2, 3, axes_size=(50, 30))
    assert axes.shape == (2, 3)


def test_subplots_attaches_figure_layout_to_fig():
    fig, _ = pp.subplots(2, 3, axes_size=(50, 30))
    from publiplots.layout.figure_layout import FigureLayout
    assert isinstance(fig._publiplots_layout, FigureLayout)
    assert fig._publiplots_layout.nrows == 2
    assert fig._publiplots_layout.ncols == 3


def test_subplots_validates_nrows():
    with pytest.raises(ValueError, match="nrows"):
        pp.subplots(nrows=0, ncols=1, axes_size=(50, 30))


def test_subplots_validates_ncols():
    with pytest.raises(ValueError, match="ncols"):
        pp.subplots(nrows=1, ncols=0, axes_size=(50, 30))


def test_subplots_validates_axes_size_scalar():
    with pytest.raises(ValueError, match="axes_size"):
        pp.subplots(axes_size=-5)


def test_subplots_validates_axes_size_tuple():
    with pytest.raises(ValueError, match="axes_size"):
        pp.subplots(axes_size=(50, 0))


def test_subplots_validates_negative_reservation():
    with pytest.raises(ValueError, match="title_space"):
        pp.subplots(axes_size=(50, 30), title_space=-1.0)


def test_subplots_rejects_legend_column_kwarg():
    with pytest.raises(TypeError, match="legend_column"):
        pp.subplots(axes_size=(50, 30), legend_column=30)


def test_subplots_sharex_true_shares_all():
    fig, axes = pp.subplots(2, 3, axes_size=(50, 30), sharex=True)
    # sharex=True -> every axes shares with (0,0)
    shared = axes[0, 0].get_shared_x_axes()
    for r in range(2):
        for c in range(3):
            assert shared.joined(axes[0, 0], axes[r, c])


def test_subplots_sharex_row_shares_within_row_only():
    fig, axes = pp.subplots(2, 3, axes_size=(50, 30), sharex="row")
    shared_top = axes[0, 0].get_shared_x_axes()
    # Same row is shared
    assert shared_top.joined(axes[0, 0], axes[0, 2])
    # Different rows are NOT shared
    assert not shared_top.joined(axes[0, 0], axes[1, 0])


def test_subplots_all_locked_skips_hook():
    # Even with all four per-side reservations locked via kwargs,
    # legend_column is still auto-measured (width-awareness, Task 5),
    # so the draw-event hook must remain attached. Users cannot lock
    # legend_column from pp.subplots(); it is always auto.
    fig, ax = pp.subplots(
        axes_size=(50, 30),
        title_space=5, xlabel_space=8, ylabel_space=10, right=2,
    )
    assert fig._publiplots_auto_layout._cid is not None


def test_subplots_any_auto_side_attaches_hook():
    fig, ax = pp.subplots(
        axes_size=(50, 30),
        title_space=5, xlabel_space=8, ylabel_space=10,
        # right left as auto
    )
    assert fig._publiplots_auto_layout._cid is not None


def test_subplots_works_with_legend_builder_after_auto_resize():
    """pp.legend() on a pp.subplots axes should follow the auto-layout resize."""
    from publiplots.utils.legend import create_legend_handles
    fig, ax = pp.subplots(axes_size=(60, 40))
    ax.set_title("A")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    handles = create_legend_handles(labels=["A"], colors=["#5d83c3"],
                                    alpha=0.2, linewidth=1.0)
    builder = pp.legend(ax, auto=False)
    builder.add_legend(handles=handles, label="group")
    fig.canvas.draw()
    # Simple sanity: axes bbox in mm matches declared size within tolerance
    pos = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    ax_w_mm = pos.width * fig_w_in / MM2INCH
    ax_h_mm = pos.height * fig_h_in / MM2INCH
    assert ax_w_mm == pytest.approx(60.0, abs=0.5)
    assert ax_h_mm == pytest.approx(40.0, abs=0.5)


# ---------------------------------------------------------------------------
# pp.subplots() — scalar/tuple coercion
# ---------------------------------------------------------------------------


def test_subplots_scalar_reservation_broadcasts_to_nrows():
    fig, _ = pp.subplots(2, 3, axes_size=(50, 30), title_space=8)
    assert fig._publiplots_layout.title_space == (8.0, 8.0)


def test_subplots_scalar_reservation_broadcasts_to_ncols():
    fig, _ = pp.subplots(2, 3, axes_size=(50, 30), right=5)
    assert fig._publiplots_layout.right == (5.0, 5.0, 5.0)


def test_subplots_tuple_reservation_preserved():
    fig, _ = pp.subplots(2, 3, axes_size=(50, 30), title_space=(12, 6))
    assert fig._publiplots_layout.title_space == (12.0, 6.0)


def test_subplots_wrong_length_title_space_raises():
    with pytest.raises(ValueError, match="title_space"):
        pp.subplots(2, 3, axes_size=(50, 30), title_space=(12, 6, 3))


def test_subplots_wrong_length_ylabel_space_raises():
    with pytest.raises(ValueError, match="ylabel_space"):
        pp.subplots(2, 3, axes_size=(50, 30), ylabel_space=(10, 10))


def test_subplots_default_reservations_broadcast_to_tuple():
    fig, _ = pp.subplots(2, 3, axes_size=(50, 30))
    layout = fig._publiplots_layout
    assert len(layout.title_space) == 2
    assert len(layout.xlabel_space) == 2
    assert len(layout.ylabel_space) == 3
    assert len(layout.right) == 3


def test_subplots_negative_tuple_element_raises():
    with pytest.raises(ValueError, match="title_space"):
        pp.subplots(2, 1, axes_size=(50, 30), title_space=(5, -1))


def test_auto_layout_excludes_legend_group_from_tightbbox():
    """pp.legend_group anchors a legend external to the axes — its width
    is handled by legend_column, not the column's `right` reservation."""
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 3, axes_size=(45, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend_group(anchor=axes[-1])
    group.add_legend(
        handles=create_legend_handles(
            labels=["A", "B", "C"],
            colors=list(pp.color_palette("pastel", 3)),
            alpha=0.2, linewidth=1.0,
        ),
        label="group",
    )
    fig.canvas.draw()
    layout = fig._publiplots_layout
    # The rightmost column's `right` should be close to baseline (< 5 mm),
    # NOT inflated to the legend's width.
    assert layout.right[-1] < 5.0, (
        f"right[-1] should be ~ baseline (legend excluded from tightbbox); "
        f"got {layout.right[-1]:.1f} mm"
    )


def test_auto_layout_per_axis_pp_legend_is_counted():
    """Per-axis pp.legend() is part of the axes' visual footprint —
    should inflate that column's `right`, unlike legend_group."""
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 2, axes_size=(45, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    builder = pp.legend(axes[0], auto=False)
    builder.add_legend(
        handles=create_legend_handles(
            labels=["A", "B"],
            colors=["#5d83c3", "#c0392b"],
            alpha=0.2, linewidth=1.0,
        ),
        label="group",
    )
    fig.canvas.draw()
    layout = fig._publiplots_layout
    # Column 0 has the per-axis legend → its `right` should exceed column 1's.
    assert layout.right[0] > layout.right[1] + 5.0, (
        f"right[0] (has pp.legend) should exceed right[1] by > 5 mm, got "
        f"right[0]={layout.right[0]:.1f}, right[1]={layout.right[1]:.1f}"
    )


# ---------------------------------------------------------------------------
# Plot functions use pp.subplots for their ax-is-None path
# ---------------------------------------------------------------------------


def test_barplot_without_ax_creates_pp_subplots_figure():
    """pp.barplot without explicit ax should produce a figure managed by pp.subplots."""
    import pandas as pd
    df = pd.DataFrame({'g': ['A', 'B', 'C'], 'v': [1.0, 2.0, 3.0]})
    ax = pp.barplot(data=df, x='g', y='v', hue='g', palette='pastel')
    fig = ax.get_figure()
    # pp.subplots attaches this attribute; plt.subplots does not.
    assert hasattr(fig, "_publiplots_layout"), (
        "barplot with ax=None should use pp.subplots (missing _publiplots_layout attribute)"
    )


# ---------------------------------------------------------------------------
# Width-awareness: legend_column auto-sizing
# ---------------------------------------------------------------------------

from publiplots.utils.legend_entries import LegendEntry, stash_entry


def _stub_handle(color="red"):
    class H:
        def __init__(self, c):
            self._c = c
        def get_facecolor(self):
            return self._c
    return H(color)


def test_legend_column_is_zero_without_group():
    fig, ax = pp.subplots(axes_size=(50, 30))
    # No pp.legend_group call.
    fig.canvas.draw()
    assert fig._publiplots_layout.legend_column == 0.0


def test_legend_column_auto_grows_with_group():
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    # Stash so auto-collect picks something up
    stash_entry(
        axes[0],
        LegendEntry.build(
            "g", "hue",
            handles=create_legend_handles(labels=["A", "B"], colors=["#000", "#fff"],
                                          alpha=0.5, linewidth=1.0),
            labels=("A", "B"),
        ),
    )
    pp.legend_group(anchor=axes[-1])
    fig.savefig("/tmp/test_legend_column_auto.png")
    layout = fig._publiplots_layout
    assert layout.legend_column > 5.0, (
        f"legend_column should have grown; got {layout.legend_column}"
    )


def test_legend_column_stays_small_with_empty_group():
    """No stashed entries and no manual adds -> group has no artists ->
    legend_column stays near zero."""
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    pp.legend_group(anchor=axes[-1])
    # No stashing, no manual add_legend calls.
    fig.savefig("/tmp/test_legend_column_empty.png")
    layout = fig._publiplots_layout
    assert layout.legend_column < 2.0, (
        f"legend_column should stay near 0; got {layout.legend_column}"
    )