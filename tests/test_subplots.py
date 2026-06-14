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
    from publiplots.layout.auto_layout import _ALL_SIDES
    reactor = SubplotsAutoLayout(fig, layout, locked=_ALL_SIDES)
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
    builder = pp.legend(ax, collect=[])
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


# ---------------------------------------------------------------------------
# pp.subplots() — mixed lock (tuple containing None entries)
# ---------------------------------------------------------------------------


def test_subplots_mixed_lock_resolves_none_to_rcparams_default():
    """``xlabel_space=(0.0, None)`` must populate position 1 with the
    rcParams default so ``FigureLayout`` sees a tuple of floats only.
    Locked positions are tracked separately."""
    fig, _ = pp.subplots(2, 1, axes_size=(50, 30), xlabel_space=(0.0, None))
    layout = fig._publiplots_layout
    rc_default = pp.rcParams["subplots.xlabel_space"]
    assert layout.xlabel_space == (0.0, float(rc_default))


def test_subplots_mixed_lock_locked_position_not_remeasured():
    """``xlabel_space=(0.0, None)`` on a 2x1 grid: row 0 is locked at 0,
    row 1 auto-measures. Place a tall xlabel decoration on row 0 that
    would normally inflate xlabel_space[0] — confirm row 0 stays at 0
    while row 1 grows for its own decoration."""
    fig, axes = pp.subplots(
        2, 1, axes_size=(50, 30), xlabel_space=(0.0, None),
    )
    # Decoration on row 0 that would normally grow xlabel_space[0].
    axes[0].set_xlabel("Row 0 xlabel — should NOT grow xlabel_space[0]")
    # Decoration on row 1 that should grow xlabel_space[1].
    axes[1].set_xlabel("Row 1 xlabel — SHOULD grow xlabel_space[1]")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    assert layout.xlabel_space[0] == pytest.approx(0.0, abs=1e-6), (
        f"row 0 locked to 0; got xlabel_space[0]={layout.xlabel_space[0]:.3f}"
    )
    assert layout.xlabel_space[1] > 3.0, (
        f"row 1 should auto-measure to fit xlabel; "
        f"got xlabel_space[1]={layout.xlabel_space[1]:.3f}"
    )


def test_subplots_mixed_lock_per_column_ylabel_space():
    """``ylabel_space=(None, 0.0)`` on a 1x2 grid: col 0 auto, col 1 locked."""
    fig, axes = pp.subplots(
        1, 2, axes_size=(50, 30), ylabel_space=(None, 0.0),
    )
    axes[0].set_ylabel("Col 0 ylabel — SHOULD grow ylabel_space[0]")
    axes[1].set_ylabel("Col 1 ylabel — should NOT grow ylabel_space[1]")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    assert layout.ylabel_space[0] > 3.0, (
        f"col 0 should auto-measure; got ylabel_space[0]={layout.ylabel_space[0]:.3f}"
    )
    assert layout.ylabel_space[1] == pytest.approx(0.0, abs=1e-6), (
        f"col 1 locked to 0; got ylabel_space[1]={layout.ylabel_space[1]:.3f}"
    )


def test_subplots_mixed_lock_all_none_equivalent_to_full_auto():
    """``title_space=(None, None)`` is equivalent to ``title_space=None``
    (every position auto-measured, no locks): the reactor must NOT
    register either position as locked, and both must grow when a tall
    title is added."""
    fig, axes = pp.subplots(2, 1, axes_size=(50, 30), title_space=(None, None))
    locked = fig._publiplots_auto_layout._locked
    locked_positions = fig._publiplots_auto_layout._locked_positions
    assert "title_space" not in locked
    assert "title_space" not in locked_positions
    # Add tall titles; both rows should grow well past the rcParams default.
    axes[0].set_title("Top title")
    axes[1].set_title("Bottom title")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    assert layout.title_space[0] > 1.0
    assert layout.title_space[1] > 1.0


def test_subplots_mixed_lock_all_floats_equivalent_to_whole_side_lock():
    """``title_space=(5.0, 8.0)`` (every position a float) is equivalent
    to a full whole-side lock — neither position should ever be remeasured."""
    fig, axes = pp.subplots(2, 1, axes_size=(50, 30), title_space=(5.0, 8.0))
    axes[0].set_title("a really really really long title that wants more space")
    axes[1].set_title("a really really really long title that wants more space")
    fig.canvas.draw()
    layout = fig._publiplots_layout
    # Locked: must stay exactly at user-supplied values.
    assert layout.title_space == (5.0, 8.0)


def test_subplots_mixed_lock_invalid_string_raises():
    with pytest.raises(ValueError, match="title_space"):
        pp.subplots(2, 1, axes_size=(50, 30), title_space=("locked", None))


def test_subplots_mixed_lock_negative_raises():
    with pytest.raises(ValueError, match="title_space"):
        pp.subplots(2, 1, axes_size=(50, 30), title_space=(-1.0, None))


def test_subplots_mixed_lock_attaches_hook():
    """Even when one position is locked, the hook stays connected so
    other positions can auto-measure."""
    fig, _ = pp.subplots(2, 1, axes_size=(50, 30), xlabel_space=(0.0, None))
    assert fig._publiplots_auto_layout._cid is not None
    # The reactor should track the per-position lock.
    locked_positions = fig._publiplots_auto_layout._locked_positions
    assert "xlabel_space" in locked_positions
    assert locked_positions["xlabel_space"] == frozenset({0})
    # And xlabel_space is NOT in the whole-side ``locked`` set.
    assert "xlabel_space" not in fig._publiplots_auto_layout._locked


def test_auto_layout_figure_anchored_legend_group_uses_legend_column():
    """Figure-anchored ``pp.legend()`` writes overhang into
    ``legend_column`` (the far-right scalar reservation), leaving every
    per-column ``right[]`` at baseline."""
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 3, axes_size=(45, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend()  # figure-anchored (no explicit anchor)
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
    # Every column's right-side reservation stays at baseline — the
    # legend lives in the separate legend_column scalar.
    for c in range(len(axes)):
        assert layout.right[c] < 5.0, (
            f"right[{c}] should be ~ baseline (legend in legend_column); "
            f"got {layout.right[c]:.1f} mm"
        )
    assert layout.legend_column > 10.0, (
        f"legend_column should absorb the legend width; got {layout.legend_column:.1f} mm"
    )


def test_auto_layout_axes_anchored_legend_group_grows_per_cell_right():
    """Axes-anchored ``pp.legend(anchor=axes[r, c])`` grows the
    anchor column's per-cell ``right[c]``, not the figure-level
    ``legend_column``."""
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 3, axes_size=(45, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(anchor=axes[-1])  # axes-anchored, last column
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
    assert layout.right[-1] > 10.0, (
        f"right[-1] should absorb the legend width (axes-anchored); "
        f"got {layout.right[-1]:.1f} mm"
    )
    assert layout.right[0] < 5.0 and layout.right[1] < 5.0, (
        f"other columns should stay at baseline; "
        f"got right[0]={layout.right[0]:.1f}, right[1]={layout.right[1]:.1f}"
    )
    assert layout.legend_column < 0.5, (
        f"legend_column should not absorb axes-anchored width; "
        f"got {layout.legend_column:.1f} mm"
    )


def test_auto_layout_per_axis_pp_legend_is_counted():
    """Per-axis pp.legend() is part of the axes' visual footprint —
    should inflate that column's `right`, unlike legend_group."""
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 2, axes_size=(45, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    builder = pp.legend(axes[0], collect=[])
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
    """Figure-anchored legend_group auto-grows the figure's legend_column."""
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    # Stash so auto-collect picks something up on the first axes (doesn't
    # matter which — the group walks every axes in the grid).
    stash_entry(
        axes[0],
        LegendEntry.build(
            "g", "hue",
            handles=create_legend_handles(labels=["A", "B"], colors=["#000", "#fff"],
                                          alpha=0.5, linewidth=1.0),
            labels=("A", "B"),
        ),
    )
    pp.legend()  # figure-anchored (no anchor=)
    fig.savefig("/tmp/test_legend_column_auto.png")
    layout = fig._publiplots_layout
    assert layout.legend_column > 5.0, (
        f"legend_column should have grown; got {layout.legend_column}"
    )


def test_legend_column_stays_small_with_empty_group():
    """No stashed entries and no manual adds -> group has no artists ->
    legend_column stays near zero."""
    fig, axes = pp.subplots(1, 2, axes_size=(40, 30))
    pp.legend(anchor=axes[-1])
    # No stashing, no manual add_legend calls.
    fig.savefig("/tmp/test_legend_column_empty.png")
    layout = fig._publiplots_layout
    assert layout.legend_column < 2.0, (
        f"legend_column should stay near 0; got {layout.legend_column}"
    )


# ---------------------------------------------------------------------------
# suptitle_space auto-measurement
# ---------------------------------------------------------------------------


def test_auto_layout_suptitle_space_is_zero_without_title():
    """With no pp.suptitle attached, suptitle_space stays at 0 after draw."""
    fig, ax = pp.subplots(axes_size=(50, 30))
    fig.canvas.draw()
    assert fig._publiplots_layout.suptitle_space == 0.0


def test_auto_layout_grows_suptitle_space_for_title():
    """A tall suptitle grows suptitle_space past a few mm."""
    fig, ax = pp.subplots(axes_size=(50, 30))
    pp.suptitle("A bold figure title", fontsize=40)
    # pp.suptitle calls settle internally; one more draw for good measure.
    fig.canvas.draw()
    assert fig._publiplots_layout.suptitle_space > 5.0, (
        f"suptitle_space should have grown past 5 mm with fontsize=40; "
        f"got {fig._publiplots_layout.suptitle_space:.2f} mm"
    )


def test_auto_layout_suptitle_coexists_with_top_legend_band():
    """Figure-anchored top legend group + pp.suptitle: suptitle must sit
    above the top legend band (higher pixel y)."""
    from publiplots.utils.legend import create_legend_handles
    fig, axes = pp.subplots(1, 3, axes_size=(45, 30))
    for ax in axes:
        ax.plot([0, 1, 2], [0, 1, 0])
    group = pp.legend(side="top")
    group.add_legend(
        handles=create_legend_handles(
            labels=["A", "B", "C"],
            colors=list(pp.color_palette("pastel", 3)),
            alpha=0.2, linewidth=1.0,
        ),
        label="group",
    )
    artist = pp.suptitle("Figure-wide title", fontsize=16)
    fig.canvas.draw()

    # Collect every managed legend artist and find its window extent.
    reactor = getattr(fig, "_publiplots_layout_reactor", None)
    assert reactor is not None, "layout reactor should be attached"
    legend_y1 = 0.0
    saw_legend = False
    for reg in reactor._registrations:
        obj = reg.artist
        extent = None
        if hasattr(obj, "get_window_extent"):
            try:
                extent = obj.get_window_extent()
            except TypeError:
                extent = None
        if extent is None:
            continue
        saw_legend = True
        legend_y1 = max(legend_y1, extent.y1)
    assert saw_legend, "expected at least one registered legend artist"
    su_ext = artist.get_window_extent()
    assert su_ext.y0 > legend_y1, (
        f"suptitle should sit above top legend band; "
        f"got suptitle.y0={su_ext.y0:.1f}, legend_top.y1={legend_y1:.1f}"
    )


# ---------------------------------------------------------------------------
# Asymmetric grids — width_ratios / height_ratios
# ---------------------------------------------------------------------------


def test_width_ratios_equal_recovers_uniform():
    """Equal ratios must be bit-for-bit identical to passing no ratios."""
    fig_uniform, _ = pp.subplots(2, 3, axes_size=(40, 30))
    fig_equal, _ = pp.subplots(2, 3, axes_size=(40, 30), width_ratios=[1, 1, 1])
    assert fig_uniform.get_size_inches().tolist() == fig_equal.get_size_inches().tolist()


def test_height_ratios_equal_recovers_uniform():
    fig_uniform, _ = pp.subplots(2, 3, axes_size=(40, 30))
    fig_equal, _ = pp.subplots(2, 3, axes_size=(40, 30), height_ratios=[1, 1])
    assert fig_uniform.get_size_inches().tolist() == fig_equal.get_size_inches().tolist()


def test_width_ratios_preserves_total_grid_budget():
    """Total grid-width budget is axes_size[0] * ncols regardless of ratios."""
    fig, _ = pp.subplots(1, 2, axes_size=(45, 30), width_ratios=[7, 2])
    layout = fig._publiplots_layout
    assert sum(layout.col_widths) == pytest.approx(45 * 2)  # 90 mm total


def test_height_ratios_preserves_total_grid_budget():
    fig, _ = pp.subplots(2, 1, axes_size=(45, 35), height_ratios=[2, 5])
    layout = fig._publiplots_layout
    assert sum(layout.row_heights) == pytest.approx(35 * 2)  # 70 mm total


def test_width_ratios_jointgrid_shape():
    """Canonical JointGrid shape: big main + thin right marginal."""
    fig, _ = pp.subplots(2, 2, axes_size=(45, 35),
                         width_ratios=[7, 2], height_ratios=[2, 5])
    layout = fig._publiplots_layout
    # Budget: 90 mm wide -> [70, 20]; 70 mm tall -> [20, 50]
    assert layout.col_widths == pytest.approx((70.0, 20.0))
    assert layout.row_heights == pytest.approx((20.0, 50.0))


def test_ratios_produce_correct_per_cell_mm_widths():
    """A cell's rendered width in mm should match the derived col_widths."""
    fig, axes = pp.subplots(1, 2, axes_size=(45, 30), width_ratios=[7, 2])
    # Use the shared layout to derive expected fraction widths
    layout = fig._publiplots_layout
    W_mm, _ = layout.figure_size()
    w0_expected = layout.col_widths[0] / W_mm
    w1_expected = layout.col_widths[1] / W_mm
    assert axes[0].get_position().width == pytest.approx(w0_expected)
    assert axes[1].get_position().width == pytest.approx(w1_expected)


def test_width_ratios_length_mismatch_raises():
    with pytest.raises(ValueError, match="width_ratios must have length 3"):
        pp.subplots(1, 3, width_ratios=[1, 2])


def test_height_ratios_length_mismatch_raises():
    with pytest.raises(ValueError, match="height_ratios must have length 2"):
        pp.subplots(2, 1, height_ratios=[1, 2, 3])


def test_width_ratios_non_positive_raises():
    with pytest.raises(ValueError, match="width_ratios\\[1\\] must be positive"):
        pp.subplots(1, 2, width_ratios=[1, 0])


def test_height_ratios_non_positive_raises():
    with pytest.raises(ValueError, match="height_ratios\\[0\\] must be positive"):
        pp.subplots(2, 1, height_ratios=[-1, 2])


def test_width_ratios_non_numeric_raises():
    with pytest.raises(ValueError, match="width_ratios"):
        pp.subplots(1, 2, width_ratios=["wide", "narrow"])


def test_asymmetric_grid_axes_dont_overlap():
    """JointGrid-shaped 2x2 cells must still partition the grid with no overlap."""
    fig, axes = pp.subplots(2, 2, axes_size=(45, 35),
                            width_ratios=[7, 2], height_ratios=[2, 5])
    rects = [ax.get_position().bounds for row in axes for ax in row]
    for i, (x0a, y0a, wa, ha) in enumerate(rects):
        for j, (x0b, y0b, wb, hb) in enumerate(rects):
            if i == j:
                continue
            x1a, y1a = x0a + wa, y0a + ha
            x1b, y1b = x0b + wb, y0b + hb
            overlaps = not (x1a <= x0b + 1e-9 or x1b <= x0a + 1e-9
                            or y1a <= y0b + 1e-9 or y1b <= y0a + 1e-9)
            assert not overlaps, f"asymmetric cells {i} and {j} overlap"


def test_asymmetric_grid_preserves_reservation_contract():
    """Per-row/per-col reservations still apply uniformly across a row/col,
    regardless of heterogeneous cell dimensions."""
    fig, _ = pp.subplots(2, 2, axes_size=(45, 35),
                         width_ratios=[7, 2], height_ratios=[2, 5],
                         title_space=6.0, xlabel_space=10.0,
                         ylabel_space=12.0, right=3.0)
    layout = fig._publiplots_layout
    # Scalars broadcast per-row / per-col as before.
    assert layout.title_space == (6.0, 6.0)
    assert layout.xlabel_space == (10.0, 10.0)
    assert layout.ylabel_space == (12.0, 12.0)
    assert layout.right == (3.0, 3.0)


def test_asymmetric_grid_sharex_col_works():
    """Sharing x across a column must still yield consistent xlim."""
    fig, axes = pp.subplots(2, 2, axes_size=(45, 35),
                            width_ratios=[7, 2], height_ratios=[2, 5],
                            sharex="col")
    axes[1, 0].set_xlim(-5, 5)
    # axes[0, 0] shares x with axes[1, 0] (same column)
    assert axes[0, 0].get_xlim() == pytest.approx((-5, 5))


def test_figure_layout_asymmetric_passthrough():
    """FigureLayout accepts col_widths/row_heights directly (internal API)."""
    from publiplots.layout.figure_layout import FigureLayout
    layout = FigureLayout(
        nrows=2, ncols=2, axes_size=(45, 35),
        col_widths=(70.0, 20.0), row_heights=(20.0, 50.0),
        title_space=(0.0, 0.0), xlabel_space=(0.0, 8.0),
        ylabel_space=(10.0, 0.0), right=(0.0, 2.0),
        hspace=1.0, wspace=1.0, outer_pad=2.0, legend_column=0.0,
    )
    W, H = layout.figure_size()
    # outer_pad + legend_band_left + ylabel_space + col_widths + right + wspace + legend_column + outer_pad
    # = 2 + 0 + (10 + 0) + (70 + 20) + (0 + 2) + 1 + 0 + 2 = 107
    assert W == pytest.approx(107.0)
    # outer_pad + suptitle_space + legend_band_top + title_space + row_heights + xlabel_space + hspace + legend_band_bottom + outer_pad
    # = 2 + 0 + 0 + (0 + 0) + (20 + 50) + (0 + 8) + 1 + 0 + 2 = 83
    assert H == pytest.approx(83.0)


def test_figure_layout_col_widths_length_mismatch_raises():
    from publiplots.layout.figure_layout import FigureLayout
    with pytest.raises(ValueError, match="col_widths must have length"):
        FigureLayout(
            nrows=1, ncols=2, axes_size=(50, 30),
            col_widths=(50.0,),  # wrong length
            title_space=(5.0,), xlabel_space=(8.0,),
            ylabel_space=(10.0, 10.0), right=(2.0, 2.0),
            hspace=0.0, wspace=0.0, outer_pad=0.0, legend_column=0.0,
        )


def test_auto_layout_reactor_settles_on_asymmetric_grid():
    """Reactor must auto-measure title_space on an asymmetric top row without
    corrupting the per-row / per-col reservation contract."""
    fig, axes = pp.subplots(2, 2, axes_size=(45, 35),
                            width_ratios=[7, 2], height_ratios=[2, 5])
    # Give only the main cell a long title; reactor should grow the
    # title_space reservation for row=0 (where the marginals live)
    # — but our main panel is in row=1, so we set titles on row 1.
    axes[1, 0].set_title("A long main-panel title that forces measurement")
    fig._publiplots_auto_layout.settle()
    layout = fig._publiplots_layout
    # After settlement, title_space for row 1 (where the title lives) must
    # be positive and the figure height must have grown to accommodate it.
    W_mm, H_mm = layout.figure_size()
    # Figure dimensions (inches * 25.4) should match the layout's mm size.
    w_actual, h_actual = fig.get_size_inches() * 25.4
    assert w_actual == pytest.approx(W_mm, rel=1e-3)
    assert h_actual == pytest.approx(H_mm, rel=1e-3)


def test_figure_layout_row_heights_non_positive_raises():
    from publiplots.layout.figure_layout import FigureLayout
    with pytest.raises(ValueError, match="row_heights\\[0\\] must be positive"):
        FigureLayout(
            nrows=2, ncols=1, axes_size=(50, 30),
            row_heights=(0.0, 30.0),  # zero height
            title_space=(5.0, 5.0), xlabel_space=(8.0, 8.0),
            ylabel_space=(10.0,), right=(2.0,),
            hspace=0.0, wspace=0.0, outer_pad=0.0, legend_column=0.0,
        )


def test_subplots_label_outer_invalid_raises():
    with pytest.raises(ValueError, match="label_outer"):
        pp.subplots(2, 2, axes_size=(30, 20), sharex=True, label_outer="bogus")


def test_subplots_label_outer_accepts_true_false_all():
    # Should not raise.
    pp.subplots(2, 2, axes_size=(30, 20), sharex=True, label_outer=True)
    pp.subplots(2, 2, axes_size=(30, 20), sharex=True, label_outer=False)
    pp.subplots(2, 2, axes_size=(30, 20), sharex=True, label_outer="all")


def test_subplots_default_hides_interior_when_shared():
    # label_outer defaults to True; sharex/sharey active -> interior hidden.
    fig, axes = pp.subplots(2, 2, axes_size=(30, 20), sharex=True, sharey=True)
    fig.canvas.draw()
    # Tick labels hidden on top-left x.
    assert not any(t.get_visible() and t.get_text()
                   for t in axes[0, 0].get_xticklabels())
    # Bottom-left keeps x tick labels.
    assert any(t.get_visible() and t.get_text()
               for t in axes[1, 0].get_xticklabels())
    plt.close(fig)


def test_subplots_label_outer_false_draws_all():
    fig, axes = pp.subplots(2, 2, axes_size=(30, 20), sharex=True, sharey=True,
                            label_outer=False)
    fig.canvas.draw()
    # Every axes keeps its x tick labels.
    assert any(t.get_visible() and t.get_text()
               for t in axes[0, 0].get_xticklabels())
    plt.close(fig)


def test_subplots_no_sharing_label_outer_true_is_noop():
    # Nothing shared -> nothing hidden even though label_outer=True (default).
    fig, axes = pp.subplots(2, 2, axes_size=(30, 20))
    fig.canvas.draw()
    assert any(t.get_visible() and t.get_text()
               for t in axes[0, 0].get_xticklabels())
    plt.close(fig)
