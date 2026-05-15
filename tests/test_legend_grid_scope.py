"""Tests for `_resolve_grid_scope` and the new pp.legend grid-scope kwargs.

PR 4 of the Composer rollout: strictly-additive upgrade to `pp.legend`
adding `rows=`, `cols=`, `span=`, `ax=` for grid-scope figure legends.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import publiplots as pp
from publiplots.utils.legend_group import _resolve_grid_scope, MultiAxesLegendGroup


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# `_resolve_grid_scope` — pure-resolver unit tests
# ---------------------------------------------------------------------------


def test_resolve_grid_scope_all_none_returns_none():
    """All-None kwargs → resolver returns None (no grid scoping; caller falls through)."""
    fig, _ = pp.subplots(nrows=2, ncols=3)
    result = _resolve_grid_scope(fig, rows=None, cols=None, span=None, ax=None)
    assert result is None


def test_resolve_grid_scope_rows_int_single_row():
    """rows=1 → all axes in row 1 of the publiplots matrix."""
    fig, axes = pp.subplots(nrows=2, ncols=3)
    result = _resolve_grid_scope(fig, rows=1, cols=None, span=None, ax=None)
    assert result is not None
    assert len(result) == 3
    expected = list(axes[1])
    assert [id(a) for a in result] == [id(a) for a in expected]


def test_resolve_grid_scope_rows_tuple_inclusive():
    """rows=(0, 1) → rows 0 AND 1 (inclusive)."""
    fig, axes = pp.subplots(nrows=3, ncols=2)
    result = _resolve_grid_scope(fig, rows=(0, 1), cols=None, span=None, ax=None)
    assert len(result) == 4  # 2 rows × 2 cols


def test_resolve_grid_scope_cols_int_single_col():
    """cols=2 → all axes in col 2."""
    fig, axes = pp.subplots(nrows=2, ncols=3)
    result = _resolve_grid_scope(fig, rows=None, cols=2, span=None, ax=None)
    assert len(result) == 2
    assert [id(a) for a in result] == [id(axes[0, 2]), id(axes[1, 2])]


def test_resolve_grid_scope_rows_and_cols_intersection():
    """rows=1, cols=2 → exactly one cell (axes[1, 2])."""
    fig, axes = pp.subplots(nrows=2, ncols=3)
    result = _resolve_grid_scope(fig, rows=1, cols=2, span=None, ax=None)
    assert len(result) == 1
    assert result[0] is axes[1, 2]


def test_resolve_grid_scope_rows_out_of_range_raises():
    """rows index outside matrix shape → ValueError naming the matrix shape."""
    fig, _ = pp.subplots(nrows=2, ncols=3)
    with pytest.raises(ValueError, match=r"rows.*out of range.*\(2, 3\)"):
        _resolve_grid_scope(fig, rows=5, cols=None, span=None, ax=None)


def test_resolve_grid_scope_cols_negative_raises_with_hint():
    """Negative col index → ValueError naming the equivalent positive index."""
    fig, _ = pp.subplots(nrows=2, ncols=3)
    with pytest.raises(ValueError, match=r"negative indices.*Use `cols=2`"):
        _resolve_grid_scope(fig, rows=None, cols=-1, span=None, ax=None)


def test_resolve_grid_scope_rows_inverted_range_raises_with_hint():
    """rows=(2, 0) → ValueError suggesting the swapped form."""
    fig, _ = pp.subplots(nrows=3, ncols=2)
    with pytest.raises(ValueError, match=r"start > end.*Use `rows=\(0, 2\)`"):
        _resolve_grid_scope(fig, rows=(2, 0), cols=None, span=None, ax=None)


def test_resolve_grid_scope_no_publiplots_axes_raises():
    """rows/cols on a non-publiplots figure → ValueError mentioning pp.subplots."""
    fig, ax = plt.subplots(2, 3)
    # plt.subplots does NOT set _publiplots_axes
    with pytest.raises(ValueError, match=r"pp\.subplots"):
        _resolve_grid_scope(fig, rows=0, cols=None, span=None, ax=None)


def test_resolve_grid_scope_canvas_figure_raises_with_pr7_hint():
    """rows/cols on a Canvas figure → ValueError pointing at PR 7 / pp.subplots.

    Canvas integration is deferred to PR 7. Until then the resolver raises
    the same way it does for a raw plt.subplots() figure (no
    `_publiplots_axes` matrix attached).
    """
    canvas = pp.Canvas("cell-2col")
    canvas.add_row(pp.PanelAxes(label="A", size=(70, 40)),
                   pp.PanelAxes(label="B", size=(70, 40)))
    fig = canvas.figure  # triggers lazy finalization
    with pytest.raises(ValueError, match=r"pp\.subplots"):
        _resolve_grid_scope(fig, rows=0, cols=None, span=None, ax=None)


def test_resolve_grid_scope_ax_list_returns_list():
    """ax=[ax1, ax2] → exactly that list (no fig._publiplots_axes lookup)."""
    fig, axes = pp.subplots(nrows=2, ncols=3)
    sel = [axes[0, 0], axes[1, 2]]
    result = _resolve_grid_scope(fig, rows=None, cols=None, span=None, ax=sel)
    assert result == sel


def test_resolve_grid_scope_ax_list_works_without_publiplots_axes():
    """ax= path doesn't require fig._publiplots_axes (works on raw plt.subplots too)."""
    fig, axes = plt.subplots(2, 2)
    sel = [axes[0, 0], axes[1, 1]]
    result = _resolve_grid_scope(fig, rows=None, cols=None, span=None, ax=sel)
    assert result == sel


def test_resolve_grid_scope_ax_empty_raises():
    """ax=[] is invalid (caller probably meant axes=None)."""
    fig, _ = pp.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError, match=r"ax.*empty"):
        _resolve_grid_scope(fig, rows=None, cols=None, span=None, ax=[])


def test_resolve_grid_scope_ax_with_rows_raises():
    """ax= is mutually exclusive with rows/cols/span."""
    fig, axes = pp.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        _resolve_grid_scope(fig, rows=0, cols=None, span=None, ax=[axes[0, 0]])


def test_resolve_grid_scope_span_fig_returns_none():
    """span='fig' → resolver returns None (full figure; caller falls through)."""
    fig, _ = pp.subplots(nrows=2, ncols=3)
    result = _resolve_grid_scope(fig, rows=None, cols=None, span="fig", ax=None)
    assert result is None


def test_resolve_grid_scope_span_invalid_raises():
    fig, _ = pp.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError, match=r"span.*'row'.*'col'.*'fig'"):
        _resolve_grid_scope(fig, rows=None, cols=None, span="invalid", ax=None)


def test_resolve_grid_scope_span_with_rows_raises():
    """span='fig' is mutually exclusive with explicit rows/cols (different modes)."""
    fig, _ = pp.subplots(nrows=2, ncols=2)
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        _resolve_grid_scope(fig, rows=0, cols=None, span="fig", ax=None)


# ---------------------------------------------------------------------------
# `pp.legend(rows=, cols=, span=, ax=)` factory integration
# ---------------------------------------------------------------------------


def test_legend_factory_rows_int_returns_group_with_correct_scope():
    """pp.legend(rows=0) returns a MultiAxesLegendGroup whose scope is row 0."""
    fig, axes = pp.subplots(nrows=2, ncols=3)
    group = pp.legend(rows=0, side="top")
    assert isinstance(group, MultiAxesLegendGroup)
    scope_ids = {id(a) for a in group._scope_axes}  # internal: list of scoped axes
    assert scope_ids == {id(axes[0, 0]), id(axes[0, 1]), id(axes[0, 2])}


def test_legend_factory_rows_tuple_and_cols_int():
    """pp.legend(rows=(0,1), cols=2) → axes[0,2] and axes[1,2]."""
    fig, axes = pp.subplots(nrows=3, ncols=3)
    group = pp.legend(rows=(0, 1), cols=2, side="right")
    scope_ids = {id(a) for a in group._scope_axes}
    assert scope_ids == {id(axes[0, 2]), id(axes[1, 2])}


def test_legend_factory_ax_list_dedupes_handles_by_label():
    """pp.legend(ax=[ax1, ax2]) collects from those axes; same-label/handle dedupes."""
    from publiplots.utils.legend_entries import LegendEntry, stash_entry
    from matplotlib.patches import Rectangle

    # NB: pp.subplots squeezes a 1-row layout to 1D, so use [0] not [0,0].
    fig, axes = pp.subplots(nrows=1, ncols=2)
    h_red = Rectangle((0, 0), 1, 1, facecolor="red", label="A")
    h_red2 = Rectangle((0, 0), 1, 1, facecolor="red", label="A")
    stash_entry(axes[0], LegendEntry.build(name="hue", kind="hue",
                                           handles=[h_red], labels=["A"]))
    stash_entry(axes[1], LegendEntry.build(name="hue", kind="hue",
                                           handles=[h_red2], labels=["A"]))
    group = pp.legend(ax=[axes[0], axes[1]], side="top")
    # Two stashes with same label + same color → group dedupes to one entry.
    # Implementation detail: _merge_entries handles this; we just assert
    # no warnings fire (mismatched-handle path would warn).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        group._materialize()  # triggers entry collection + merge


def test_legend_factory_span_fig_equals_axes_none():
    """pp.legend(span='fig') is sugar for the figure-level default."""
    fig, _ = pp.subplots(nrows=2, ncols=2)
    group_span = pp.legend(span="fig", side="bottom")
    plt.close(fig)
    fig, _ = pp.subplots(nrows=2, ncols=2)
    group_default = pp.legend(side="bottom")
    # Both should produce a group with scope=None (full grid).
    assert group_span._scope_axes is None
    assert group_default._scope_axes is None


def test_legend_factory_rows_with_legacy_axes_positional_raises():
    """Mixing the new `rows=` with the old positional `axes=` is a TypeError-equivalent."""
    fig, axes = pp.subplots(nrows=2, ncols=2)
    # Implementation message: "...`axes=` is mutually exclusive with the new `rows=`..."
    with pytest.raises(ValueError, match=r"`axes=`.*mutually exclusive.*`rows=`"):
        pp.legend(axes[0, 0], rows=0, side="top")


def test_legend_factory_grid_scope_renders_without_error():
    """Smoke: a row-scoped band on a 2x3 grid actually renders to PNG."""
    import io
    from publiplots.utils.legend_entries import LegendEntry, stash_entry
    from matplotlib.patches import Rectangle

    fig, axes = pp.subplots(nrows=2, ncols=3)
    for ax in axes.flat:
        stash_entry(ax, LegendEntry.build(name="hue", kind="hue",
                                          handles=[Rectangle((0,0),1,1,
                                                             facecolor="C0",
                                                             label="A")],
                                          labels=["A"]))
    pp.legend(rows=0, side="top")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")  # no exceptions
    assert buf.getbuffer().nbytes > 0
