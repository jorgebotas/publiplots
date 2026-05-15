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
