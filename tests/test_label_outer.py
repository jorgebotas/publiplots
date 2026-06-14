import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import publiplots as pp
from publiplots.layout.label_outer import _resolve_outer_edges


def test_resolve_edges_share_all_2x3():
    # sharex=True, sharey=True on a 2x3 grid.
    # Hide x everywhere except bottom row (r==1).
    # Hide y everywhere except left col (c==0).
    hide_x, hide_y = _resolve_outer_edges(2, 3, True, True)
    assert hide_x == {(0, 0), (0, 1), (0, 2)}
    assert hide_y == {(0, 1), (0, 2), (1, 1), (1, 2)}


def test_resolve_edges_col_only_hides_x_only():
    hide_x, hide_y = _resolve_outer_edges(2, 3, "col", "col")
    # sharex="col" hides x on non-bottom; sharey="col" does NOT hide y.
    assert hide_x == {(0, 0), (0, 1), (0, 2)}
    assert hide_y == set()


def test_resolve_edges_row_only_hides_y_only():
    hide_x, hide_y = _resolve_outer_edges(2, 3, "row", "row")
    # sharex="row" does NOT hide x; sharey="row" hides y on non-left.
    assert hide_x == set()
    assert hide_y == {(0, 1), (0, 2), (1, 1), (1, 2)}


def test_resolve_edges_none_and_false():
    assert _resolve_outer_edges(2, 2, False, False) == (set(), set())
    assert _resolve_outer_edges(2, 2, "none", "none") == (set(), set())


def test_resolve_edges_1x1_noop():
    assert _resolve_outer_edges(1, 1, True, True) == (set(), set())


def test_resolve_edges_1xN_row():
    # single row: it IS the bottom row, so no x hidden; left col kept for y.
    hide_x, hide_y = _resolve_outer_edges(1, 3, True, True)
    assert hide_x == set()
    assert hide_y == {(0, 1), (0, 2)}


def test_resolve_edges_Nx1_col():
    hide_x, hide_y = _resolve_outer_edges(3, 1, True, True)
    assert hide_x == {(0, 0), (1, 0)}
    assert hide_y == set()


def test_resolve_edges_invalid_share_raises():
    with pytest.raises(ValueError, match="sharex"):
        _resolve_outer_edges(2, 2, "bogus", True)
    with pytest.raises(ValueError, match="sharey"):
        _resolve_outer_edges(2, 2, True, "bogus")


from publiplots.layout.label_outer import _as_matrix


def test_as_matrix_prefers_publiplots_axes():
    # pp.subplots stores the full unsqueezed grid at fig._publiplots_axes.
    fig, axes = pp.subplots(2, 3, axes_size=(30, 20))
    mat = _as_matrix(axes)
    assert len(mat) == 2 and len(mat[0]) == 3
    # Matches the stored matrix exactly.
    assert mat[1][2] is fig._publiplots_axes[1][2]
    plt.close(fig)


def test_as_matrix_squeezed_1xN_uses_stored_grid():
    fig, axes = pp.subplots(1, 3, axes_size=(30, 20))
    assert axes.ndim == 1  # squeezed
    mat = _as_matrix(axes)
    assert len(mat) == 1 and len(mat[0]) == 3
    assert mat[0][2] is fig._publiplots_axes[0][2]
    plt.close(fig)


def test_as_matrix_single_axes():
    fig, ax = pp.subplots(1, 1, axes_size=(30, 20))
    mat = _as_matrix(ax)
    assert len(mat) == 1 and len(mat[0]) == 1 and mat[0][0] is ax
    plt.close(fig)


def test_as_matrix_foreign_2d():
    fig, axes = plt.subplots(2, 2)  # plain matplotlib, no _publiplots_axes
    mat = _as_matrix(axes)
    assert len(mat) == 2 and len(mat[0]) == 2
    plt.close(fig)


def test_as_matrix_foreign_1d_is_single_row():
    fig, axes = plt.subplots(1, 3)  # 1D array, foreign
    mat = _as_matrix(axes)
    assert len(mat) == 1 and len(mat[0]) == 3
    plt.close(fig)


def _xlabels_visible(ax):
    """True if x tick labels are currently drawn for ax (after a draw)."""
    return any(t.get_visible() and t.get_text() for t in ax.get_xticklabels())


def test_label_outer_hides_interior_x_and_y_2x2():
    fig, axes = pp.subplots(2, 2, axes_size=(30, 20), label_outer="all")
    for r in range(2):
        for c in range(2):
            axes[r, c].set_xlabel("xlab")
            axes[r, c].set_ylabel("ylab")
    pp.label_outer(axes, sharex=True, sharey=True)
    fig.canvas.draw()

    # Bottom row keeps xlabel; top row hidden.
    assert axes[1, 0].xaxis.get_label().get_visible()
    assert not axes[0, 0].xaxis.get_label().get_visible()
    # Left col keeps ylabel; right col hidden.
    assert axes[0, 0].yaxis.get_label().get_visible()
    assert not axes[0, 1].yaxis.get_label().get_visible()
    # Tick labels: top-left has no x tick labels, bottom-left does.
    assert not _xlabels_visible(axes[0, 0])
    assert _xlabels_visible(axes[1, 0])
    plt.close(fig)


def test_label_outer_persists_after_redraw():
    # Regression guard: locator regenerates tick labels each draw; the
    # tick_params toggle must survive.
    fig, axes = pp.subplots(2, 2, axes_size=(30, 20), label_outer="all")
    pp.label_outer(axes, sharex=True, sharey=True)
    fig.canvas.draw()
    fig.canvas.draw()  # second draw regenerates ticks
    assert not _xlabels_visible(axes[0, 0])
    plt.close(fig)


def test_label_outer_offset_text_hidden_interior():
    fig, axes = pp.subplots(2, 2, axes_size=(30, 20), label_outer="all")
    pp.label_outer(axes, sharex=True, sharey=True)
    assert not axes[0, 0].xaxis.offsetText.get_visible()
    assert not axes[0, 1].yaxis.offsetText.get_visible()
    # Outer edge keeps offset text visible.
    assert axes[1, 0].xaxis.offsetText.get_visible()
    plt.close(fig)


def test_label_outer_on_foreign_axes():
    fig, axes = plt.subplots(2, 2)
    pp.label_outer(axes, sharex=True, sharey=True)
    fig.canvas.draw()
    assert not axes[0, 0].xaxis.get_label().get_visible()
    assert axes[1, 0].xaxis.get_label().get_visible()
    plt.close(fig)
