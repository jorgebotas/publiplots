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
