"""Tests for the global lines.markersize default applied to every
marker-bearing plot (scatter, regplot, residplot, swarm, strip, point,
line). Diameter in points; pp.scatterplot / pp.regplot / pp.residplot
square it internally for matplotlib's points² area.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _restore_markersize_rcparam():
    saved = pp.rcParams["lines.markersize"]
    yield
    pp.rcParams["lines.markersize"] = saved


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(size=20),
        "y": rng.normal(size=20),
        "g": ["a"] * 10 + ["b"] * 10,
    })


# ---- default value -----------------------------------------------------------


def test_default_markersize_is_4():
    """Publication-grade default: 4pt diameter ≈ 1.4mm at 600 dpi."""
    assert pp.rcParams["lines.markersize"] == 4


# ---- scatter family (PathCollection — area in points²) -----------------------


def _scatter_size_pt2(ax):
    pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(pcs) >= 1
    return float(pcs[0].get_sizes()[0])


def test_scatterplot_default_is_markersize_squared(df):
    fig, ax = pp.subplots(axes_size=(40, 30))
    pp.scatterplot(data=df, x="x", y="y", ax=ax)
    assert _scatter_size_pt2(ax) == 16.0  # 4²


def test_regplot_default_is_markersize_squared(df):
    fig, ax = pp.subplots(axes_size=(40, 30))
    pp.regplot(data=df, x="x", y="y", ax=ax)
    assert _scatter_size_pt2(ax) == 16.0


def test_residplot_default_is_markersize_squared(df):
    fig, ax = pp.subplots(axes_size=(40, 30))
    pp.residplot(data=df, x="x", y="y", ax=ax)
    assert _scatter_size_pt2(ax) == 16.0


# ---- categorical scatter family (PathCollection but takes diameter) ----------


def test_swarmplot_default_uses_markersize(df):
    fig, ax = pp.subplots(axes_size=(40, 30))
    pp.swarmplot(data=df, x="g", y="y", ax=ax)
    # seaborn's swarm passes ``size`` (diameter) to scatter internally
    # and stores ``size²`` as the PathCollection area.
    assert _scatter_size_pt2(ax) == 16.0


def test_stripplot_default_uses_markersize(df):
    fig, ax = pp.subplots(axes_size=(40, 30))
    pp.stripplot(data=df, x="g", y="y", ax=ax)
    assert _scatter_size_pt2(ax) == 16.0


# ---- Line2D-based markers ----------------------------------------------------


def test_pointplot_default_is_markersize_diameter(df):
    fig, ax = pp.subplots(axes_size=(40, 30))
    pp.pointplot(data=df, x="g", y="y", ax=ax)
    lines_with_markers = [
        a for a in ax.get_children()
        if isinstance(a, Line2D) and a.get_marker() not in ("", "None") and a.get_markersize() > 0
    ]
    assert len(lines_with_markers) >= 1
    # Line2D markersize is diameter in points.
    assert all(l.get_markersize() == 4.0 for l in lines_with_markers)


# ---- rcParam override ---------------------------------------------------------


def test_rcparam_override_propagates_to_scatter(df):
    pp.rcParams["lines.markersize"] = 8
    fig, ax = pp.subplots(axes_size=(40, 30))
    pp.scatterplot(data=df, x="x", y="y", ax=ax)
    assert _scatter_size_pt2(ax) == 64.0  # 8²


def test_rcparam_override_propagates_to_swarm(df):
    pp.rcParams["lines.markersize"] = 8
    fig, ax = pp.subplots(axes_size=(40, 30))
    pp.swarmplot(data=df, x="g", y="y", ax=ax)
    assert _scatter_size_pt2(ax) == 64.0


def test_rcparam_override_propagates_to_pointplot(df):
    pp.rcParams["lines.markersize"] = 8
    fig, ax = pp.subplots(axes_size=(40, 30))
    pp.pointplot(data=df, x="g", y="y", ax=ax)
    lines_with_markers = [
        a for a in ax.get_children()
        if isinstance(a, Line2D) and a.get_marker() not in ("", "None") and a.get_markersize() > 0
    ]
    assert all(l.get_markersize() == 8.0 for l in lines_with_markers)


# ---- explicit kwarg overrides rcParam ----------------------------------------


def test_explicit_size_wins_over_rcparam_swarm(df):
    fig, ax = pp.subplots(axes_size=(40, 30))
    pp.swarmplot(data=df, x="g", y="y", ax=ax, size=10)
    assert _scatter_size_pt2(ax) == 100.0  # 10²


def test_explicit_scatter_kws_s_wins_over_rcparam_regplot(df):
    fig, ax = pp.subplots(axes_size=(40, 30))
    pp.regplot(data=df, x="x", y="y", ax=ax, scatter_kws={"s": 50})
    assert _scatter_size_pt2(ax) == 50.0
