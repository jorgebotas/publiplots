"""Tests for pp.regplot."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend
from matplotlib.path import Path as MplPath

import publiplots as pp
from publiplots.plot.regplot import regplot
from publiplots.utils.legend_entries import get_entries


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def simple_df():
    rng = np.random.default_rng(0)
    n = 80
    x = rng.normal(size=n)
    y = 1.5 * x + 0.3 * rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def hue_df():
    rng = np.random.default_rng(0)
    per_group = 40
    rows = []
    for i, g in enumerate(["A", "B", "C"]):
        x = rng.normal(size=per_group)
        slope = 0.5 + 0.7 * i
        y = slope * x + 0.2 * rng.normal(size=per_group)
        for xi, yi in zip(x, y):
            rows.append({"x": xi, "y": yi, "g": g})
    return pd.DataFrame(rows)


@pytest.fixture
def numeric_hue_df():
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=n)
    y = 0.8 * x + 0.3 * rng.normal(size=n)
    score = rng.uniform(0, 1, size=n)
    return pd.DataFrame({"x": x, "y": y, "score": score})


# ---- Contract ----


def test_returns_axes(simple_df):
    ax = regplot(data=simple_df, x="x", y="y")
    assert isinstance(ax, Axes)


def test_rejects_figsize(simple_df):
    with pytest.raises(TypeError, match="figsize"):
        regplot(data=simple_df, x="x", y="y", figsize=(4, 3))


def test_respects_ax(simple_df):
    fig, ax0 = pp.subplots(axes_size=(60, 40))
    ax1 = regplot(data=simple_df, x="x", y="y", ax=ax0)
    assert ax1 is ax0


def test_missing_column_raises(simple_df):
    with pytest.raises((ValueError, KeyError)):
        regplot(data=simple_df, x="does_not_exist", y="y")


# ---- Legend stashing ----


def test_no_hue_no_entry(simple_df):
    ax = regplot(data=simple_df, x="x", y="y")
    assert get_entries(ax) == []


def test_hue_stashes_one_entry(hue_df):
    ax = regplot(data=hue_df, x="x", y="y", hue="g", palette="pastel")
    entries = get_entries(ax)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "hue"
    assert entry.name == "g"
    # One handle per hue level
    assert len(entry.handles) == 3
    assert set(entry.labels) == {"A", "B", "C"}


def test_hue_draws_one_regression_per_group(hue_df):
    ax = regplot(data=hue_df, x="x", y="y", hue="g", palette="pastel")
    # Each group contributes >=1 regression line; with 3 groups we expect >= 3 lines.
    assert len(ax.lines) >= 3


def test_scatter_kws_edgecolor_override(simple_df):
    ax = regplot(
        data=simple_df, x="x", y="y", edgecolor="black",
    )
    # seaborn's scatter is PathCollection at ax.collections[0]
    pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(pcs) >= 1
    edges = pcs[0].get_edgecolor()
    # All edges must be black-ish (RGB == (0, 0, 0))
    edges_arr = np.asarray(edges)
    assert edges_arr.size > 0
    assert np.allclose(edges_arr[..., :3], 0.0)


def test_scatter_kws_user_edgecolor_wins(simple_df):
    ax = regplot(
        data=simple_df, x="x", y="y",
        edgecolor="black",
        scatter_kws={"edgecolor": "red"},
    )
    pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(pcs) >= 1
    edges = pcs[0].get_edgecolor()
    edges_arr = np.asarray(edges)
    # Red-ish => R channel high, G/B low.
    assert edges_arr.size > 0
    rgb = edges_arr[..., :3]
    assert np.all(rgb[..., 0] > 0.5)
    assert np.all(rgb[..., 1] < 0.5)
    assert np.all(rgb[..., 2] < 0.5)


def test_marker_top_level(simple_df):
    """When marker='s' is passed at the top level, the scatter must use a
    square path (not the default circle). We check the underlying PathCollection
    holds a 4-sided (square) path rather than a ~24-vertex circle.
    """
    ax = regplot(data=simple_df, x="x", y="y", marker="s")
    pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(pcs) >= 1
    paths = pcs[0].get_paths()
    assert len(paths) >= 1
    verts = paths[0].vertices
    # A square marker has exactly 4 distinct corners (5 verts with closure),
    # far fewer than a circle's ~25.
    assert len(verts) < 10


def test_legend_false_stashes_nothing(hue_df):
    ax = regplot(
        data=hue_df, x="x", y="y", hue="g", palette="pastel",
        legend=False,
    )
    assert get_entries(ax) == []


def test_legend_dict_per_kind(hue_df):
    ax = regplot(
        data=hue_df, x="x", y="y", hue="g", palette="pastel",
        legend={"hue": False},
    )
    # hue entry suppressed
    names = [e.name for e in get_entries(ax)]
    assert "g" not in names


def test_polynomial_order(simple_df):
    """order=2 should produce a non-linear regression curve. We check that
    the slope between x-points varies across the line (not constant as for
    a linear fit).
    """
    # Build quadratic data so the order=2 fit has a visible curvature.
    rng = np.random.default_rng(0)
    n = 80
    x = rng.uniform(-2, 2, size=n)
    y = x ** 2 + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y})

    ax = regplot(data=df, x="x", y="y", order=2)
    # The last Line2D should be the regression curve.
    assert len(ax.lines) >= 1
    line = ax.lines[-1]
    xs = np.asarray(line.get_xdata())
    ys = np.asarray(line.get_ydata())
    assert len(xs) >= 4
    # Compare slope on the first third vs the last third; for a quadratic
    # they should differ noticeably (sign flip around the vertex).
    third = len(xs) // 3
    slope_left = (ys[third] - ys[0]) / (xs[third] - xs[0])
    slope_right = (ys[-1] - ys[-third]) / (xs[-1] - xs[-third])
    assert abs(slope_left - slope_right) > 0.5


def test_lowess_reg(simple_df):
    pytest.importorskip("statsmodels")
    ax = regplot(data=simple_df, x="x", y="y", lowess=True)
    # lowess adds a single non-linear line without crashing.
    assert len(ax.lines) >= 1


def test_scatter_false_no_points(simple_df):
    ax = regplot(data=simple_df, x="x", y="y", scatter=False)
    # Without scatter, there should be no PathCollection (only the fit
    # line + optional FillBetween for the CI).
    path_collections = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert path_collections == []


def test_fit_reg_false_no_line(simple_df):
    ax = regplot(data=simple_df, x="x", y="y", fit_reg=False)
    assert len(ax.lines) == 0


def test_continuous_hue_warns_and_falls_back(numeric_hue_df):
    with pytest.warns(UserWarning, match="continuous hue"):
        ax = regplot(data=numeric_hue_df, x="x", y="y", hue="score")
    # Fallback: single regression, no hue entry stashed.
    names = [e.name for e in get_entries(ax)]
    assert "score" not in names


def test_rcparams_defaults_flow_through(simple_df):
    """linewidth, alpha, edgecolor must resolve from rcParams when the
    caller leaves them as None.
    """
    saved_lw = pp.rcParams["lines.linewidth"]
    saved_alpha = pp.rcParams["alpha"]
    try:
        pp.rcParams["lines.linewidth"] = 1.25
        pp.rcParams["alpha"] = 0.4
        ax = regplot(data=simple_df, x="x", y="y")
        pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(pcs) >= 1
        pc = pcs[0]
        lws = np.asarray(pc.get_linewidths())
        assert np.allclose(lws, 1.25)
        fa = np.asarray(pc.get_facecolor())
        # alpha is the 4th channel of the facecolor
        assert fa.size > 0
        assert np.allclose(fa[..., -1], 0.4)
    finally:
        pp.rcParams["lines.linewidth"] = saved_lw
        pp.rcParams["alpha"] = saved_alpha
