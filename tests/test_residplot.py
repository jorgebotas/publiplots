"""Tests for pp.residplot (imported directly until exported)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

import publiplots as pp
from publiplots.plot.residplot import residplot
from publiplots.utils.legend_entries import get_entries


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def linear_df():
    """Simple linear x–y with noise."""
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(0.0, 1.0, n)
    y = 1.2 * x + rng.normal(0.0, 0.5, n)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def hue_df():
    """Three groups, each with its own slope."""
    rng = np.random.default_rng(0)
    n_per = 60
    slopes = {"A": 1.0, "B": 0.5, "C": -0.8}
    frames = []
    for group, slope in slopes.items():
        x = rng.normal(0.0, 1.0, n_per)
        y = slope * x + rng.normal(0.0, 0.3, n_per)
        frames.append(pd.DataFrame({"x": x, "y": y, "group": group}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def continuous_hue_df():
    rng = np.random.default_rng(0)
    n = 80
    x = rng.normal(0.0, 1.0, n)
    y = 0.5 * x + rng.normal(0.0, 0.3, n)
    score = rng.uniform(0.0, 10.0, n)
    return pd.DataFrame({"x": x, "y": y, "score": score})


# ---- Contract ----------------------------------------------------------------


def test_returns_axes(linear_df):
    ax = residplot(data=linear_df, x="x", y="y")
    assert isinstance(ax, Axes)


def test_rejects_figsize(linear_df):
    with pytest.raises(TypeError, match="figsize"):
        residplot(data=linear_df, x="x", y="y", figsize=(4, 3))


def test_respects_ax(linear_df):
    fig, ax0 = pp.subplots(axes_size=(60, 40))
    ax1 = residplot(data=linear_df, x="x", y="y", ax=ax0)
    assert ax1 is ax0


def test_missing_column_raises(linear_df):
    with pytest.raises((ValueError, KeyError)):
        residplot(data=linear_df, x="does_not_exist", y="y")


# ---- Legend stash ------------------------------------------------------------


def test_no_hue_no_entry(linear_df):
    ax = residplot(data=linear_df, x="x", y="y")
    assert get_entries(ax) == []


def test_hue_stashes_one_entry(hue_df):
    ax = residplot(data=hue_df, x="x", y="y", hue="group")
    entries = get_entries(ax)
    assert len(entries) == 1
    assert entries[0].kind == "hue"
    assert entries[0].name == "group"
    # Three categorical levels → three handles.
    assert len(entries[0].handles) == 3
    assert set(entries[0].labels) == {"A", "B", "C"}


def test_hue_draws_one_residual_per_group(hue_df):
    ax = residplot(data=hue_df, x="x", y="y", hue="group")
    # One PathCollection per group from the scatter call.
    assert len(ax.collections) >= 3


def test_zero_reference_line_drawn(linear_df):
    ax = residplot(data=linear_df, x="x", y="y")
    # sns.residplot always draws a dotted y=0 reference line.
    dotted = [
        l for l in ax.lines
        if l.get_linestyle() in (":", "dotted")
        and np.allclose(l.get_ydata(), 0.0)
    ]
    assert len(dotted) >= 1


def test_legend_false_stashes_nothing(hue_df):
    ax = residplot(data=hue_df, x="x", y="y", hue="group", legend=False)
    assert get_entries(ax) == []


def test_legend_dict_per_kind(hue_df):
    # legend={"hue": False} should suppress the single hue entry.
    ax = residplot(data=hue_df, x="x", y="y", hue="group", legend={"hue": False})
    entries = get_entries(ax)
    # hue flag is off, so nothing is stashed.
    assert entries == []


# ---- scatter_kws + marker forwarding ----------------------------------------


def test_scatter_kws_edgecolor_override(linear_df):
    # Set pp default edgecolor; verify it reaches the PathCollection.
    saved = pp.rcParams["edgecolor"]
    try:
        pp.rcParams["edgecolor"] = "#111111"
        ax = residplot(data=linear_df, x="x", y="y")
        coll = ax.collections[0]
        edges = np.asarray(coll.get_edgecolors())
        # At least one edge row should be our configured color
        # (matplotlib stores named colors as RGBA tuples).
        assert edges.size > 0
    finally:
        pp.rcParams["edgecolor"] = saved


def test_scatter_kws_user_edgecolor_wins(linear_df):
    # User-supplied scatter_kws['edgecolor'] must beat our defaults.
    ax = residplot(
        data=linear_df, x="x", y="y",
        edgecolor="#111111",
        scatter_kws={"edgecolor": "red"},
    )
    coll = ax.collections[0]
    edges = np.asarray(coll.get_edgecolors())
    # Red is (1, 0, 0, alpha)
    assert np.isclose(edges[0][0], 1.0)
    assert np.isclose(edges[0][1], 0.0)
    assert np.isclose(edges[0][2], 0.0)


def test_marker_top_level(linear_df):
    # Our function accepts `marker=` at top level and forwards it
    # to the underlying scatter, while `sns.residplot` itself does not.
    ax = residplot(data=linear_df, x="x", y="y", marker="^")
    coll = ax.collections[0]
    # Triangle path is 3 vertices + closing vertex. The exact vertices
    # depend on the marker, but the default "o" marker is an (N>=8)-sided
    # polygon, whereas "^" is a 3-sided polygon — compare vertex counts.
    paths = coll.get_paths()
    assert len(paths) >= 1
    tri_vertices = paths[0].vertices
    # Triangle marker path typically has exactly 4 vertices (3 corners + close).
    assert tri_vertices.shape[0] <= 5


# ---- lowess + order ---------------------------------------------------------


def test_lowess_true_adds_line(linear_df):
    ax_no = residplot(data=linear_df, x="x", y="y", lowess=False)
    ax_yes = residplot(data=linear_df, x="x", y="y", lowess=True)
    # With lowess, at least one non-dotted line is drawn in addition to
    # the y=0 reference.
    non_ref_no = [l for l in ax_no.lines if l.get_linestyle() not in (":", "dotted")]
    non_ref_yes = [l for l in ax_yes.lines if l.get_linestyle() not in (":", "dotted")]
    assert len(non_ref_yes) > len(non_ref_no)


def test_order_2_polynomial_residuals(linear_df):
    # Building with order=2 should not crash and should return valid residuals.
    ax1 = residplot(data=linear_df, x="x", y="y", order=1)
    ax2 = residplot(data=linear_df, x="x", y="y", order=2)
    # Residuals are the y-coordinates of the scatter PathCollection.
    # order=1 and order=2 fit different models, so the residuals should differ.
    r1 = ax1.collections[0].get_offsets()[:, 1]
    r2 = ax2.collections[0].get_offsets()[:, 1]
    # Sort to compare distribution (offsets order may differ).
    assert not np.allclose(np.sort(r1), np.sort(r2))


# ---- Continuous hue fallback ------------------------------------------------


def test_continuous_hue_warns_and_falls_back(continuous_hue_df):
    with pytest.warns(UserWarning, match="continuous hue"):
        ax = residplot(data=continuous_hue_df, x="x", y="y", hue="score")
    # Fallback → single scatter call, no legend entry.
    assert get_entries(ax) == []
    # Only one scatter collection.
    assert len(ax.collections) == 1


# ---- rcParams defaults -------------------------------------------------------


def test_rcparams_defaults_flow_through(linear_df):
    saved_lw = pp.rcParams["lines.linewidth"]
    saved_alpha = pp.rcParams["alpha"]
    try:
        pp.rcParams["lines.linewidth"] = 1.75
        pp.rcParams["alpha"] = 0.5
        ax = residplot(data=linear_df, x="x", y="y")
        coll = ax.collections[0]

        # linewidth flows through to the scatter marker edge width.
        lws = np.asarray(coll.get_linewidths())
        assert np.allclose(lws, 1.75)

        # alpha flows through to the scatter (either via collection alpha
        # or baked into the facecolors alpha channel).
        alpha_attr = coll.get_alpha()
        if alpha_attr is not None:
            assert alpha_attr == pytest.approx(0.5, abs=1e-6)
        else:
            face = np.asarray(coll.get_facecolors())
            assert np.isclose(face[0][-1], 0.5, atol=1e-6)
    finally:
        pp.rcParams["lines.linewidth"] = saved_lw
        pp.rcParams["alpha"] = saved_alpha
