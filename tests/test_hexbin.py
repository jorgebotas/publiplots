"""Tests for pp.hexbinplot."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable

import publiplots as pp
from publiplots.utils.legend_entries import get_entries, is_continuous_hue


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def hex_df():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(0.0, 1.0, n)
    y = rng.normal(0.0, 1.0, n)
    value = rng.normal(5.0, 1.0, n)
    return pd.DataFrame({"x": x, "y": y, "value": value})


@pytest.fixture(scope="module")
def two_cluster_df():
    """Two well-separated clusters with known per-cluster means of ``value``."""
    rng = np.random.default_rng(1)
    n = 500
    x0 = rng.normal(0.0, 0.2, n)
    x1 = rng.normal(10.0, 0.2, n)
    y0 = rng.normal(0.0, 0.2, n)
    y1 = rng.normal(0.0, 0.2, n)
    v0 = rng.normal(1.0, 0.01, n)  # cluster-0 mean ~1
    v1 = rng.normal(9.0, 0.01, n)  # cluster-1 mean ~9
    return pd.DataFrame({
        "x": np.concatenate([x0, x1]),
        "y": np.concatenate([y0, y1]),
        "value": np.concatenate([v0, v1]),
    })


# ---- Contract ----

def test_returns_axes(hex_df):
    ax = pp.hexbinplot(data=hex_df, x="x", y="y")
    assert isinstance(ax, Axes)
    assert len(ax.collections) >= 1


def test_rejects_figsize(hex_df):
    with pytest.raises(TypeError, match="figsize"):
        pp.hexbinplot(data=hex_df, x="x", y="y", figsize=(4, 3))


def test_respects_ax(hex_df):
    fig, ax0 = pp.subplots(axes_size=(60, 40))
    ax1 = pp.hexbinplot(data=hex_df, x="x", y="y", ax=ax0)
    assert ax1 is ax0


def test_missing_column_raises(hex_df):
    with pytest.raises(ValueError, match="Missing columns"):
        pp.hexbinplot(data=hex_df, x="does_not_exist", y="y")


# ---- C + reduce ----

def test_C_mean_reduction(two_cluster_df):
    ax = pp.hexbinplot(
        data=two_cluster_df, x="x", y="y",
        C="value", reduce_C_function=np.mean,
    )
    arr = ax.collections[0].get_array()
    # handle masked array
    if np.ma.isMaskedArray(arr):
        filled = arr.filled(np.nan)
    else:
        filled = np.asarray(arr, dtype=float)
    vmin = np.nanmin(filled)
    vmax = np.nanmax(filled)
    # known cluster means are ~1 and ~9; loose bounds
    assert vmin < 2.0
    assert vmax > 8.0


def test_reduce_C_function_max_differs_from_mean(two_cluster_df):
    ax_mean = pp.hexbinplot(
        data=two_cluster_df, x="x", y="y",
        C="value", reduce_C_function=np.mean,
    )
    ax_max = pp.hexbinplot(
        data=two_cluster_df, x="x", y="y",
        C="value", reduce_C_function=np.max,
    )
    arr_mean = np.asarray(ax_mean.collections[0].get_array().filled(np.nan)
                          if np.ma.isMaskedArray(ax_mean.collections[0].get_array())
                          else ax_mean.collections[0].get_array(), dtype=float)
    arr_max = np.asarray(ax_max.collections[0].get_array().filled(np.nan)
                         if np.ma.isMaskedArray(ax_max.collections[0].get_array())
                         else ax_max.collections[0].get_array(), dtype=float)
    # At least some bins should differ between mean and max reductions.
    diff = np.nanmax(np.abs(arr_max - arr_mean))
    assert diff > 1e-6


# ---- bins / mincnt ----

def test_bins_log_sets_lognorm(hex_df):
    ax = pp.hexbinplot(data=hex_df, x="x", y="y", bins="log")
    entry = get_entries(ax)[0]
    assert isinstance(entry.handles[0].norm, mcolors.LogNorm)


def test_mincnt_masks_empty_cells():
    """With mincnt=1, matplotlib returns a masked array whose size is
    strictly smaller than the mincnt=0 output (empty cells filtered so
    they render transparent)."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "x": rng.uniform(0, 1, 100),
        "y": rng.uniform(0, 1, 100),
    })
    ax0 = pp.hexbinplot(data=df, x="x", y="y", gridsize=30, mincnt=0)
    ax1 = pp.hexbinplot(data=df, x="x", y="y", gridsize=30, mincnt=1)
    arr0 = ax0.collections[0].get_array()
    arr1 = ax1.collections[0].get_array()
    # Publiplots default hexbin path returns a masked array.
    assert np.ma.isMaskedArray(arr1)
    # mincnt=1 drops empty cells relative to mincnt=0.
    assert arr1.size < arr0.size


# ---- Legend stash ----

def test_legend_default_stashes_count(hex_df):
    ax = pp.hexbinplot(data=hex_df, x="x", y="y")
    entries = get_entries(ax)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "hue"
    assert entry.name == "count"
    assert is_continuous_hue(entry.handles)


def test_legend_stashes_C_column_name(hex_df):
    df = hex_df.rename(columns={"value": "my_c_column"})
    ax = pp.hexbinplot(data=df, x="x", y="y", C="my_c_column")
    entries = get_entries(ax)
    assert len(entries) == 1
    assert entries[0].name == "my_c_column"
    assert entries[0].kind == "hue"
    assert is_continuous_hue(entries[0].handles)


def test_legend_false_stashes_nothing(hex_df):
    ax = pp.hexbinplot(data=hex_df, x="x", y="y", legend=False)
    assert get_entries(ax) == []


def test_legend_kws_hue_label_overrides(hex_df):
    ax = pp.hexbinplot(
        data=hex_df, x="x", y="y",
        legend_kws={"hue_label": "custom title"},
    )
    entries = get_entries(ax)
    assert len(entries) == 1
    assert entries[0].name == "custom title"


# ---- rcParams defaults ----

def test_rcparams_defaults_flow_through(hex_df):
    """linewidth + edgecolor resolve from rcParams; alpha has a literal
    default of 1.0 (hex cells are solid density patches, not marker
    overlays, so the rcParams['alpha']=0.1 scatter-tuned default is
    intentionally bypassed)."""
    saved_lw = pp.rcParams["lines.linewidth"]
    saved_edge = pp.rcParams["edgecolor"]
    try:
        pp.rcParams["lines.linewidth"] = 1.5
        pp.rcParams["edgecolor"] = None  # default: no visible edges
        ax = pp.hexbinplot(data=hex_df, x="x", y="y")
        coll = ax.collections[0]

        # alpha default is 1.0 (NOT pulled from rcParams)
        assert coll.get_alpha() == pytest.approx(1.0, abs=1e-6)

        # linewidth flows through (get_linewidths returns a 1-element array)
        lws = coll.get_linewidths()
        assert np.allclose(np.asarray(lws), 1.5)

        # edgecolor=None => 'none' => collection reports transparent edges
        # (matplotlib stores 'none' as an empty array, or an RGBA with alpha=0).
        edges = coll.get_edgecolor()
        edges_arr = np.asarray(edges)
        if edges_arr.size == 0:
            # Empty => no edges drawn (matplotlib's representation of 'none').
            assert True
        else:
            # Otherwise edges must be fully transparent.
            assert np.all(edges_arr[..., -1] == 0)
    finally:
        pp.rcParams["lines.linewidth"] = saved_lw
        pp.rcParams["edgecolor"] = saved_edge


def test_alpha_kwarg_overrides_default(hex_df):
    ax = pp.hexbinplot(data=hex_df, x="x", y="y", alpha=0.5)
    assert ax.collections[0].get_alpha() == pytest.approx(0.5, abs=1e-6)


# ---- Misc ----

def test_gridsize_tuple(hex_df):
    ax = pp.hexbinplot(data=hex_df, x="x", y="y", gridsize=(15, 10))
    assert len(ax.collections) >= 1
    arr = ax.collections[0].get_array()
    # Some cells should have been rendered.
    assert arr is not None
    assert np.asarray(arr).size > 0
