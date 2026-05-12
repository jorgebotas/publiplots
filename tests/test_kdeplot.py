"""Tests for pp.kdeplot."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import publiplots as pp
from publiplots.plot.kdeplot import kdeplot
from publiplots.utils.legend_entries import (
    LegendEntry,
    get_entries,
    is_continuous_hue,
)
from publiplots.utils.legend import RectanglePatch


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def kde_df():
    rng = np.random.default_rng(0)
    n = 200
    value = np.concatenate([
        rng.normal(-2.0, 1.0, n // 2),
        rng.normal(2.0, 1.0, n // 2),
    ])
    group = rng.choice(["A", "B", "C"], size=n)
    return pd.DataFrame({"value": value, "group": group})


@pytest.fixture(scope="module")
def kde2d_df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "x": rng.normal(0.0, 1.0, n),
        "y": rng.normal(0.0, 1.0, n),
        "group": rng.choice(["A", "B"], size=n),
    })


# ---- Contract ----


def test_returns_axes_1d(kde_df):
    ax = kdeplot(data=kde_df, x="value")
    assert isinstance(ax, Axes)


def test_returns_axes_2d(kde2d_df):
    ax = kdeplot(data=kde2d_df, x="x", y="y")
    assert isinstance(ax, Axes)


def test_rejects_figsize(kde_df):
    with pytest.raises(TypeError, match="figsize"):
        kdeplot(data=kde_df, x="value", figsize=(4, 3))


def test_respects_ax(kde_df):
    fig, ax0 = pp.subplots(axes_size=(60, 40))
    ax1 = kdeplot(data=kde_df, x="value", ax=ax0)
    assert ax1 is ax0


def test_missing_column_raises(kde_df):
    with pytest.raises((KeyError, ValueError)):
        kdeplot(data=kde_df, x="does_not_exist")


def test_requires_x_or_y(kde_df):
    with pytest.raises(ValueError, match="x.*y|Provide"):
        kdeplot(data=kde_df)


# ---- 1D legend stashing ----


def test_1d_no_hue_no_entry(kde_df):
    ax = kdeplot(data=kde_df, x="value")
    assert get_entries(ax) == []


def test_1d_hue_stashes_one_hue_entry(kde_df):
    ax = kdeplot(data=kde_df, x="value", hue="group")
    entries = get_entries(ax)
    assert [(e.name, e.kind) for e in entries] == [("group", "hue")]


def test_1d_fill_true_yields_rectangle_handles(kde_df):
    ax = kdeplot(data=kde_df, x="value", hue="group", fill=True)
    entries = get_entries(ax)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "hue"
    # Rectangle-style handles for filled plots (RectanglePatch from publiplots).
    from matplotlib.patches import Patch
    assert all(isinstance(h, Patch) for h in entry.handles)
    assert any(isinstance(h, RectanglePatch) for h in entry.handles)


def test_1d_fill_false_yields_line_handles(kde_df):
    ax = kdeplot(data=kde_df, x="value", hue="group", fill=False)
    entries = get_entries(ax)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "hue"
    # Line-only handles. publiplots uses Patch-based handles but with
    # linestyle='-'; check they are not RectanglePatch.
    assert not any(isinstance(h, RectanglePatch) for h in entry.handles)


# ---- 2D ----


def test_2d_contours_draw(kde2d_df):
    ax = kdeplot(data=kde2d_df, x="x", y="y")
    assert len(ax.collections) >= 1
    # No hue, no cbar → nothing stashed.
    assert get_entries(ax) == []


def test_2d_cbar_stashes_continuous_hue(kde2d_df):
    ax = kdeplot(data=kde2d_df, x="x", y="y", cbar=True)
    entries = get_entries(ax)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "hue"
    assert is_continuous_hue(entry.handles)
    assert isinstance(entry.handles[0], ScalarMappable)


def test_2d_cbar_false_stashes_nothing(kde2d_df):
    ax = kdeplot(data=kde2d_df, x="x", y="y", cbar=False)
    assert get_entries(ax) == []


def test_2d_hue_stashes_categorical(kde2d_df):
    ax = kdeplot(data=kde2d_df, x="x", y="y", hue="group")
    entries = get_entries(ax)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "hue"
    assert entry.name == "group"
    # Categorical hue → not continuous.
    assert not is_continuous_hue(entry.handles)
    assert len(entry.labels) == 2


# ---- Legend flag routing ----


def test_legend_false_stashes_nothing(kde_df):
    ax = kdeplot(data=kde_df, x="value", hue="group", legend=False)
    assert get_entries(ax) == []


def test_legend_dict_per_kind(kde_df):
    # legend={"hue": False} should suppress the hue entry
    ax = kdeplot(data=kde_df, x="value", hue="group", legend={"hue": False})
    entries = get_entries(ax)
    # With hue suppressed, no entry should be stashed.
    assert entries == [] or all(e.kind != "hue" for e in entries)


# ---- Label defaults ----


def test_xlabel_ylabel_none_preserves_seaborn_default(kde_df):
    ax = kdeplot(data=kde_df, x="value", xlabel=None, ylabel=None)
    # seaborn sets xlabel from the column name and ylabel='Density'
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""


def test_xlabel_explicit_overrides(kde_df):
    ax = kdeplot(data=kde_df, x="value", xlabel="My Label", ylabel="My Y")
    assert ax.get_xlabel() == "My Label"
    assert ax.get_ylabel() == "My Y"


def test_title_set(kde_df):
    ax = kdeplot(data=kde_df, x="value", title="A Title")
    assert ax.get_title() == "A Title"


# ---- rcParams flow-through ----


def test_rcparams_linewidth_edgecolor_flow_through(kde_df):
    saved_lw = pp.rcParams["lines.linewidth"]
    try:
        pp.rcParams["lines.linewidth"] = 2.5
        ax = kdeplot(data=kde_df, x="value", hue="group", fill=True)
        # At least one drawn artist should carry the resolved linewidth.
        colls_lw = []
        for c in ax.collections:
            lw = c.get_linewidth()
            arr = np.atleast_1d(np.asarray(lw))
            colls_lw.extend(arr.tolist())
        lines_lw = [l.get_linewidth() for l in ax.lines]
        all_lw = colls_lw + lines_lw
        assert any(abs(lw - 2.5) < 1e-6 for lw in all_lw)
    finally:
        pp.rcParams["lines.linewidth"] = saved_lw


# ---- multiple= ----


def test_multiple_stack_with_hue(kde_df):
    ax = kdeplot(
        data=kde_df, x="value", hue="group",
        multiple="stack", palette="pastel",
    )
    # stacking yields fill collections per hue level.
    assert len(ax.collections) >= 1
    entries = get_entries(ax)
    assert len(entries) == 1
    assert entries[0].kind == "hue"


# ---- fill tri-state ----


def test_fill_none_preserves_seaborn_default(kde_df):
    # With multiple='layer' (default), seaborn's derived default for fill
    # is False. The function should not crash and should produce a plot.
    ax = kdeplot(data=kde_df, x="value", hue="group", fill=None)
    assert isinstance(ax, Axes)


# ---- cbar_ax warning ----


def test_cbar_ax_warns(kde2d_df):
    fig, (ax1, ax2) = pp.subplots(nrows=1, ncols=2, axes_size=(40, 30))
    with pytest.warns(UserWarning):
        kdeplot(data=kde2d_df, x="x", y="y", cbar=True, cbar_ax=ax2, ax=ax1)
