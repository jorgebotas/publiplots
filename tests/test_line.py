"""Tests for pp.lineplot."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.utils.legend_entries import get_entries


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def line_df():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "t": np.tile(np.linspace(0, 10, 20), 2),
        "y": rng.normal(size=n),
        "g": np.repeat(["A", "B"], 20),
        "s": rng.uniform(1, 5, n),
    })


# ---- Contract ----

def test_lineplot_returns_axes(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y")
    assert isinstance(ax, Axes)


def test_lineplot_respects_ax(line_df):
    fig, ax0 = pp.subplots(axes_size=(50, 30))
    ax1 = pp.lineplot(data=line_df, x="t", y="y", ax=ax0)
    assert ax1 is ax0


def test_lineplot_no_hue_stashes_nothing(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y")
    assert get_entries(ax) == []


def test_lineplot_with_hue_stashes_hue_entry(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y", hue="g", palette="pastel")
    names_kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("g", "hue") in names_kinds


def test_lineplot_with_hue_size_style(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y",
                     hue="g", size="s", style="g",
                     palette="pastel", markers=True)
    kinds = {e.kind for e in get_entries(ax)}
    assert {"hue", "size", "style"} <= kinds


def test_lineplot_sort_false(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y", sort=False)
    assert isinstance(ax, Axes)


def test_lineplot_err_style_bars(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y", hue="g",
                     palette="pastel", err_style="bars")
    assert isinstance(ax, Axes)


def test_lineplot_continuous_hue_colorbar(line_df):
    df = line_df.assign(score=np.linspace(0, 100, len(line_df)))
    ax = pp.lineplot(data=df, x="t", y="y", hue="score",
                     palette="viridis", hue_norm=(0, 100))
    entries = [e for e in get_entries(ax) if e.kind == "hue"]
    assert entries and isinstance(entries[0].handles[0], ScalarMappable)


# ---- Legend stash ----

def test_lineplot_legend_false_stashes_nothing(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y", hue="g",
                     palette="pastel", legend=False)
    assert get_entries(ax) == []


def test_lineplot_legend_dict_suppresses_hue(line_df):
    ax = pp.lineplot(data=line_df, x="t", y="y",
                     hue="g", size="s", palette="pastel",
                     legend={"hue": False})
    names = [e.name for e in get_entries(ax)]
    assert "g" not in names
    assert "s" in names


def test_lineplot_in_group_suppresses_per_axis_render(line_df):
    from matplotlib.legend import Legend
    fig, axes = pp.subplots(1, 2, axes_size=(50, 40))
    pp.legend_group(anchor=axes[-1])
    pp.lineplot(data=line_df, x="t", y="y", hue="g",
                palette="pastel", ax=axes[0])
    fig.canvas.draw()
    assert [c for c in axes[0].get_children() if isinstance(c, Legend)] == []


# ---- Reject ----

def test_lineplot_rejects_figsize(line_df):
    with pytest.raises(TypeError, match="figsize"):
        pp.lineplot(data=line_df, x="t", y="y", figsize=(4, 3))
