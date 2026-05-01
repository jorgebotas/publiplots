"""Tests for boxplot legend stashing via LegendEntry."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

import publiplots as pp
from publiplots.utils.legend_entries import get_entries


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _box_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 60
    return pd.DataFrame({
        "x": rng.choice(["A", "B", "C"], size=n),
        "y": rng.normal(size=n),
        "g": rng.choice(["ctrl", "trt"], size=n),
    })


def test_boxplot_stashes_hue_entry():
    df = _box_df()
    fig, ax = pp.boxplot(data=df, x="x", y="y", hue="g", palette="pastel")
    entries = get_entries(ax)
    names_kinds = [(e.name, e.kind) for e in entries]
    assert ("g", "hue") in names_kinds


def test_boxplot_no_hue_stashes_nothing():
    df = _box_df()
    fig, ax = pp.boxplot(data=df, x="x", y="y")
    assert get_entries(ax) == []


def test_boxplot_legend_false_stashes_nothing():
    df = _box_df()
    fig, ax = pp.boxplot(
        data=df, x="x", y="y", hue="g", palette="pastel",
        legend=False,
    )
    assert get_entries(ax) == []


def test_boxplot_in_group_suppresses_per_axis_render():
    from matplotlib.legend import Legend
    df = _box_df()
    fig, axes = pp.subplots(1, 2, axes_size=(50, 40))
    pp.legend_group(anchor=axes[-1])
    pp.boxplot(data=df, x="x", y="y", hue="g", palette="pastel", ax=axes[0])
    fig.canvas.draw()
    per_axis_legends = [c for c in axes[0].get_children() if isinstance(c, Legend)]
    assert per_axis_legends == []


def test_boxplot_legend_dict_suppresses_hue():
    df = _box_df()
    fig, ax = pp.boxplot(
        data=df, x="x", y="y", hue="g", palette="pastel",
        legend={"hue": False},
    )
    assert get_entries(ax) == []
