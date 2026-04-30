"""Tests for the global edgecolor rcParam."""
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import pandas as pd
import numpy as np

import publiplots as pp


@pytest.fixture(autouse=True)
def _restore_edgecolor_rcparam():
    """Snapshot and restore pp.rcParams['edgecolor'] so tests don't leak state."""
    original = pp.rcParams["edgecolor"]
    yield
    pp.rcParams["edgecolor"] = original


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_edgecolor_rcparam_default_is_none():
    """The default edgecolor rcParam is None — 'auto' mode, preserves current behavior."""
    assert pp.rcParams["edgecolor"] is None


def _simple_bar_data():
    return pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
    })


def test_barplot_rcparam_applies_when_arg_omitted():
    """Setting rcParams['edgecolor'] colors bar edges when no arg is passed."""
    pp.rcParams["edgecolor"] = "red"
    fig, ax = plt.subplots()
    pp.barplot(data=_simple_bar_data(), x="category", y="value", ax=ax)
    red = to_rgba("red")
    edges = [to_rgba(patch.get_edgecolor()) for patch in ax.patches]
    assert edges, "expected at least one bar patch"
    for edge in edges:
        assert edge == red


def test_barplot_explicit_arg_wins_over_rcparam():
    """When both rcParam and arg are set, the arg wins."""
    pp.rcParams["edgecolor"] = "red"
    fig, ax = plt.subplots()
    pp.barplot(data=_simple_bar_data(), x="category", y="value",
               edgecolor="blue", ax=ax)
    blue = to_rgba("blue")
    for patch in ax.patches:
        assert to_rgba(patch.get_edgecolor()) == blue


def test_barplot_passthrough_preserves_auto_edge():
    """With rcParam at its default None and no arg, bar edge matches face color."""
    assert pp.rcParams["edgecolor"] is None
    fig, ax = plt.subplots()
    pp.barplot(data=_simple_bar_data(), x="category", y="value", ax=ax)
    for patch in ax.patches:
        face = to_rgba(patch.get_facecolor())
        edge = to_rgba(patch.get_edgecolor())
        # Compare only RGB — bar face is alpha-dimmed but edge is full opacity
        assert face[:3] == edge[:3]


def test_boxplot_rcparam_applies():
    """boxplot respects rcParams['edgecolor']."""
    pp.rcParams["edgecolor"] = "red"
    data = pd.DataFrame({
        "group": pd.Categorical(np.repeat(["A", "B"], 20)),
        "value": np.concatenate([np.random.RandomState(0).randn(20),
                                  np.random.RandomState(1).randn(20)]),
    })
    fig, ax = plt.subplots()
    pp.boxplot(data=data, x="group", y="value", ax=ax)
    red = to_rgba("red")
    # Box edges live on ax.patches in seaborn's boxplot
    for patch in ax.patches:
        assert to_rgba(patch.get_edgecolor()) == red


def test_scatterplot_rcparam_applies():
    """scatterplot respects rcParams['edgecolor']."""
    pp.rcParams["edgecolor"] = "red"
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    fig, ax = plt.subplots()
    pp.scatterplot(data=data, x="x", y="y", ax=ax)
    red = to_rgba("red")
    assert ax.collections, "expected a PathCollection from scatter"
    for collection in ax.collections:
        edges = collection.get_edgecolors()
        assert len(edges) > 0
        for edge in edges:
            assert tuple(edge) == red


def test_violinplot_rcparam_applies_to_cloud_polys():
    """violinplot cloud PolyCollections respect rcParams['edgecolor'].

    Regression guard: seaborn's linecolor kwarg colors inner stat lines only;
    the FillBetweenPolyCollection edges keep the palette color unless we
    override them explicitly in violinplot's post-processing.
    """
    from matplotlib.collections import FillBetweenPolyCollection

    pp.rcParams["edgecolor"] = "red"
    data = pd.DataFrame({
        "group": pd.Categorical(np.repeat(["A", "B"], 30)),
        "value": np.concatenate([np.random.RandomState(0).randn(30),
                                  np.random.RandomState(1).randn(30)]),
    })
    fig, ax = plt.subplots()
    pp.violinplot(data=data, x="group", y="value", hue="group",
                  palette="pastel", ax=ax)
    red = to_rgba("red")
    poly_collections = [c for c in ax.collections
                        if isinstance(c, FillBetweenPolyCollection)]
    assert poly_collections, "expected FillBetweenPolyCollection violin bodies"
    for coll in poly_collections:
        edges = coll.get_edgecolors()
        assert len(edges) > 0
        for edge in edges:
            assert tuple(edge) == red
