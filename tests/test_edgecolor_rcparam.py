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
