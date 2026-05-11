"""Tests for `pp.barplot(multiple="gain")`.

Pairwise comparison mode: per cat, base segment = min(v0, v1) colored by
the losing level; top segment = max - min colored by the winning level.
Absolute values via annotate. Ties → single bar in hue_order[0] color.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

import publiplots as pp
from publiplots.utils.legend_entries import get_entries


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _bars(ax):
    return [p for p in ax.patches if hasattr(p, "get_height")]


def _simple_gain_df():
    """3 metrics × 2 models. Proposed wins AUC/F1; Baseline wins Recall."""
    return pd.DataFrame({
        "metric": pd.Categorical(
            ["AUC", "F1", "Recall", "AUC", "F1", "Recall"],
            categories=["AUC", "F1", "Recall"],
        ),
        "model": pd.Categorical(
            ["Baseline"] * 3 + ["Proposed"] * 3,
            categories=["Baseline", "Proposed"],
        ),
        "score": [0.80, 0.75, 0.88, 0.90, 0.82, 0.85],
    })


def test_gain_three_level_hue_raises():
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "B"] * 3),
        "grp": pd.Categorical(["x", "y", "z"] * 2),
        "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    with pytest.raises(ValueError, match="exactly 2 levels"):
        pp.barplot(data=df, x="cat", y="val", hue="grp",
                   multiple="gain", errorbar=None)
