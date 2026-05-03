"""Tests for adaptive legend row spacing when handles are oversized."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.utils.legend import (
    MarkerPatch,
    compute_min_labelspacing,
    create_legend_handles,
)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# ---- compute_min_labelspacing ----

def test_compute_min_labelspacing_text_only_returns_default():
    """Rectangle / line handles (no markersize) keep the matplotlib default."""
    handles = create_legend_handles(labels=["A", "B"], colors=["#ff0000", "#00ff00"])
    assert compute_min_labelspacing(handles, fontsize=8.0) == 0.3


def test_compute_min_labelspacing_small_markers_returns_default():
    """Markers at or below the font size don't need extra breathing room."""
    handles = [
        MarkerPatch(marker="o", facecolor="red", edgecolor="red",
                    alpha=1.0, linewidth=1.0, label="small", markersize=6.0),
        MarkerPatch(marker="o", facecolor="blue", edgecolor="blue",
                    alpha=1.0, linewidth=1.0, label="smaller", markersize=4.0),
    ]
    assert compute_min_labelspacing(handles, fontsize=8.0) == 0.3


def test_compute_min_labelspacing_large_markers_scales_with_tallest():
    """Matplotlib's legend row slot is fixed at ``fontsize *
    handleheight``, so oversized markers overflow into adjacent rows.
    labelspacing must scale with the tallest marker:

        labelspacing ≥ (tallest / fontsize) - 1 + breathing

    With breathing=0.5, a 24-pt marker on an 8-pt font needs
    labelspacing ≥ 24/8 - 1 + 0.5 = 2.5.
    """
    handles = [
        MarkerPatch(marker="o", facecolor="red", edgecolor="red",
                    alpha=1.0, linewidth=1.0, label="big", markersize=24.0),
        MarkerPatch(marker="o", facecolor="red", edgecolor="red",
                    alpha=1.0, linewidth=1.0, label="small", markersize=6.0),
    ]
    got = compute_min_labelspacing(handles, fontsize=8.0)
    assert got == pytest.approx(2.5)


def test_compute_min_labelspacing_never_below_default():
    """Tiny handles shouldn't compress below the default."""
    handles = [
        MarkerPatch(marker="o", facecolor="red", edgecolor="red",
                    alpha=1.0, linewidth=1.0, label="tiny", markersize=1.0),
    ]
    assert compute_min_labelspacing(handles, fontsize=12.0) == 0.3


# ---- Integration: the scatter size legend picks up the wider spacing ----

def test_scatter_size_legend_uses_wider_labelspacing():
    """A scatter with large ``sizes=`` must render its legend with a
    labelspacing bigger than matplotlib's 0.3 default."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=40),
        "y": rng.normal(size=40),
        "m": rng.uniform(1, 10, size=40),
    })
    ax = pp.scatterplot(data=df, x="x", y="y", size="m", sizes=(50, 500))
    fig = ax.get_figure()
    fig.canvas.draw()
    from matplotlib.legend import Legend
    [legend] = [c for c in ax.get_children() if isinstance(c, Legend)]
    # labelspacing is stored as ``handleheight`` / ``labelspacing``;
    # matplotlib exposes it via the Legend's _get_loc params — easier to
    # read the raw attribute set by our builder.
    assert legend.labelspacing > 0.3


def test_text_only_legend_uses_default_labelspacing():
    """Categorical hue legend (no markersize) keeps the default spacing."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=30),
        "y": rng.normal(size=30),
        "g": rng.choice(["A", "B", "C"], size=30),
    })
    ax = pp.scatterplot(data=df, x="x", y="y", hue="g", palette="pastel")
    fig = ax.get_figure()
    fig.canvas.draw()
    from matplotlib.legend import Legend
    [legend] = [c for c in ax.get_children() if isinstance(c, Legend)]
    # Hue-only legends use the default 10-pt circle swatch → no inflation.
    # The MarkerPatch default markersize is ``lines.markersize`` (6 pt),
    # which is below the legend fontsize (8), so we stay at 0.3.
    assert legend.labelspacing == pytest.approx(0.3)
