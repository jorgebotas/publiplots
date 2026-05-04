"""Tests that the figure auto-grows to contain per-axis colorbars and legends.

Prior behavior: per-axis colorbar with no title got clipped under the
sphinx-gallery scraper settings (``savefig.bbox='standard',
pad_inches=0``) because ``_artist_window_extent`` returned the bare
colorbar strip's window extent, missing the tick-label overhang.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def scrape_rcparams():
    """Mimic sphinx-gallery's scraper settings — the crops only show up
    here, not under the default ``savefig.bbox='tight'``."""
    prev_bbox = mpl.rcParams["savefig.bbox"]
    prev_pad = mpl.rcParams["savefig.pad_inches"]
    mpl.rcParams["savefig.bbox"] = "standard"
    mpl.rcParams["savefig.pad_inches"] = 0.0
    try:
        yield
    finally:
        mpl.rcParams["savefig.bbox"] = prev_bbox
        mpl.rcParams["savefig.pad_inches"] = prev_pad


def _tight_right_x1(fig):
    """Rightmost tight-bbox x of any child axes + any pinned reactor artist."""
    fig.canvas.draw()
    max_x1 = 0.0
    for a in fig.axes:
        tb = a.get_tightbbox()
        if tb is not None:
            max_x1 = max(max_x1, tb.x1)
    reactor = getattr(fig, "_publiplots_layout_reactor", None)
    if reactor is not None:
        layout = fig._publiplots_auto_layout
        for reg in reactor._registrations:
            ext = layout._artist_window_extent(reg.artist)
            if ext is not None:
                max_x1 = max(max_x1, ext.x1)
    return max_x1


# ---- colorbar, no title ----

def test_heatmap_no_title_colorbar_fits_in_figure(scrape_rcparams):
    """Per-axis colorbar without a value_label title: tick labels must
    fit inside the figure width. Regression for the reactor reading the
    bare color strip's extent instead of tick-label-inclusive tight
    bbox."""
    df = pd.DataFrame(np.random.randn(5, 5))
    ax = pp.heatmap(df, cmap="RdBu_r", center=0)
    fig = ax.get_figure()
    fig.canvas.draw()
    fw = fig.get_window_extent().width
    assert _tight_right_x1(fig) <= fw + 0.5


def test_heatmap_with_title_colorbar_fits_in_figure(scrape_rcparams):
    df = pd.DataFrame(np.random.randn(5, 5))
    ax = pp.heatmap(df, cmap="RdBu_r", center=0,
                    legend_kws={"value_label": "Score"})
    fig = ax.get_figure()
    fig.canvas.draw()
    fw = fig.get_window_extent().width
    assert _tight_right_x1(fig) <= fw + 0.5


# ---- dot heatmap: colorbar + size legend ----

def test_dot_heatmap_colorbar_and_size_legend_fit(scrape_rcparams):
    """Dot heatmap with both a continuous-hue colorbar and a size
    legend — both must land inside the figure bounds."""
    rows = [
        {"row": r, "col": c,
         "value": np.random.uniform(1, 5),
         "size_var": np.random.uniform(1, 10)}
        for r in ["A", "B", "C", "D"]
        for c in ["x", "y", "z"]
    ]
    df = pd.DataFrame(rows)
    ax = pp.heatmap(df, x="col", y="row", value="value", size="size_var",
                    cmap="Reds")
    fig = ax.get_figure()
    fig.canvas.draw()
    fw = fig.get_window_extent().width
    assert _tight_right_x1(fig) <= fw + 0.5


# ---- scatter with per-axis continuous hue colorbar ----

def test_scatter_continuous_hue_colorbar_fits(scrape_rcparams):
    """Verify the fix also covers continuous-hue scatter (the same
    code path stashes a ScalarMappable that becomes a colorbar)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=30),
        "y": rng.normal(size=30),
        "v": rng.uniform(-2, 2, size=30),
    })
    ax = pp.scatterplot(data=df, x="x", y="y", hue="v",
                        palette="RdBu_r", hue_norm=(-2, 2))
    fig = ax.get_figure()
    fig.canvas.draw()
    fw = fig.get_window_extent().width
    assert _tight_right_x1(fig) <= fw + 0.5
