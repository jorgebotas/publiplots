"""Regression: annotate(color='hue') must render opaque text when palette
lookup falls back to the bar's translucent facecolor.

Bug #171: when hue == x, seaborn collapses the hue dimension internally
so `row['hue_value']` is None for every bar; palette lookup fails and
hue_color falls back to `rect.get_facecolor()`. With publiplots' default
outline-style (alpha=0.1) the bar's alpha leaked into the label color,
producing 10%-opacity text. The three bar-record builders now pin the
fallback alpha to 1.0.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_barplot_hue_eq_x_color_hue_renders_opaque_labels():
    """Exact repro from #171: hue == x with a palette dict + color='hue'.

    `hue_value` collapses to None inside the builder so palette lookup
    misses; the fallback used to inherit the bar's alpha=0.1.
    """
    df = pd.DataFrame([
        {"tau": "-1", "value": 0.04},
        {"tau": "-0.5", "value": 0.05},
        {"tau": "0", "value": 0.06},
    ])
    palette = {"-1": "#440154", "-0.5": "#21918c", "0": "#fde725"}
    ax = pp.barplot(
        data=df, x="tau", y="value", hue="tau",
        order=["-1", "-0.5", "0"], palette=palette,
        legend=False,
        annotate={"fmt": "{:.1%}", "anchor": "inside",
                  "rotation": 90, "color": "hue"},
    )
    meta = ax._publiplots_bar_meta
    # Sanity: this is the fallback branch (hue_value is None for all bars).
    assert all(b.hue_value is None for b in meta.bars)
    # Every bar's hue_color must be fully opaque despite the alpha=0.1 fill.
    for b in meta.bars:
        assert b.hue_color is not None
        assert b.hue_color[3] == 1.0, (
            f"hue_color alpha={b.hue_color[3]} (expected 1.0)"
        )
    # And the labels themselves render opaque.
    assert len(ax.texts) == 3
    for t in ax.texts:
        assert to_rgba(t.get_color())[3] == 1.0


def test_stacked_barplot_palette_none_fallback_strips_alpha():
    """Stacked path: ``multiple='stack'`` requires hue != x, so we force
    the fallback by re-invoking ``build_from_stacked_barplot_call`` with
    ``palette=None`` against an already-drawn stacked axes (the legitimate
    foreign-axes-style fallback path the patch covers)."""
    from publiplots.annotate._builders import build_from_stacked_barplot_call
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
        "h": pd.Categorical(["x", "y", "x", "y"], categories=["x", "y"]),
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    ax = pp.barplot(
        data=df, x="cat", y="value", hue="h",
        multiple="stack", legend=False,
    )
    # Re-build meta with palette=None to exercise the fallback branch.
    meta = build_from_stacked_barplot_call(
        ax=ax, data=df, x="cat", y="value", hue="h",
        categorical_axis="cat", palette=None,
        multiple="stack", source_frame=df,
    )
    assert len(meta.bars) > 0
    for b in meta.bars:
        assert b.hue_color is not None
        assert b.hue_color[3] == 1.0


def test_histplot_no_hue_fallback_strips_alpha():
    """histplot path: with hue=None, palette_map is None so the builder's
    `palette is not None and palette` guard fails and the fallback fires.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0.0, 1.0, 200)})
    ax = pp.histplot(
        data=df, x="x", bins=5,
        annotate={"color": "hue"},
    )
    meta = ax._publiplots_bar_meta
    assert len(meta.bars) > 0
    for b in meta.bars:
        assert b.hue_color is not None
        assert b.hue_color[3] == 1.0
    for t in ax.texts:
        assert to_rgba(t.get_color())[3] == 1.0
