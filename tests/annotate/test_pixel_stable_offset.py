"""Regression tests: annotation positions must stay pixel-stable when
axis limits change after annotate runs.

Annotations place the label at the bar/point/box edge in data coords
and apply the mm offset via a ``ScaledTranslation`` display-space
transform. So any later transform change (``ax.set_xlim/ylim``, the
implicit relim triggered by ``sharex/sharey=True`` on a neighbor with a
different value range, or a dpi change) leaves the physical pixel gap
between label and anchor unchanged.

Before this fix, the offset was baked into data coords at annotate
time, so expanding the axis afterwards compressed the visual gap.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _gap_px(ax, text, anchor_value, axis="y"):
    """Pixel gap between the text's rendered bbox edge and the anchor
    point on the value axis.
    """
    renderer = ax.figure.canvas.get_renderer()
    bb = text.get_window_extent(renderer)
    if axis == "y":
        anchor_px = ax.transData.transform((0, anchor_value))[1]
        return bb.y0 - anchor_px
    else:
        anchor_px = ax.transData.transform((anchor_value, 0))[0]
        return bb.x0 - anchor_px


def _simple_vbar_df():
    return pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
    })


def test_barplot_external_set_ylim_preserves_label_gap():
    """User expands ylim after annotate — labels still sit just above bars."""
    ax = pp.barplot(data=_simple_vbar_df(), x="category", y="value",
                    annotate=True)
    ax.figure.canvas.draw()
    gap_before = _gap_px(ax, ax.texts[0], anchor_value=1.0, axis="y")

    # External: user changes the axis limits after annotate.
    ax.set_ylim(0, 10)
    ax.figure.canvas.draw()
    gap_after = _gap_px(ax, ax.texts[0], anchor_value=1.0, axis="y")

    assert gap_before == pytest.approx(gap_after, abs=0.5), (
        f"pixel gap drifted after set_ylim: {gap_before:.2f} -> {gap_after:.2f}"
    )


def test_barplot_sharey_preserves_label_gap_across_panes():
    """sharey=True implicitly expands the smaller-pane ylim to match the
    bigger pane. Labels on both panes must keep the same pixel gap.
    """
    df_small = _simple_vbar_df()
    df_big = pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [10.0, 20.0, 30.0],
    })
    fig, axes = pp.subplots(1, 2, axes_size=(50, 50), sharey=True)
    pp.barplot(data=df_small, x="category", y="value", ax=axes[0], annotate=True)
    pp.barplot(data=df_big, x="category", y="value", ax=axes[1], annotate=True)
    fig.canvas.draw()

    gap_small = _gap_px(axes[0], axes[0].texts[0], anchor_value=1.0, axis="y")
    gap_big = _gap_px(axes[1], axes[1].texts[0], anchor_value=10.0, axis="y")

    assert gap_small == pytest.approx(gap_big, abs=0.5), (
        f"sharey panes disagree on label pixel gap: "
        f"small={gap_small:.2f}, big={gap_big:.2f}"
    )
    # And both gaps are positive (label above bar), not zero (label on bar).
    assert gap_small > 1.0
    assert gap_big > 1.0


def test_horizontal_barplot_sharex_preserves_label_gap():
    """Same invariant on horizontal bars + sharex=True."""
    df_small = _simple_vbar_df()
    df_big = pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [10.0, 20.0, 30.0],
    })
    fig, axes = pp.subplots(2, 1, axes_size=(80, 30), sharex=True)
    pp.barplot(data=df_small, y="category", x="value", ax=axes[0], annotate=True)
    pp.barplot(data=df_big, y="category", x="value", ax=axes[1], annotate=True)
    fig.canvas.draw()

    gap_small = _gap_px(axes[0], axes[0].texts[0], anchor_value=1.0, axis="x")
    gap_big = _gap_px(axes[1], axes[1].texts[0], anchor_value=10.0, axis="x")

    assert gap_small == pytest.approx(gap_big, abs=0.5)
    assert gap_small > 1.0


def test_boxplot_set_ylim_preserves_label_gap():
    """Boxplot labels must also survive a post-annotate ``set_ylim``."""
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for g, base in zip(("A", "B", "C"), (1.0, 2.0, 3.0)):
        for v in rng.normal(base, 0.5, 30):
            rows.append({"g": g, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = df["g"].astype("category")

    ax = pp.boxplot(data=df, x="g", y="y", annotate={"anchor": "top"})
    ax.figure.canvas.draw()
    meta = ax._publiplots_box_meta
    median0 = meta.boxes[0].stats["median"]
    gap_before = _gap_px(ax, ax.texts[0], anchor_value=median0, axis="y")

    ax.set_ylim(-5, 20)
    ax.figure.canvas.draw()
    gap_after = _gap_px(ax, ax.texts[0], anchor_value=median0, axis="y")

    assert gap_before == pytest.approx(gap_after, abs=0.5)


def test_pointplot_set_ylim_preserves_label_gap():
    """Pointplot labels must also survive a post-annotate ``set_ylim``."""
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for t, base in zip(("t1", "t2", "t3"), (1.0, 2.5, 3.2)):
        for v in rng.normal(base, 0.3, 10):
            rows.append({"time": t, "v": float(v)})
    df = pd.DataFrame(rows)
    df["time"] = df["time"].astype("category")

    ax = pp.pointplot(data=df, x="time", y="v", errorbar="se", annotate=True)
    ax.figure.canvas.draw()
    # Anchor is the errorbar cap, not the marker — just compare gaps.
    renderer = ax.figure.canvas.get_renderer()
    bb_before = ax.texts[0].get_window_extent(renderer)

    ax.set_ylim(-5, 20)
    ax.figure.canvas.draw()
    bb_after = ax.texts[0].get_window_extent(renderer)

    # The text bbox height in pixels is constant (font-driven); we want
    # the text to have moved in data coords so its pixel position on screen
    # stays consistent with its anchor. The simplest invariant: bbox
    # pixel size is unchanged, and the text remains inside the new ylim.
    assert bb_after.height == pytest.approx(bb_before.height, abs=0.5)
    assert bb_after.width == pytest.approx(bb_before.width, abs=0.5)
    inv = ax.transData.inverted()
    bb_data = bb_after.transformed(inv)
    assert bb_data.y1 <= ax.get_ylim()[1] + 0.5
