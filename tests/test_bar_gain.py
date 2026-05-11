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


def test_gain_single_dim_produces_two_rects_per_cat_when_distinct():
    df = _simple_gain_df()
    ax = pp.barplot(data=df, x="metric", y="score", hue="model",
                    multiple="gain", errorbar=None)
    # AUC: 0.80/0.90 → 2 rects; F1: 0.75/0.82 → 2 rects;
    # Recall: 0.88/0.85 → 2 rects. Total 6.
    assert len(_bars(ax)) == 6


def test_gain_base_y_zero_top_stacked_on_min():
    df = _simple_gain_df()
    ax = pp.barplot(data=df, x="metric", y="score", hue="model",
                    multiple="gain", errorbar=None)
    # Group rects by x-center; each group = 2 rects, base at y=0, delta
    # at y=min.
    groups: dict = {}
    for r in _bars(ax):
        k = round(r.get_x() + r.get_width() / 2, 3)
        groups.setdefault(k, []).append(r)
    for k, segs in groups.items():
        assert len(segs) == 2
        segs.sort(key=lambda r: r.get_y())
        lo = segs[0].get_height()
        assert segs[0].get_y() == pytest.approx(0.0)
        assert segs[1].get_y() == pytest.approx(lo)
        hi = segs[1].get_y() + segs[1].get_height()
        # hi matches one of the two model scores at that metric
        metric = ["AUC", "F1", "Recall"][int(round(k))]
        scores = df.loc[df["metric"] == metric, "score"].tolist()
        assert hi == pytest.approx(max(scores))
        assert lo == pytest.approx(min(scores))


def test_gain_base_color_is_loser_top_is_winner():
    df = _simple_gain_df()
    palette = {"Baseline": "#ff0000", "Proposed": "#00ff00"}
    ax = pp.barplot(data=df, x="metric", y="score", hue="model",
                    multiple="gain", errorbar=None, palette=palette)

    groups: dict = {}
    for r in _bars(ax):
        k = round(r.get_x() + r.get_width() / 2, 3)
        groups.setdefault(k, []).append(r)

    # AUC (x=0): Baseline=0.80, Proposed=0.90 → Baseline loses (red base),
    # Proposed wins (green top).
    # F1 (x=1): same direction (Baseline loses).
    # Recall (x=2): Baseline=0.88, Proposed=0.85 → Baseline WINS → green
    # top, red base flip: red is now on top, green on bottom.
    auc_segs = sorted(groups[0.0], key=lambda r: r.get_y())
    assert tuple(auc_segs[0].get_facecolor()[:3]) == to_rgba("#ff0000")[:3]
    assert tuple(auc_segs[1].get_facecolor()[:3]) == to_rgba("#00ff00")[:3]

    recall_segs = sorted(groups[2.0], key=lambda r: r.get_y())
    # Baseline wins Recall → base is Proposed (loser) green, top is
    # Baseline (winner) red.
    assert tuple(recall_segs[0].get_facecolor()[:3]) == to_rgba("#00ff00")[:3]
    assert tuple(recall_segs[1].get_facecolor()[:3]) == to_rgba("#ff0000")[:3]


def test_gain_tie_produces_single_rect():
    df = pd.DataFrame({
        "metric": pd.Categorical(["Tied", "Tied", "Distinct", "Distinct"]),
        "model": pd.Categorical(["A", "B"] * 2, categories=["A", "B"]),
        "score": [0.8, 0.8, 0.7, 0.9],
    })
    ax = pp.barplot(data=df, x="metric", y="score", hue="model",
                    multiple="gain", errorbar=None)
    # Tied → 1 rect; Distinct → 2 rects. Total 3.
    assert len(_bars(ax)) == 3


def test_gain_tie_uses_hue_order_first_color():
    df = pd.DataFrame({
        "metric": pd.Categorical(["Tied"] * 2),
        "model": pd.Categorical(["A", "B"], categories=["A", "B"]),
        "score": [0.5, 0.5],
    })
    palette = {"A": "#ff0000", "B": "#00ff00"}
    ax = pp.barplot(data=df, x="metric", y="score", hue="model",
                    multiple="gain", errorbar=None, palette=palette)
    rects = _bars(ax)
    assert len(rects) == 1
    assert tuple(rects[0].get_facecolor()[:3]) == to_rgba("#ff0000")[:3]
    assert rects[0].get_height() == pytest.approx(0.5)


def test_gain_annotate_labels_show_absolute_values():
    df = _simple_gain_df()
    ax = pp.barplot(data=df, x="metric", y="score", hue="model",
                    multiple="gain", errorbar=None,
                    annotate={"fmt": ".2f"})
    # 6 rects → 6 labels. Labels show absolute values (0.80, 0.90, 0.75,
    # 0.82, 0.85, 0.88) — never the delta (0.10, 0.07, 0.03).
    assert len(ax.texts) == 6
    texts = {t.get_text() for t in ax.texts}
    assert texts == {"0.80", "0.90", "0.75", "0.82", "0.85", "0.88"}


def test_gain_annotate_tie_single_label():
    df = pd.DataFrame({
        "metric": pd.Categorical(["Tied"] * 2),
        "model": pd.Categorical(["A", "B"], categories=["A", "B"]),
        "score": [0.5, 0.5],
    })
    ax = pp.barplot(data=df, x="metric", y="score", hue="model",
                    multiple="gain", errorbar=None,
                    annotate={"fmt": ".2f"})
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() == "0.50"


def test_gain_missing_level_at_cat_raises():
    # "Recall" only has Baseline data — Proposed missing there.
    df = pd.DataFrame({
        "metric": pd.Categorical(["AUC", "AUC", "Recall"]),
        "model": pd.Categorical(["Baseline", "Proposed", "Baseline"],
                                categories=["Baseline", "Proposed"]),
        "score": [0.80, 0.90, 0.88],
    })
    with pytest.raises(ValueError, match="missing"):
        pp.barplot(data=df, x="metric", y="score", hue="model",
                   multiple="gain", errorbar=None)


def test_gain_without_hue_or_hatch_raises():
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "B", "C"]),
        "val": [1.0, 2.0, 3.0],
    })
    with pytest.raises(ValueError, match="stack'\\|'fill"):
        pp.barplot(data=df, x="cat", y="val",
                   multiple="gain", errorbar=None)


def test_gain_with_errorbar_warns():
    df = _simple_gain_df()
    with pytest.warns(UserWarning, match="errorbars"):
        pp.barplot(data=df, x="metric", y="score", hue="model",
                   multiple="gain", errorbar="se")
