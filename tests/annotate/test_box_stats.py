"""Integration tests for pp.boxplot(..., annotate=...) + box_stats strategy."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate._cache import BoxStatsMeta


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _box_df(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for g, base in zip(("A", "B", "C"), (1.0, 2.0, 3.0)):
        for v in rng.normal(base, 0.5, 30):
            rows.append({"g": g, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = df["g"].astype("category")
    return df


def _hue_box_df(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for grp in ("ctrl", "trt"):
        for g, base in zip(("A", "B"), (1.0, 2.0)):
            bump = 0 if grp == "ctrl" else 0.5
            for v in rng.normal(base + bump, 0.4, 30):
                rows.append({"g": g, "grp": grp, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = df["g"].astype("category")
    df["grp"] = df["grp"].astype("category")
    return df


# ----------------------------------------------------------------------------
# Meta building
# ----------------------------------------------------------------------------

def test_boxplot_annotate_attaches_meta():
    fig, ax = pp.boxplot(data=_box_df(), x="g", y="y", annotate=True)
    assert isinstance(ax._publiplots_box_meta, BoxStatsMeta)
    assert ax._publiplots_box_meta.owner_is_publiplots is True


def test_boxplot_annotate_default_labels_median():
    fig, ax = pp.boxplot(data=_box_df(), x="g", y="y", annotate=True)
    # 3 boxes × 1 stat (median) = 3 labels
    assert len(ax.texts) == 3


def test_boxplot_annotate_multi_stats():
    fig, ax = pp.boxplot(data=_box_df(), x="g", y="y",
                         annotate={"stats": ["median", "q1", "q3"]})
    assert len(ax.texts) == 9


def test_boxplot_annotate_stats_compute_correctly():
    fig, ax = pp.boxplot(data=_box_df(seed=42), x="g", y="y",
                         annotate={"stats": ["median", "q1", "q3"]})
    meta = ax._publiplots_box_meta
    for box in meta.boxes:
        assert box.stats["q1"] < box.stats["median"] < box.stats["q3"]
        assert box.stats["whisker_low"] <= box.stats["q1"]
        assert box.stats["whisker_high"] >= box.stats["q3"]


def test_boxplot_annotate_with_hue_has_hue_active():
    fig, ax = pp.boxplot(data=_hue_box_df(), x="g", y="y", hue="grp",
                         annotate=True)
    meta = ax._publiplots_box_meta
    assert meta.hue_active is True
    # 2 groups × 2 hues × 1 stat = 4
    assert len(ax.texts) == 4


# ----------------------------------------------------------------------------
# Anchor positioning
# ----------------------------------------------------------------------------

def test_boxplot_annotate_anchor_right_default():
    fig, ax = pp.boxplot(data=_box_df(), x="g", y="y", annotate=True)
    for t in ax.texts:
        assert t.get_ha() == "left"
        assert t.get_va() == "center"


def test_boxplot_annotate_anchor_top():
    fig, ax = pp.boxplot(data=_box_df(), x="g", y="y",
                         annotate={"anchor": "top"})
    meta = ax._publiplots_box_meta
    for t, box in zip(ax.texts, meta.boxes):
        _, y = t.get_position()
        assert y > box.stats["median"]
        assert t.get_va() == "bottom"


def test_boxplot_annotate_anchor_center_on_median():
    fig, ax = pp.boxplot(data=_box_df(), x="g", y="y",
                         annotate={"anchor": "center"})
    meta = ax._publiplots_box_meta
    for t, box in zip(ax.texts, meta.boxes):
        x, y = t.get_position()
        assert x == pytest.approx(box.center_pos)
        assert y == pytest.approx(box.stats["median"])


def test_boxplot_annotate_invalid_anchor_raises():
    with pytest.raises(ValueError, match="box_stats anchor"):
        pp.boxplot(data=_box_df(), x="g", y="y",
                   annotate={"anchor": "inside"})


def test_boxplot_annotate_invalid_stat_raises():
    with pytest.raises(ValueError, match="unknown box stat"):
        pp.boxplot(data=_box_df(), x="g", y="y",
                   annotate={"stats": ["median", "bogus"]})


# ----------------------------------------------------------------------------
# Format + text kws
# ----------------------------------------------------------------------------

def test_boxplot_annotate_fmt_forwarded():
    fig, ax = pp.boxplot(data=_box_df(), x="g", y="y",
                         annotate={"fmt": ".1f"})
    for t in ax.texts:
        parts = t.get_text().split(".")
        assert len(parts) == 2
        assert len(parts[1]) == 1


def test_boxplot_annotate_text_kws_forwarded():
    fig, ax = pp.boxplot(data=_box_df(), x="g", y="y",
                         annotate={"fontsize": 14, "fontweight": "bold"})
    for t in ax.texts:
        assert t.get_fontsize() == 14
        assert t.get_fontweight() == "bold"


# ----------------------------------------------------------------------------
# Foreign-axes warning
# ----------------------------------------------------------------------------

def test_box_stats_foreign_axes_warns():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 2, 3])
    with pytest.warns(UserWarning, match="pp.boxplot"):
        result = pp.annotate(ax, kind="box_stats")
    assert result == []


# ----------------------------------------------------------------------------
# Violinplot integration (shares BoxStatsMeta + box_stats strategy)
# ----------------------------------------------------------------------------

def test_violinplot_annotate_attaches_meta():
    fig, ax = pp.violinplot(data=_box_df(), x="g", y="y", annotate=True)
    assert isinstance(ax._publiplots_box_meta, BoxStatsMeta)
    assert len(ax.texts) == 3


def test_violinplot_annotate_multi_stats():
    fig, ax = pp.violinplot(data=_box_df(), x="g", y="y",
                            annotate={"stats": ["median", "q1", "q3"]})
    assert len(ax.texts) == 9


def test_violinplot_annotate_stats_match_data():
    """Violin stats must agree with raw-data medians."""
    df = _box_df(seed=42)
    fig, ax = pp.violinplot(data=df, x="g", y="y", annotate=True)
    meta = ax._publiplots_box_meta
    expected_medians = [
        float(df.loc[df["g"] == g, "y"].median()) for g in ("A", "B", "C")
    ]
    for box, expected in zip(meta.boxes, expected_medians):
        assert box.stats["median"] == pytest.approx(expected)
