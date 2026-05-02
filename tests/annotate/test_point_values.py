"""Integration tests for pp.pointplot(..., annotate=...) + point_values strategy."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate._cache import PointValueMeta


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _multi_sample_df():
    rng = np.random.default_rng(0)
    rows = []
    for t in ("t1", "t2", "t3"):
        base = {"t1": 1.0, "t2": 2.5, "t3": 3.2}[t]
        for v in rng.normal(base, 0.3, 10):
            rows.append({"time": t, "v": float(v)})
    df = pd.DataFrame(rows)
    df["time"] = df["time"].astype("category")
    return df


def _hue_df():
    rng = np.random.default_rng(0)
    rows = []
    for grp in ("A", "B"):
        for t in ("t1", "t2", "t3"):
            base = {"t1": 1.0, "t2": 2.5, "t3": 3.2}[t] + (0 if grp == "A" else 0.5)
            for v in rng.normal(base, 0.3, 10):
                rows.append({"group": grp, "time": t, "v": float(v)})
    df = pd.DataFrame(rows)
    df["time"] = df["time"].astype("category")
    df["group"] = df["group"].astype("category")
    return df


# ----------------------------------------------------------------------------
# Owned-axes meta building
# ----------------------------------------------------------------------------

def test_pointplot_annotate_true_attaches_meta():
    ax = pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                           errorbar="se", annotate=True)
    assert isinstance(ax._publiplots_point_meta, PointValueMeta)
    assert ax._publiplots_point_meta.owner_is_publiplots is True


def test_pointplot_annotate_draws_one_label_per_point():
    ax = pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                           errorbar="se", annotate=True)
    assert len(ax.texts) == 3


def test_pointplot_annotate_with_hue_produces_two_series():
    ax = pp.pointplot(data=_hue_df(), x="time", y="v", hue="group",
                           errorbar="se", annotate=True)
    meta = ax._publiplots_point_meta
    assert meta.hue_active is True
    # Two hues × three time points = 6 points.
    assert len(meta.points) == 6
    assert len(ax.texts) == 6


def test_pointplot_annotate_errorbars_from_drawn_artists():
    ax = pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                           errorbar="se", annotate=True)
    meta = ax._publiplots_point_meta
    for p in meta.points:
        assert p.err_high is not None
        assert p.err_low is not None
        assert p.err_low < p.value < p.err_high


def test_pointplot_annotate_errorbars_pair_per_hue():
    """With hue and no dodge, two points share an x-position. Each must
    be paired with its own errorbar, not the first-at-x wins.

    Regression: a naive x-only matcher gives both hue series the same
    err_high, which causes labels to stack at identical y-coords.
    """
    ax = pp.pointplot(data=_hue_df(), x="time", y="v", hue="group",
                           errorbar="se", annotate=True)
    meta = ax._publiplots_point_meta
    # Group by x-position; within each x, err_high values must differ
    # across the two hues (since their means differ by ~0.5).
    by_x = {}
    for p in meta.points:
        by_x.setdefault(p.xy[0], []).append(p.err_high)
    for px, heights in by_x.items():
        assert len(heights) == 2, f"expected 2 hues at x={px}"
        assert heights[0] != heights[1], (
            f"x={px}: both hues got same err_high {heights[0]} — matcher is wrong"
        )


def test_pointplot_annotate_errorbar_none_all_none():
    ax = pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                           errorbar=None, annotate=True)
    meta = ax._publiplots_point_meta
    for p in meta.points:
        assert p.err_low is None
        assert p.err_high is None


# ----------------------------------------------------------------------------
# Anchor positioning
# ----------------------------------------------------------------------------

def test_pointplot_annotate_default_anchor_is_top():
    """Labels by default sit above the point (past errorbar cap)."""
    ax = pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                           errorbar="se", annotate=True)
    meta = ax._publiplots_point_meta
    for text, point in zip(ax.texts, meta.points):
        _, y = text.get_position()
        assert y >= point.err_high
        assert text.get_va() == "bottom"


def test_pointplot_annotate_anchor_bottom_flips_below():
    ax = pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                           errorbar="se",
                           annotate={"anchor": "bottom"})
    meta = ax._publiplots_point_meta
    for text, point in zip(ax.texts, meta.points):
        _, y = text.get_position()
        assert y <= point.err_low
        assert text.get_va() == "top"


def test_pointplot_annotate_anchor_right_places_beside_marker():
    ax = pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                           annotate={"anchor": "right"})
    for text in ax.texts:
        assert text.get_ha() == "left"
        assert text.get_va() == "center"


def test_pointplot_annotate_anchor_center_overlays_marker():
    ax = pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                           annotate={"anchor": "center"})
    meta = ax._publiplots_point_meta
    for text, point in zip(ax.texts, meta.points):
        x, y = text.get_position()
        assert x == pytest.approx(point.xy[0])
        assert y == pytest.approx(point.xy[1])
        assert text.get_ha() == "center"
        assert text.get_va() == "center"


def test_pointplot_annotate_text_zorder_above_markers():
    """pp.pointplot draws its double-layer markers at zorder ~99-100;
    label text must sit above them (especially relevant for anchor='center').
    """
    ax = pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                           annotate={"anchor": "center"})
    marker_zorders = [
        ln.get_zorder() for ln in ax.lines if ln.get_marker() not in (None, "None")
    ]
    max_marker_zorder = max(marker_zorders) if marker_zorders else 0
    for text in ax.texts:
        assert text.get_zorder() > max_marker_zorder


def test_pointplot_annotate_respects_user_zorder():
    """User-supplied zorder in annotate dict wins over the default bump."""
    ax = pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                           annotate={"anchor": "center", "zorder": 5})
    for text in ax.texts:
        assert text.get_zorder() == 5


def test_pointplot_annotate_invalid_anchor_raises():
    with pytest.raises(ValueError, match="point_values anchor"):
        pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                     annotate={"anchor": "inside"})


def test_pointplot_annotate_label_positions_finite():
    """Labels on a pointplot must have finite (non-NaN) positions even with
    single-sample groups (NaN errorbars)."""
    df = pd.DataFrame({
        "t": pd.Categorical(["t1", "t2", "t3"]),
        "v": [1.0, 2.0, 3.0],
    })
    ax = pp.pointplot(data=df, x="t", y="v", errorbar="se", annotate=True)
    for text in ax.texts:
        x, y = text.get_position()
        assert not math.isnan(float(x))
        assert not math.isnan(float(y))


# ----------------------------------------------------------------------------
# Foreign-axes warning
# ----------------------------------------------------------------------------

def test_pointplot_annotate_foreign_axes_warns():
    """kind='point_values' on a non-pp.pointplot axes warns and returns []."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1.0, 2.0, 3.0], "o")
    with pytest.warns(UserWarning, match="pp.pointplot-owned"):
        result = pp.annotate(ax, kind="point_values")
    assert result == []
    assert len(ax.texts) == 0


# ----------------------------------------------------------------------------
# Format forwarding
# ----------------------------------------------------------------------------

def test_pointplot_annotate_fmt_forwarded():
    ax = pp.pointplot(data=_multi_sample_df(), x="time", y="v",
                           errorbar="se", annotate={"fmt": ".1f"})
    for text in ax.texts:
        # one decimal place
        assert "." in text.get_text()
        assert len(text.get_text().split(".")[1]) == 1
