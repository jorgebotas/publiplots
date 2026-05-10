"""Tests for the ``rotation=`` support in pp.annotate.

matplotlib applies ``(ha, va)`` to the **post-rotation** bbox, so the
``(ha, va)`` returned by ``resolve_anchor`` for a given anchor kind
already positions the text correctly at any rotation — ``va='bottom'``
at rotation=90° still means "anchor at text bbox's bottom edge", which
visually puts the text to the right of the anchor.

This file covers:
    * top-level ``annotate(rotation=...)`` validation (NaN / Inf raise),
    * dispatcher forwarding to each rendered Text artist,
    * ``rotation`` smuggled via ``**text_kws`` doesn't crash,
    * end-to-end ``pp.barplot`` / ``pp.boxplot`` / ``pp.pointplot``
      integration: rotation applied, anchor geometry preserved from the
      rotation=0 baseline, inverted y-axis orientation preserved on
      horizontal barplots, categorical axis expanded under rotation to
      keep labels inside the axes frame.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate import annotate
from publiplots.annotate._cache import BarRecord, BarValueMeta
from publiplots.annotate.bar_values import _bar_values_strategy


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# -----------------------------------------------------------------------------
# Top-level validation
# -----------------------------------------------------------------------------

def test_non_finite_rotation_raises():
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        annotate(ax, kind="bar_values", rotation=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        annotate(ax, kind="bar_values", rotation=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        annotate(ax, kind="bar_values", rotation=float("-inf"))


# -----------------------------------------------------------------------------
# Dispatcher forwarding
# -----------------------------------------------------------------------------

def test_rotation_forwarded_to_text_artist():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0])
    fig.canvas.draw()
    texts = annotate(ax, kind="bar_values", rotation=90)
    assert len(texts) == 3
    for t in texts:
        assert t.get_rotation() == pytest.approx(90.0)


def test_rotation_non_right_angle_forwarded():
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    fig.canvas.draw()
    texts = annotate(ax, kind="bar_values", rotation=45)
    assert texts[0].get_rotation() == pytest.approx(45.0)


def test_rotation_in_text_kws_does_not_crash_strategy():
    """Each strategy pops ``rotation`` from ``text_kws`` defensively, so a
    caller who smuggles it through the splat (same-name keyword binding) still
    gets a clean call and the expected Text artist.
    """
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    fig.canvas.draw()
    texts = _bar_values_strategy(
        ax, fmt=".1f", anchor="outside", offset=0.0, color="auto", pad=0.0,
        **{"rotation": 30},
    )
    assert len(texts) == 1
    assert texts[0].get_rotation() == pytest.approx(30.0)


# -----------------------------------------------------------------------------
# Vertical barplot end-to-end
# -----------------------------------------------------------------------------

def _simple_vbar_df():
    return pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
    })


def _simple_hbar_df():
    return pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
    })


@pytest.mark.parametrize("rotation", [0.0, 45.0, 90.0, 180.0, 270.0])
def test_barplot_annotate_rotation_preserves_anchor_geometry(rotation):
    """For anchor='outside' on a positive vertical bar, resolve_anchor returns
    (ha='center', va='bottom'). matplotlib applies these to the post-rotation
    bbox, so the anchor stays at the bbox's lower edge at any rotation — the
    label always extends "up" from the bar top in the post-rotation frame,
    which visually places it above the bar (or beside it, when rotated).
    """
    ax = pp.barplot(data=_simple_vbar_df(), x="category", y="value",
                    annotate={"rotation": rotation})
    assert len(ax.texts) == 3
    for t in ax.texts:
        assert t.get_rotation() == pytest.approx(rotation)
        assert t.get_ha() == "center"
        assert t.get_va() == "bottom"


def test_barplot_annotate_rotation_90_bbox_sits_above_bar_top():
    """Regression guard for the visual invariant: rotation=90° with default
    anchor keeps the text's bbox ABOVE each bar's top edge (bbox.y0 ≈ bar
    top + offset), so the label doesn't overlap the bar.
    """
    df = _simple_vbar_df()
    ax = pp.barplot(data=df, x="category", y="value",
                    annotate={"rotation": 90, "offset": 1.0})
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    for t, bar_top in zip(ax.texts, [1.0, 2.0, 3.0]):
        bb = t.get_window_extent(renderer).transformed(inv)
        assert bb.y0 >= bar_top, (
            f"rotated label bbox starts at y={bb.y0}, should be >= bar top {bar_top}"
        )


# -----------------------------------------------------------------------------
# Horizontal barplot end-to-end
# -----------------------------------------------------------------------------

def test_barplot_annotate_rotation_horizontal_bar_preserves_anchor():
    """For anchor='outside' on a positive horizontal bar, resolve_anchor
    returns (ha='left', va='center'). Those stay unchanged under rotation.
    """
    ax = pp.barplot(data=_simple_hbar_df(), x="value", y="category",
                    annotate={"rotation": 90})
    for t in ax.texts:
        assert t.get_rotation() == pytest.approx(90.0)
        assert t.get_ha() == "left"
        assert t.get_va() == "center"


def test_barplot_annotate_rotation_preserves_inverted_yaxis():
    """Seaborn draws horizontal barplots with an inverted y-axis
    (ylim[0] > ylim[1]) so category 0 sits at the top. Rotation-triggered
    categorical-axis expansion must not flip that orientation or clip
    bars out of view.
    """
    df = _simple_hbar_df()
    ax = pp.barplot(data=df, x="value", y="category",
                    annotate={"rotation": 90})
    lo, hi = ax.get_ylim()
    assert lo > hi, (
        f"horizontal barplot ylim lost its seaborn-default inversion; "
        f"got ({lo}, {hi})"
    )
    y_min, y_max = min(lo, hi), max(lo, hi)
    for t in ax.texts:
        _, ty = t.get_position()
        assert y_min <= ty <= y_max, (
            f"text anchor y={ty} falls outside ylim=({lo}, {hi})"
        )


# -----------------------------------------------------------------------------
# Two-axis limit expansion
# -----------------------------------------------------------------------------

def _owned_meta_axes(xlim, ylim, figsize=(4, 4)):
    """Publiplots-owned bar meta on a clean axes with tight lims."""
    fig, ax = plt.subplots(figsize=figsize)
    rects = ax.bar([0, 1], [10.0, 10.0])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.canvas.draw()
    bars = [
        BarRecord(patch=r, value=10.0, err_low=None, err_high=None, hue_color=None)
        for r in rects
    ]
    ax._publiplots_bar_meta = BarValueMeta(
        orient="v", bars=bars, errorbar_kind=None,
        hue_active=False, owner_is_publiplots=True,
    )
    return fig, ax


def test_barplot_annotate_rotation_limits_expand_both_axes():
    """Rotation swaps the label bbox's width and height relative to the
    axes, so the value-axis expansion at rotation=90° must exceed what
    rotation=0° would need (the tall-height-of-rotated-text dominates).
    Both axes always get expanded when needed.
    """
    _fig0, ax0 = _owned_meta_axes(xlim=(0.7, 1.1), ylim=(0, 10))
    _bar_values_strategy(
        ax0, fmt="{:.1f}", anchor="outside", offset=1.0,
        color="auto", pad=1.0, rotation=0.0, fontsize=80,
    )
    ylim_rot0 = ax0.get_ylim()

    _fig90, ax90 = _owned_meta_axes(xlim=(0.7, 1.1), ylim=(0, 10))
    _bar_values_strategy(
        ax90, fmt="{:.1f}", anchor="outside", offset=1.0,
        color="auto", pad=1.0, rotation=90.0, fontsize=80,
    )
    ylim_rot90 = ax90.get_ylim()

    assert ylim_rot0[1] > 10.0, f"rotation=0 ylim did not expand: {ylim_rot0}"
    # rot=90 rotates the label so its *long* side now aligns with the
    # value axis → y-axis must expand more at rot=90 than at rot=0.
    assert ylim_rot90[1] > ylim_rot0[1], (
        f"rotation=90 should expand y-axis more than rotation=0; "
        f"got rot0={ylim_rot0[1]:.3f}, rot90={ylim_rot90[1]:.3f}"
    )


def test_barplot_annotate_long_label_expands_categorical_axis():
    """Without rotation, a long label on an outer bar can still spill past
    the categorical axis limits and must trigger x-expansion.
    """
    _fig, ax = _owned_meta_axes(xlim=(-0.5, 1.5), ylim=(0, 10))
    _bar_values_strategy(
        ax, fmt="{:.6f} units", anchor="outside", offset=1.0,
        color="auto", pad=1.0, rotation=0.0, fontsize=20,
    )
    xlim = ax.get_xlim()
    assert xlim[0] < -0.5 or xlim[1] > 1.5, (
        f"long label should expand x-axis past bars; got {xlim}"
    )


# -----------------------------------------------------------------------------
# Boxplot / pointplot end-to-end — anchor geometry preserved
# -----------------------------------------------------------------------------

def _box_df():
    rng = np.random.default_rng(0)
    rows = []
    for g, base in zip(("A", "B", "C"), (1.0, 2.0, 3.0)):
        for v in rng.normal(base, 0.5, 30):
            rows.append({"g": g, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = df["g"].astype("category")
    return df


def test_boxplot_annotate_rotation_90_preserves_anchor():
    """Default box_stats anchor='right' returns (ha='left', va='center')."""
    ax = pp.boxplot(data=_box_df(), x="g", y="y",
                    annotate={"rotation": 90})
    assert len(ax.texts) == 3
    for t in ax.texts:
        assert t.get_rotation() == pytest.approx(90.0)
        assert t.get_ha() == "left"
        assert t.get_va() == "center"


def _point_df():
    rng = np.random.default_rng(0)
    rows = []
    for t, base in zip(("t1", "t2", "t3"), (1.0, 2.5, 3.2)):
        for v in rng.normal(base, 0.3, 10):
            rows.append({"time": t, "v": float(v)})
    df = pd.DataFrame(rows)
    df["time"] = df["time"].astype("category")
    return df


def test_pointplot_annotate_rotation_90_preserves_anchor():
    """Default point_values anchor='top' returns (ha='center', va='bottom')."""
    ax = pp.pointplot(data=_point_df(), x="time", y="v",
                      errorbar="se", annotate={"rotation": 90})
    assert len(ax.texts) == 3
    for t in ax.texts:
        assert t.get_rotation() == pytest.approx(90.0)
        assert t.get_ha() == "center"
        assert t.get_va() == "bottom"
