"""Tests for the `rotation=` support in pp.annotate.

Covers:
    * `remap_alignment_for_rotation` pure-helper truth table at 0/90/180/270°,
    * special-case handling of `va="baseline"`, negative and wrapped rotations,
      near-right-angle tolerance, and non-right-angle pass-through,
    * the top-level `annotate(...)` validation of `rotation` (NaN/Inf raise),
    * dispatcher forwarding of `rotation` to each rendered Text artist,
    * text_kws smuggling of `rotation` (top-level wins, no TypeError),
    * end-to-end `pp.barplot` / `pp.boxplot` / `pp.pointplot` annotate=
      dict integration including horizontal barplots and two-axis limit
      expansion under rotation.
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
from publiplots.annotate._positioning import remap_alignment_for_rotation
from publiplots.annotate.bar_values import _bar_values_strategy


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# -----------------------------------------------------------------------------
# `remap_alignment_for_rotation` — pure helper, no matplotlib needed
# -----------------------------------------------------------------------------

_ALL_ANCHORS = [
    ("left", "top"),
    ("center", "top"),
    ("right", "top"),
    ("left", "center"),
    ("center", "center"),
    ("right", "center"),
    ("left", "bottom"),
    ("center", "bottom"),
    ("right", "bottom"),
]


def test_rotation_0_is_identity():
    for ha, va in _ALL_ANCHORS:
        assert remap_alignment_for_rotation(ha, va, 0) == (ha, va)


def test_rotation_90_truth_table():
    expected = {
        ("left", "top"): ("right", "top"),
        ("center", "top"): ("right", "center"),
        ("right", "top"): ("right", "bottom"),
        ("left", "center"): ("center", "top"),
        ("center", "center"): ("center", "center"),
        ("right", "center"): ("center", "bottom"),
        ("left", "bottom"): ("left", "top"),
        ("center", "bottom"): ("left", "center"),
        ("right", "bottom"): ("left", "bottom"),
    }
    for (ha, va), out in expected.items():
        assert remap_alignment_for_rotation(ha, va, 90) == out, (ha, va)


def test_rotation_180_truth_table():
    expected = {
        ("left", "top"): ("right", "bottom"),
        ("center", "top"): ("center", "bottom"),
        ("right", "top"): ("left", "bottom"),
        ("left", "center"): ("right", "center"),
        ("center", "center"): ("center", "center"),
        ("right", "center"): ("left", "center"),
        ("left", "bottom"): ("right", "top"),
        ("center", "bottom"): ("center", "top"),
        ("right", "bottom"): ("left", "top"),
    }
    for (ha, va), out in expected.items():
        assert remap_alignment_for_rotation(ha, va, 180) == out, (ha, va)


def test_rotation_270_truth_table():
    expected = {
        ("left", "top"): ("left", "bottom"),
        ("center", "top"): ("left", "center"),
        ("right", "top"): ("left", "top"),
        ("left", "center"): ("center", "bottom"),
        ("center", "center"): ("center", "center"),
        ("right", "center"): ("center", "top"),
        ("left", "bottom"): ("right", "bottom"),
        ("center", "bottom"): ("right", "center"),
        ("right", "bottom"): ("right", "top"),
    }
    for (ha, va), out in expected.items():
        assert remap_alignment_for_rotation(ha, va, 270) == out, (ha, va)


def test_non_right_angle_passthrough():
    # 45° isn't a right-angle multiple; the helper returns the input unchanged.
    for ha, va in [("left", "top"), ("center", "bottom"), ("right", "center")]:
        assert remap_alignment_for_rotation(ha, va, 45) == (ha, va)


def test_baseline_normalized_to_bottom():
    # va="baseline" is treated as va="bottom" before the table lookup.
    assert (
        remap_alignment_for_rotation("center", "baseline", 90)
        == remap_alignment_for_rotation("center", "bottom", 90)
        == ("left", "center")
    )


def test_negative_rotation_normalized():
    # -90° wraps to 270° → same remap as 270°.
    assert (
        remap_alignment_for_rotation("center", "bottom", -90)
        == remap_alignment_for_rotation("center", "bottom", 270)
        == ("right", "center")
    )


def test_wrapped_rotation_normalized():
    # 450° wraps to 90° → same remap as 90°.
    assert (
        remap_alignment_for_rotation("center", "bottom", 450)
        == remap_alignment_for_rotation("center", "bottom", 90)
        == ("left", "center")
    )


def test_rotation_tolerance():
    # Near-right-angle (within 1e-6) treated as the exact right angle.
    assert remap_alignment_for_rotation("center", "bottom", 90.0000001) == ("left", "center")
    assert remap_alignment_for_rotation("center", "bottom", 89.9999999) == ("left", "center")


def test_non_finite_rotation_raises():
    # The helper itself is pure and doesn't guard, but the top-level
    # annotate(...) must reject NaN / Inf.
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


def test_rotation_via_text_kws_still_works():
    # `rotation` is now a first-class top-level kwarg; we exercise the
    # common non-right-angle case end-to-end to confirm the helper doesn't
    # clobber it.
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    fig.canvas.draw()
    texts = annotate(ax, kind="bar_values", rotation=45)
    assert texts[0].get_rotation() == pytest.approx(45.0)


def test_rotation_in_text_kws_does_not_crash_strategy():
    """Defensive: each strategy pops ``rotation`` from ``text_kws`` at
    function entry. Even though the strategies' signatures now capture
    ``rotation`` as a keyword-only param (so a duplicate would normally
    raise ``TypeError`` at call-site, not inside the body), exercise the
    dead-but-intended-defensive ``text_kws.pop("rotation", None)`` path by
    confirming that calling the strategy with ``rotation`` coming in through
    ``**text_kws`` alone succeeds and produces the expected artist.
    """
    fig, ax = plt.subplots()
    ax.bar([0], [1.0])
    fig.canvas.draw()
    # Only route rotation through the **text_kws splat; no explicit
    # ``rotation=`` keyword. Python binds the splat to the keyword-only
    # `rotation` param (since names match), so no crash and the Text artist
    # carries the splatted value.
    texts = _bar_values_strategy(
        ax, fmt=".1f", anchor="outside", offset=0.0, color="auto", pad=0.0,
        **{"rotation": 30},
    )
    assert len(texts) == 1
    # No TypeError escaped, which is the real guarantee under test.
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
    # Same frame, but we'll plot with x=numeric / y=categorical for horizontal.
    return pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
    })


def test_barplot_annotate_rotation_90_remaps_ha_va():
    # Default anchor='outside' on vertical bars → (ha='center', va='bottom').
    # Remap at 90° → (ha='left', va='center').
    ax = pp.barplot(data=_simple_vbar_df(), x="category", y="value",
                    annotate={"rotation": 90})
    assert len(ax.texts) == 3
    for t in ax.texts:
        assert t.get_rotation() == pytest.approx(90.0)
        assert t.get_ha() == "left"
        assert t.get_va() == "center"


def test_barplot_annotate_rotation_270_remaps_ha_va():
    # (center, bottom) @ 270° → (right, center).
    ax = pp.barplot(data=_simple_vbar_df(), x="category", y="value",
                    annotate={"rotation": 270})
    for t in ax.texts:
        assert t.get_rotation() == pytest.approx(270.0)
        assert t.get_ha() == "right"
        assert t.get_va() == "center"


def test_barplot_annotate_rotation_180():
    # (center, bottom) @ 180° → (center, top).
    ax = pp.barplot(data=_simple_vbar_df(), x="category", y="value",
                    annotate={"rotation": 180})
    for t in ax.texts:
        assert t.get_rotation() == pytest.approx(180.0)
        assert t.get_ha() == "center"
        assert t.get_va() == "top"


def test_barplot_annotate_rotation_0_is_unchanged():
    # Regression guard: default (no rotation kwarg) keeps the pre-rotation
    # anchor geometry exactly — outside-top-of-bar => (center, bottom).
    ax = pp.barplot(data=_simple_vbar_df(), x="category", y="value",
                    annotate=True)
    for t in ax.texts:
        assert t.get_rotation() == pytest.approx(0.0)
        assert t.get_ha() == "center"
        assert t.get_va() == "bottom"


def test_barplot_annotate_rotation_horizontal_bar():
    # Horizontal barplot: x=numeric, y=categorical. resolve_anchor for
    # anchor='outside' on a positive horizontal bar returns (ha='left', va='center').
    # Remap at 90° → (center, top).
    ax = pp.barplot(data=_simple_hbar_df(), x="value", y="category",
                    annotate={"rotation": 90})
    for t in ax.texts:
        assert t.get_rotation() == pytest.approx(90.0)
        assert t.get_ha() == "center"
        assert t.get_va() == "top"


def test_barplot_annotate_rotation_preserves_inverted_yaxis():
    # Seaborn draws horizontal barplots with an inverted y-axis
    # (ylim[0] > ylim[1]) so category 0 sits at the top. Rotation-triggered
    # categorical-axis expansion must not flip that orientation or clip
    # existing bars out of view.
    df = _simple_hbar_df()
    ax = pp.barplot(data=df, x="value", y="category",
                    annotate={"rotation": 90})
    lo, hi = ax.get_ylim()
    assert lo > hi, (
        "horizontal barplot ylim lost its seaborn-default inversion after "
        f"rotation expansion; got ({lo}, {hi})"
    )
    # All text anchor y-coords (0, 1, 2, 3) must lie inside the final ylim
    # range, otherwise bar A (y=0) renders below the axis frame.
    y_min, y_max = min(lo, hi), max(lo, hi)
    for t in ax.texts:
        _, ty = t.get_position()
        assert y_min <= ty <= y_max, (
            f"text anchor y={ty} falls outside ylim=({lo}, {hi})"
        )


def _owned_meta_axes(xlim, ylim, figsize=(4, 4)):
    """Build a fresh publiplots-owned bar meta on a clean axes.

    Tight xlim/ylim disable autoscale; owner_is_publiplots=True forces
    _expand_axis to run regardless. This is the setup pattern from
    test_publiplots_owned_expands_limits_past_seaborn_default.
    """
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
    """At rotation=90 the label bbox spills onto the categorical axis too,
    so BOTH axes must grow past what rotation=0 would need.

    Uses oversized font + tight xlim so the rotated label provably spills
    past the initial categorical range; rotation=0 on the same setup should
    leave x untouched (bar centers sit well inside xlim and unrotated text
    bbox is narrow in y).
    """
    # Rotation=0 baseline: only y expands past 10; x stays put.
    _fig0, ax0 = _owned_meta_axes(xlim=(0.7, 1.1), ylim=(0, 10))
    _bar_values_strategy(
        ax0, fmt="{:.1f}", anchor="outside", offset=1.0,
        color="auto", pad=1.0, rotation=0.0, fontsize=80,
    )
    xlim_rot0 = ax0.get_xlim()
    ylim_rot0 = ax0.get_ylim()

    # Rotation=90: x should also expand (post-rotation bbox extends past xlim).
    _fig90, ax90 = _owned_meta_axes(xlim=(0.7, 1.1), ylim=(0, 10))
    _bar_values_strategy(
        ax90, fmt="{:.1f}", anchor="outside", offset=1.0,
        color="auto", pad=1.0, rotation=90.0, fontsize=80,
    )
    xlim_rot90 = ax90.get_xlim()
    ylim_rot90 = ax90.get_ylim()

    # y expands in both (labels always extend past bar top on outside anchor).
    assert ylim_rot0[1] > 10.0, f"rotation=0 ylim did not expand: {ylim_rot0}"
    assert ylim_rot90[1] > 10.0, f"rotation=90 ylim did not expand: {ylim_rot90}"
    # x should NOT expand under rotation=0 — unrotated label has narrow height.
    assert xlim_rot0 == pytest.approx((0.7, 1.1)), (
        f"rotation=0 xlim should not change, got {xlim_rot0}"
    )
    # x MUST expand under rotation=90 — the rotated label bbox spills
    # horizontally past the initial categorical range.
    assert xlim_rot90[0] < 0.7 or xlim_rot90[1] > 1.1, (
        f"x-axis did not expand under rotation=90: {xlim_rot90}"
    )


# -----------------------------------------------------------------------------
# Boxplot end-to-end
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


def test_boxplot_annotate_rotation_90():
    # Default anchor='right' → (ha='left', va='center').
    # Remap at 90° → (center, top).
    ax = pp.boxplot(data=_box_df(), x="g", y="y",
                    annotate={"rotation": 90})
    assert len(ax.texts) == 3
    for t in ax.texts:
        assert t.get_rotation() == pytest.approx(90.0)
        assert t.get_ha() == "center"
        assert t.get_va() == "top"


# -----------------------------------------------------------------------------
# Pointplot end-to-end
# -----------------------------------------------------------------------------

def _point_df():
    rng = np.random.default_rng(0)
    rows = []
    for t, base in zip(("t1", "t2", "t3"), (1.0, 2.5, 3.2)):
        for v in rng.normal(base, 0.3, 10):
            rows.append({"time": t, "v": float(v)})
    df = pd.DataFrame(rows)
    df["time"] = df["time"].astype("category")
    return df


def test_pointplot_annotate_rotation_90():
    # Default anchor='top' → (ha='center', va='bottom').
    # Remap at 90° → (left, center).
    ax = pp.pointplot(data=_point_df(), x="time", y="v",
                      errorbar="se", annotate={"rotation": 90})
    assert len(ax.texts) == 3
    for t in ax.texts:
        assert t.get_rotation() == pytest.approx(90.0)
        assert t.get_ha() == "left"
        assert t.get_va() == "center"
