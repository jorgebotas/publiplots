"""Tests for the BarValueMeta cache contract and axes introspection."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

from publiplots.annotate._cache import (
    BarRecord,
    BarValueMeta,
    BoxStatsMeta,
    BoxStatsRecord,
    PointRecord,
    PointValueMeta,
    _introspect,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_bar_value_meta_minimal_construct():
    meta = BarValueMeta(orient="v", bars=[], errorbar_kind=None,
                        hue_active=False, owner_is_publiplots=False)
    assert meta.orient == "v"
    assert meta.bars == []
    assert meta.hue_active is False
    assert meta.owner_is_publiplots is False


def test_bar_record_fields():
    fig, ax = plt.subplots()
    rect = ax.bar([0], [1.0])[0]
    rec = BarRecord(patch=rect, value=1.0, err_low=None, err_high=None, hue_color=None)
    assert rec.value == 1.0
    assert rec.err_low is None
    assert rec.err_high is None


def test_introspect_vertical_bars_no_errorbars():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0], width=0.8)
    meta = _introspect(ax)
    assert meta.orient == "v"
    assert len(meta.bars) == 3
    assert [b.value for b in meta.bars] == pytest.approx([1.0, 2.0, 3.0])
    assert all(b.err_low is None and b.err_high is None for b in meta.bars)
    assert meta.owner_is_publiplots is False
    assert meta.hue_active is False


def test_introspect_horizontal_bars_no_errorbars():
    fig, ax = plt.subplots()
    ax.barh([0, 1, 2], [1.0, 2.0, 3.0], height=0.8)
    meta = _introspect(ax)
    assert meta.orient == "h"
    assert [b.value for b in meta.bars] == pytest.approx([1.0, 2.0, 3.0])


def test_introspect_empty_axes():
    fig, ax = plt.subplots()
    meta = _introspect(ax)
    assert meta.bars == []


def test_introspect_vertical_bars_with_errorbars():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0], yerr=[0.1, 0.2, 0.3], width=0.8)
    meta = _introspect(ax)
    assert len(meta.bars) == 3
    # Matplotlib's errorbar plots vertical segments whose endpoints match bar ± yerr.
    for bar, exp_low, exp_high, val in zip(
        meta.bars, [0.9, 1.8, 2.7], [1.1, 2.2, 3.3], [1.0, 2.0, 3.0]
    ):
        # err_low/err_high are full-extent y values (including the bar tip direction).
        # For positive bars, err_high is the top-cap y, err_low is the bottom-cap y.
        assert bar.err_high == pytest.approx(exp_high, rel=1e-2)
        assert bar.err_low == pytest.approx(exp_low, rel=1e-2)


def test_introspect_ignores_nan_errorbar_segments():
    """Seaborn can emit NaN-bounded errorbar segments for single-sample groups
    (std with ddof=1 → NaN). Those must not leak into err_low/err_high."""
    fig, ax = plt.subplots()
    ax.bar([0, 1], [1.0, 2.0], yerr=[float("nan"), float("nan")], width=0.8)
    meta = _introspect(ax)
    assert len(meta.bars) == 2
    for bar in meta.bars:
        assert bar.err_low is None
        assert bar.err_high is None


def test_introspect_vertical_bars_with_capped_errorbars():
    """With capsize>0, matplotlib draws each errorbar as ONE Line2D whose data
    is nan-separated into [bottom cap | vertical stem | top cap]. The stem must
    still be recovered as err_low/err_high — regression for labels overlapping
    capped error bars."""
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0], yerr=[0.1, 0.2, 0.3], width=0.8, capsize=3)
    meta = _introspect(ax)
    assert len(meta.bars) == 3
    for bar, exp_low, exp_high in zip(
        meta.bars, [0.9, 1.8, 2.7], [1.1, 2.2, 3.3]
    ):
        assert bar.err_high == pytest.approx(exp_high, rel=1e-2)
        assert bar.err_low == pytest.approx(exp_low, rel=1e-2)


def test_introspect_horizontal_bars_with_capped_errorbars():
    """Same as above for orient='h': caps run along y, stem along x."""
    fig, ax = plt.subplots()
    ax.barh([0, 1, 2], [1.0, 2.0, 3.0], xerr=[0.1, 0.2, 0.3], height=0.8, capsize=3)
    meta = _introspect(ax)
    assert meta.orient == "h"
    assert len(meta.bars) == 3
    for bar, exp_low, exp_high in zip(
        meta.bars, [0.9, 1.8, 2.7], [1.1, 2.2, 3.3]
    ):
        assert bar.err_high == pytest.approx(exp_high, rel=1e-2)
        assert bar.err_low == pytest.approx(exp_low, rel=1e-2)


def test_bar_record_has_new_public_fields():
    rect = Rectangle((0, 0), 1, 1)
    rec = BarRecord(
        patch=rect,
        value=1.0,
        err_low=None, err_high=None,
        hue_color=None,
        # anchor_override is pre-existing; keep passing None for it
        anchor_override=None,
        category="A",
        hue_value=None,
        hatch_value=None,
        draw_index=0,
        frame_row_index=None,
    )
    assert rec.category == "A"
    assert rec.hue_value is None
    assert rec.hatch_value is None
    assert rec.draw_index == 0
    assert rec.frame_row_index is None
    assert rec.anchor_override is None


def test_bar_value_meta_has_source_frame_and_group_keys():
    df = pd.DataFrame({"x": ["A", "B"], "y": [1.0, 2.0]})
    meta = BarValueMeta(
        orient="v",
        bars=[],
        errorbar_kind=None,
        hue_active=False,
        owner_is_publiplots=True,
        source_frame=df,
        group_keys=("x",),
    )
    assert meta.source_frame is df
    assert meta.group_keys == ("x",)


def test_introspect_fills_draw_index_and_nones_other_fields():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0])
    fig.canvas.draw()
    meta = _introspect(ax)
    assert meta.source_frame is None
    assert meta.group_keys is None
    for i, bar in enumerate(meta.bars):
        assert bar.draw_index == i
        assert bar.category is None
        assert bar.hue_value is None
        assert bar.hatch_value is None
        assert bar.frame_row_index is None
        assert bar.anchor_override is None
    plt.close(fig)


def test_builder_populates_record_fields_and_meta_frame():
    import publiplots as pp
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
        "hue": pd.Categorical(["x", "y", "x", "y"], categories=["x", "y"]),
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    fig, ax = plt.subplots()
    pp.barplot(data=df, x="cat", y="value", hue="hue", ax=ax, annotate=True)

    meta = ax._publiplots_bar_meta
    assert meta.source_frame is df
    assert meta.group_keys == ("cat", "hue")
    # 4 bars in hue-outer / cat-inner draw order: (x,A), (x,B), (y,A), (y,B)
    expected = [
        ("A", "x"),
        ("B", "x"),
        ("A", "y"),
        ("B", "y"),
    ]
    for i, bar in enumerate(meta.bars):
        assert (bar.category, bar.hue_value) == expected[i]
        assert bar.hatch_value is None
        assert bar.draw_index == i
        row = df.iloc[bar.frame_row_index]
        assert row["cat"] == bar.category
        assert row["hue"] == bar.hue_value


def test_stacked_builder_populates_source_frame_and_keys():
    import publiplots as pp
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
        "hue": pd.Categorical(["x", "y", "x", "y"], categories=["x", "y"]),
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    fig, ax = plt.subplots()
    pp.barplot(data=df, x="cat", y="value", hue="hue",
               multiple="stack", ax=ax, annotate=True)
    meta = ax._publiplots_bar_meta
    assert meta.source_frame is df
    assert meta.group_keys == ("cat", "hue")
    for bar in meta.bars:
        assert bar.category in ("A", "B")
        assert bar.hue_value in ("x", "y")
        assert bar.frame_row_index is not None


def test_builder_frame_row_index_is_positional_not_label():
    """frame_row_index must be usable via iloc on the caller's frame,
    even when the caller's DataFrame has a non-RangeIndex."""
    import publiplots as pp
    df = pd.DataFrame(
        {
            "cat": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
            "value": [10.0, 20.0, 30.0, 40.0],
        },
        index=["r10", "r20", "r30", "r40"],  # non-RangeIndex
    )
    fig, ax = plt.subplots()
    pp.barplot(data=df, x="cat", y="value", ax=ax, annotate=True)
    meta = ax._publiplots_bar_meta
    assert meta.source_frame is df
    # For each bar, iloc on the caller's frame must land on a row whose
    # cat matches — i.e. the stored index is a position, not a label.
    for bar in meta.bars:
        row = df.iloc[bar.frame_row_index]
        assert row["cat"] == bar.category


def test_builder_requires_source_frame_kwarg():
    """Missing source_frame= is a TypeError, not a silent fallback."""
    import publiplots as pp
    from publiplots.annotate._builders import build_from_barplot_call
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "B"], categories=["A", "B"]),
        "value": [1.0, 2.0],
    })
    fig, ax = plt.subplots()
    pp.barplot(data=df, x="cat", y="value", ax=ax)
    with pytest.raises(TypeError):
        build_from_barplot_call(
            ax=ax, data=df, x="cat", y="value", hue=None,
            categorical_axis="cat", palette=None, errorbar=None,
            # source_frame deliberately omitted
        )


def test_bar_record_is_publicly_importable():
    """BarRecord is part of the public surface for kind='bar_custom' callables."""
    from publiplots.annotate import BarRecord as PublicBarRecord
    from publiplots.annotate._cache import BarRecord as PrivateBarRecord
    assert PublicBarRecord is PrivateBarRecord


def test_point_record_has_new_public_fields():
    rec = PointRecord(
        xy=(1.0, 2.0),
        value=2.0,
        err_low=None, err_high=None,
        hue_color=None,
        category="A",
        hue_value=None,
        hatch_value=None,
        draw_index=3,
        frame_row_index=7,
    )
    assert rec.category == "A"
    assert rec.hue_value is None
    assert rec.hatch_value is None
    assert rec.draw_index == 3
    assert rec.frame_row_index == 7


def test_point_value_meta_has_source_frame_and_group_keys():
    df = pd.DataFrame({"x": ["A"], "y": [1.0]})
    meta = PointValueMeta(
        orient="v",
        points=[],
        errorbar_kind=None,
        hue_active=False,
        owner_is_publiplots=True,
        source_frame=df,
        group_keys=("x",),
        group_dims=("cat",),
    )
    assert meta.source_frame is df
    assert meta.group_keys == ("x",)
    assert meta.group_dims == ("cat",)


def test_box_stats_record_has_new_public_fields():
    rec = BoxStatsRecord(
        center_pos=1.0,
        cat_half_width=0.4,
        stats={"median": 2.0},
        hue_color=None,
        category="A",
        hue_value=None,
        hatch_value=None,
        draw_index=3,
        frame_row_index=7,
    )
    assert rec.category == "A"
    assert rec.draw_index == 3
    assert rec.frame_row_index == 7


def test_box_stats_meta_has_source_frame_and_group_keys():
    df = pd.DataFrame({"x": ["A"], "y": [1.0]})
    meta = BoxStatsMeta(
        orient="v",
        boxes=[],
        hue_active=False,
        owner_is_publiplots=True,
        source_frame=df,
        group_keys=("x",),
        group_dims=("cat",),
    )
    assert meta.source_frame is df
    assert meta.group_keys == ("x",)
    assert meta.group_dims == ("cat",)
