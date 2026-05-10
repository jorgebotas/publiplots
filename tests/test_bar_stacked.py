"""Tests for `pp.barplot(multiple="stack"|"fill")`.

Covers the stacked path that bypasses `sns.barplot` and draws with raw
`ax.bar` / `ax.barh` at integer category positions with cumulative
`bottom=` / `left=`. The dodge path is the default and is exercised by
the rest of the bar test suite — we only assert that passing
`multiple="dodge"` explicitly stays backwards-compatible here.
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


def _heights_by_cat(ax):
    """Group bar heights by (rounded) x-center → [heights...] in draw order."""
    groups: dict = {}
    for r in _bars(ax):
        key = round(r.get_x() + r.get_width() / 2.0, 3)
        groups.setdefault(key, []).append(r.get_height())
    return groups


def _simple_stack_df():
    """3 categories × 2 hue levels; per-group means 1,2,3,4,5,6."""
    rows = []
    for cat, (p_base, q_base) in zip(("A", "B", "C"), ((1.0, 4.0), (2.0, 5.0), (3.0, 6.0))):
        for _ in range(5):
            rows.append({"cat": cat, "grp": "P", "val": p_base})
            rows.append({"cat": cat, "grp": "Q", "val": q_base})
    df = pd.DataFrame(rows)
    df["cat"] = pd.Categorical(df["cat"], categories=["A", "B", "C"])
    df["grp"] = pd.Categorical(df["grp"], categories=["P", "Q"])
    return df


def _tri_stack_df():
    rows = []
    for cat, (a, b, c) in zip(("A", "B", "C"),
                              ((1.0, 2.0, 3.0), (2.0, 2.0, 2.0), (3.0, 1.0, 1.0))):
        for _ in range(4):
            rows.append({"cat": cat, "grp": "x", "val": a})
            rows.append({"cat": cat, "grp": "y", "val": b})
            rows.append({"cat": cat, "grp": "z", "val": c})
    df = pd.DataFrame(rows)
    df["cat"] = pd.Categorical(df["cat"], categories=["A", "B", "C"])
    df["grp"] = pd.Categorical(df["grp"], categories=["x", "y", "z"])
    return df


# -----------------------------------------------------------------------------
# Contract
# -----------------------------------------------------------------------------


def test_dodge_is_default_and_unchanged():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp")
    # Dodge puts bars at fractional x positions (not integer-only).
    xs = sorted({round(r.get_x() + r.get_width() / 2.0, 3) for r in _bars(ax)})
    assert len(xs) == 6  # 3 cats × 2 hue levels, side-by-side


def test_multiple_stack_returns_axes():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None)
    assert isinstance(ax, Axes)


def test_multiple_rejects_figsize():
    df = _simple_stack_df()
    with pytest.raises(TypeError):
        pp.barplot(data=df, x="cat", y="val", hue="grp",
                   multiple="stack", errorbar=None, figsize=(4, 3))


def test_invalid_multiple_raises():
    df = _simple_stack_df()
    with pytest.raises(ValueError, match="multiple"):
        pp.barplot(data=df, x="cat", y="val", hue="grp", multiple="nope")


# -----------------------------------------------------------------------------
# Stack with hue
# -----------------------------------------------------------------------------


def test_stack_hue_produces_one_rect_per_cat_x_level():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None)
    assert len(_bars(ax)) == 3 * 2


def test_stack_hue_cumulative_bottoms():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None)
    # For each category: first segment sits on 0, second on the first's height.
    for cat_pos in (0, 1, 2):
        segs = sorted(
            (r for r in _bars(ax) if abs(r.get_x() + r.get_width() / 2 - cat_pos) < 0.1),
            key=lambda r: r.get_y(),
        )
        assert len(segs) == 2
        assert segs[0].get_y() == pytest.approx(0.0)
        assert segs[1].get_y() == pytest.approx(segs[0].get_height())


def test_stack_hue_totals_match_sum_of_means():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None)
    # A: P=1, Q=4 → 5; B: 2+5=7; C: 3+6=9
    expected = {0: 5.0, 1: 7.0, 2: 9.0}
    for pos, tot in expected.items():
        segs = [r for r in _bars(ax) if abs(r.get_x() + r.get_width() / 2 - pos) < 0.1]
        assert sum(r.get_height() for r in segs) == pytest.approx(tot)


def test_stack_hue_order_reorders_stack_bottom_to_top():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None, hue_order=["Q", "P"])
    # With hue_order=[Q, P], bottom segment (y==0) should be Q — height matches Q's mean.
    # Q means: A=4, B=5, C=6.
    expected_bottom = {0: 4.0, 1: 5.0, 2: 6.0}
    for pos, h in expected_bottom.items():
        bottom = next(
            r for r in _bars(ax)
            if abs(r.get_x() + r.get_width() / 2 - pos) < 0.1
            and abs(r.get_y()) < 1e-9
        )
        assert bottom.get_height() == pytest.approx(h)


def test_stack_hue_palette_dict_drives_face_colors():
    df = _simple_stack_df()
    palette = {"P": "#ff0000", "Q": "#00ff00"}
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None, palette=palette)
    rgbs = {tuple(r.get_facecolor()[:3]) for r in _bars(ax)}
    assert to_rgba("#ff0000")[:3] in rgbs
    assert to_rgba("#00ff00")[:3] in rgbs


def test_stack_palette_str_name_works():
    df = _simple_stack_df()
    # Just assert the plot succeeds and produces 2 distinct face colors.
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None, palette="pastel")
    rgbs = {tuple(r.get_facecolor()[:3]) for r in _bars(ax)}
    assert len(rgbs) == 2


# -----------------------------------------------------------------------------
# Stack with hatch (hue=None)
# -----------------------------------------------------------------------------


def test_stack_hatch_only_paints_per_level_hatches():
    df = _simple_stack_df()
    hatch_map = {"P": "", "Q": "///"}
    ax = pp.barplot(data=df, x="cat", y="val", hatch="grp",
                    multiple="stack", errorbar=None, hatch_map=hatch_map)
    assert len(_bars(ax)) == 3 * 2
    hatches = {r.get_hatch() or "" for r in _bars(ax)}
    assert "" in hatches
    assert "///" in hatches


# -----------------------------------------------------------------------------
# Fill mode
# -----------------------------------------------------------------------------


def test_fill_totals_are_one():
    df = _tri_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="fill", errorbar=None)
    for pos in (0, 1, 2):
        segs = [r for r in _bars(ax) if abs(r.get_x() + r.get_width() / 2 - pos) < 0.1]
        assert sum(r.get_height() for r in segs) == pytest.approx(1.0, abs=1e-9)


# -----------------------------------------------------------------------------
# Orientation
# -----------------------------------------------------------------------------


def test_horizontal_stack_widths_and_inverted_y():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="val", y="cat", hue="grp",
                    multiple="stack", errorbar=None)
    assert len(_bars(ax)) == 3 * 2
    # Horizontal: y-axis inverted so category 0 is at the top.
    assert ax.get_ylim()[0] > ax.get_ylim()[1]
    # Widths per category sum to expected totals; lefts are cumulative.
    for cat_pos in (0, 1, 2):
        segs = sorted(
            (r for r in _bars(ax)
             if abs(r.get_y() + r.get_height() / 2 - cat_pos) < 0.1),
            key=lambda r: r.get_x(),
        )
        assert len(segs) == 2
        assert segs[0].get_x() == pytest.approx(0.0)
        assert segs[1].get_x() == pytest.approx(segs[0].get_width())


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


def test_stack_with_hue_and_distinct_hatch_raises():
    # hue + hatch as two different non-cat columns.
    rows = []
    for cat in ("A", "B"):
        for grp in ("P", "Q"):
            for trt in ("ctrl", "trt"):
                rows.append({"cat": cat, "grp": grp, "trt": trt, "val": 1.0})
    df = pd.DataFrame(rows)
    df["cat"] = pd.Categorical(df["cat"])
    df["grp"] = pd.Categorical(df["grp"])
    df["trt"] = pd.Categorical(df["trt"])
    with pytest.raises(NotImplementedError, match="stack"):
        pp.barplot(data=df, x="cat", y="val", hue="grp", hatch="trt",
                   multiple="stack", errorbar=None)


def test_stack_without_hue_or_hatch_raises():
    df = _simple_stack_df()
    with pytest.raises(ValueError, match="stack"):
        pp.barplot(data=df, x="cat", y="val",
                   multiple="stack", errorbar=None)


def test_stack_with_errorbar_warns_and_drops():
    df = _simple_stack_df()
    with pytest.warns(UserWarning, match="errorbars"):
        ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                        multiple="stack", errorbar="se")
    # No errorbar artists — only the Rectangles live in ax.patches; errorbar
    # lines would appear as Line2D in ax.lines or LineCollection in
    # ax.collections. Stack path draws with raw ax.bar, which returns a
    # BarContainer (no errorbars).
    from matplotlib.collections import LineCollection
    err_lines = [l for l in ax.lines if len(l.get_xdata()) >= 2]
    err_collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert not err_lines
    assert not err_collections


# -----------------------------------------------------------------------------
# Annotate
# -----------------------------------------------------------------------------


def test_stack_annotate_draws_per_segment_labels():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None, annotate=True)
    # One label per drawn segment.
    assert len(ax.texts) == len(_bars(ax)) == 3 * 2


def test_stack_annotate_label_values_match_segment_heights():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None,
                    annotate={"fmt": ".1f"})
    assert len(ax.texts) == 6
    # Every formatted label corresponds to some segment's height.
    heights = {f"{r.get_height():.1f}" for r in _bars(ax)}
    for t in ax.texts:
        assert t.get_text() in heights


def test_stack_annotate_default_anchor_is_inside():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None, annotate=True)
    ax.figure.canvas.draw()
    # "inside" places the label center inside the segment in data coords.
    for t in ax.texts:
        _, y = t.get_position()
        found = False
        for r in _bars(ax):
            if r.get_y() - 1e-6 <= y <= r.get_y() + r.get_height() + 1e-6:
                found = True
                break
        assert found, f"label y={y!r} not inside any segment"


def test_stack_annotate_outside_override_works():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None,
                    annotate={"anchor": "outside"})
    # "outside" labels sit on segment boundaries (top edge of each segment).
    ax.figure.canvas.draw()
    tops = {round(r.get_y() + r.get_height(), 3) for r in _bars(ax)}
    for t in ax.texts:
        _, y = t.get_position()
        assert round(y, 3) in tops


def test_fill_annotate_labels_are_fractions():
    df = _tri_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="fill", errorbar=None,
                    annotate={"fmt": ".3f"})
    for t in ax.texts:
        v = float(t.get_text())
        assert 0.0 <= v <= 1.0


# -----------------------------------------------------------------------------
# Legend stash
# -----------------------------------------------------------------------------


def test_stack_hue_stashes_single_hue_entry():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None)
    entries = get_entries(ax)
    kinds = [(e.name, e.kind) for e in entries]
    assert ("grp", "hue") in kinds
    hue_entry = next(e for e in entries if e.kind == "hue")
    assert list(hue_entry.labels) == ["P", "Q"]


def test_stack_hue_order_propagates_to_legend_entry():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None, hue_order=["Q", "P"])
    entries = get_entries(ax)
    hue_entry = next(e for e in entries if e.kind == "hue")
    assert list(hue_entry.labels) == ["Q", "P"]


def test_stack_hatch_only_stashes_hatch_entry():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hatch="grp",
                    multiple="stack", errorbar=None,
                    hatch_map={"P": "", "Q": "///"})
    kinds = [(e.name, e.kind) for e in get_entries(ax)]
    assert ("grp", "hatch") in kinds
    assert not any(k == "hue" for _, k in kinds)


def test_stack_legend_false_does_not_stash():
    df = _simple_stack_df()
    ax = pp.barplot(data=df, x="cat", y="val", hue="grp",
                    multiple="stack", errorbar=None, legend=False)
    assert get_entries(ax) == []
