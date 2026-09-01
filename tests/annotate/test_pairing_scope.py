"""Passes that must see only the artists of the call they describe.

Follow-ups from reviewing the #103/#199 fixes. Each test here pins a pass
that scanned the whole Axes and so paired records against artists it did
not own — the same root cause as both of those issues, in places the
original fixes did not scope.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

import publiplots as pp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- _introspect (foreign axes) must not label empty bins ----

def test_introspect_skips_empty_hist_bins():
    """``pp.annotate`` on a plain ``plt.hist`` must not label empty bins.

    An empty bin is a Rectangle with a full-width categorical extent and a
    zero value extent. Loosening `_is_bar_rect` to keep exactly-zero *bars*
    (#199) also let empty bins through on this path, which labels eight
    ``0.00``s across the gaps of a sparse histogram.
    """
    fig, ax = plt.subplots()
    ax.hist([0, 0, 0, 9], bins=10)
    pp.annotate(ax, kind="bar_values")

    assert [t.get_text() for t in ax.texts] == ["3.00", "1.00"]


def test_introspect_cannot_label_a_zero_bar_but_pp_barplot_can():
    """The documented limit of the foreign path, and why #199 lives elsewhere.

    On an axes publiplots did not draw, a zero-valued *bar* and an empty
    histogram *bin* are the same shape — a full categorical extent with a
    zero value extent — so `_introspect` cannot label one without labelling
    the other. It drops both, matching pre-#199 behaviour. `pp.barplot`
    knows it drew a bar chart, so the owned path labels the zero, which is
    where issue #199 was actually reported.
    """
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [3.0, 0.0, 5.0])
    pp.annotate(ax, kind="bar_values")
    assert [t.get_text() for t in ax.texts] == ["3.00", "5.00"]

    df = pd.DataFrame({"cat": list("ABC"), "val": [3.0, 0.0, 5.0]})
    fig, ax2 = pp.subplots()
    pp.barplot(data=df, x="cat", y="val", legend=False, ax=ax2,
               annotate={"fmt": ".2f"})
    assert [t.get_text() for t in ax2.texts] == ["3.00", "0.00", "5.00"]


def test_introspect_all_zero_horizontal_bars_draw_no_labels():
    """All-zero horizontal bars must not be labelled with their thickness.

    With every value zero, the width/height spreads tie, so `_infer_orient`
    fell back to ``"v"`` and read each bar's *thickness* (0.8) as its value.
    """
    fig, ax = plt.subplots()
    ax.barh([0, 1, 2], [0, 0, 0])
    pp.annotate(ax, kind="bar_values")

    assert [t.get_text() for t in ax.texts] == []


def test_introspect_all_zero_vertical_bars_draw_no_labels():
    """Same on the vertical axis."""
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [0, 0, 0])
    pp.annotate(ax, kind="bar_values")

    assert [t.get_text() for t in ax.texts] == []


def test_introspect_mixed_zero_horizontal_still_infers_h():
    """A zero among non-zero horizontal bars keeps the orientation right.

    The values reported are widths, not the bars' 0.8 thickness, which is
    what a wrong ``"v"`` inference would have produced.
    """
    fig, ax = plt.subplots()
    ax.barh([0, 1, 2], [3, 0, 5])
    pp.annotate(ax, kind="bar_values")

    assert [t.get_text() for t in ax.texts] == ["3.00", "5.00"]


def test_introspect_still_drops_seaborn_empty_hue_combos():
    """The premise behind `_is_bar_rect`, exercised where it actually holds.

    Raw ``sns.barplot`` emits a Rectangle with *both* extents zero for a
    ``(category, hue)`` combination with no observations. Those must stay
    filtered out; only the publiplots path avoids emitting them at all.
    """
    df = pd.DataFrame({
        "cat": ["A", "A", "B", "C"],
        "grp": ["x", "y", "x", "y"],
        "val": [1.0, 2.0, 3.0, 4.0],
    })
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="cat", y="val", hue="grp", ax=ax)

    degenerate = [p for p in ax.patches
                  if p.get_width() == 0 and p.get_height() == 0]
    assert len(degenerate) == 2, "premise broken: seaborn changed its empty-combo rect"

    pp.annotate(ax, kind="bar_values")
    assert [t.get_text() for t in ax.texts] == ["1.00", "3.00", "2.00", "4.00"]


# ---- errorbar matching must be scoped to the drawing call ----

def _two_call_df():
    rng = np.random.default_rng(0)
    rows = []
    for pipe, base in [("P1", 1.0), ("P2", 5.0)]:
        for cat in ("A", "B"):
            for v in rng.normal(base, 0.3, 8):
                rows.append({"pipe": pipe, "cat": cat, "v": float(v)})
    return pd.DataFrame(rows)


def test_second_barplot_call_anchors_labels_on_its_own_errorbars():
    """Two ``pp.barplot`` calls on one axes must not share errorbar anchors.

    ``_match_errorbars`` rescanned every line on the axes and took the first
    x-aligned segment, so the second call's labels were anchored on the
    *first* call's error bars — correct text, positioned next to a different
    bar entirely, with no warning because the rect/group counts still matched.
    """
    df = _two_call_df()
    fig, ax = pp.subplots()
    for pipe in ("P1", "P2"):
        pp.barplot(data=df[df["pipe"] == pipe], x="cat", y="v",
                   errorbar="se", legend=False, ax=ax, annotate={"fmt": ".2f"})

    heights = [p.get_height() for p in ax.patches]
    assert len(ax.texts) == 4
    # Each label must sit above the bar whose value it reports.
    for text, height in zip(ax.texts, heights):
        assert text.get_text() == f"{height:.2f}"
        assert text.get_position()[1] >= height, (
            f"label {text.get_text()!r} at y={text.get_position()[1]:.3f} "
            f"is below its own bar top {height:.3f}"
        )


def test_second_barplot_call_errorbar_extents_are_its_own():
    """The meta's err_low/err_high come from the call's own errorbars."""
    df = _two_call_df()
    fig, ax = pp.subplots()
    for pipe in ("P1", "P2"):
        pp.barplot(data=df[df["pipe"] == pipe], x="cat", y="v",
                   errorbar="se", legend=False, ax=ax)

    for bar in ax._publiplots_bar_meta.bars:
        assert bar.err_low is not None and bar.err_high is not None
        assert bar.err_low <= bar.value <= bar.err_high, (
            f"bar value {bar.value:.3f} outside its own errorbar "
            f"[{bar.err_low:.3f}, {bar.err_high:.3f}]"
        )


# ---- phantom groups must not enter the aggregation ----

def _mismatch_warnings(record):
    return [str(w.message) for w in record
            if "does not match the number of data groups" in str(w.message)]


def test_filtered_categorical_does_not_warn_or_mispair():
    """An unused Categorical level must not become a phantom group.

    ``BarSplitSpec.iter_draw_order``'s no-split branch yielded every declared
    category without checking that any row matched — unlike its hue and hatch
    branches. Filtering a Categorical column leaves the dropped level in
    ``.cat.categories``, so the aggregation invented a group with no bar and
    every later category paired one bar early.
    """
    raw = pd.DataFrame({
        "cat": pd.Categorical(["A", "B", "C"] * 3),
        "val": [1.0, 2.0, 3.0] * 3,
    })
    fig, ax = pp.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pp.barplot(data=raw[raw["cat"] != "B"], x="cat", y="val",
                   legend=False, ax=ax)
    assert not _mismatch_warnings(caught)

    meta = ax._publiplots_bar_meta
    assert [(b.category, b.value) for b in meta.bars] == [("A", 1.0), ("C", 3.0)]


def test_all_nan_group_does_not_warn_or_mispair():
    """A group whose values are all NaN draws no bar, so it is not a group.

    Seaborn drops NaN rows before drawing, so it emits two bars for three
    categories; the aggregation counted three and paired ``C``'s bar as ``B``.
    """
    df = pd.DataFrame({"cat": ["A", "B", "C"], "val": [1.0, np.nan, 3.0]})
    fig, ax = pp.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pp.barplot(data=df, x="cat", y="val", legend=False, ax=ax)
    assert not _mismatch_warnings(caught)

    meta = ax._publiplots_bar_meta
    assert [(b.category, b.value) for b in meta.bars] == [("A", 1.0), ("C", 3.0)]


def test_partially_nan_group_is_kept():
    """A group with some NaN and some real values is still a group."""
    df = pd.DataFrame({
        "cat": ["A", "B", "B", "C"],
        "val": [1.0, np.nan, 4.0, 3.0],
    })
    fig, ax = pp.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pp.barplot(data=df, x="cat", y="val", legend=False, ax=ax)
    assert not _mismatch_warnings(caught)

    meta = ax._publiplots_bar_meta
    assert [b.category for b in meta.bars] == ["A", "B", "C"]
    assert [b.value for b in meta.bars] == [1.0, 4.0, 3.0]


def test_frame_keyed_labels_follow_the_right_row_after_filtering():
    """The mispairing was visible through frame-keyed custom labels too."""
    raw = pd.DataFrame({
        "cat": pd.Categorical(["A", "B", "C"]),
        "val": [1.0, 2.0, 3.0],
        "tag": ["tA", "tB", "tC"],
    })
    fig, ax = pp.subplots()
    pp.barplot(data=raw[raw["cat"] != "B"], x="cat", y="val",
               legend=False, ax=ax,
               annotate={"kind": "bar_custom", "labels": "tag"})

    assert [t.get_text() for t in ax.texts] == ["tA", "tC"]
