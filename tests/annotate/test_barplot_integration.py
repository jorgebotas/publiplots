"""Integration: pp.barplot(..., annotate=...) cache-building + end-to-end."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate._cache import BarValueMeta


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _simple_df():
    return pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
    })


def _grouped_df():
    rng = np.random.default_rng(0)
    rows = []
    for group in ("A", "B"):
        for cond in ("ctrl", "trt"):
            for v in rng.normal(loc=(1 if group == "A" else 2) + (0 if cond == "ctrl" else 1),
                                scale=0.1, size=8):
                rows.append({"group": group, "cond": cond, "y": float(v)})
    df = pd.DataFrame(rows)
    df["group"] = df["group"].astype("category")
    df["cond"] = df["cond"].astype("category")
    return df


def test_barplot_annotate_false_attaches_meta_but_draws_no_labels():
    """Meta is attached unconditionally; labels only when annotate= truthy."""
    ax = pp.barplot(data=_simple_df(), x="category", y="value")
    assert isinstance(ax._publiplots_bar_meta, BarValueMeta)
    assert len(ax.texts) == 0


def test_barplot_without_annotate_still_attaches_meta():
    """Follow-up pp.annotate calls need the cache even when annotate= is False."""
    ax = pp.barplot(data=_simple_df(), x="category", y="value")
    assert isinstance(ax._publiplots_bar_meta, BarValueMeta)
    assert ax._publiplots_bar_meta.owner_is_publiplots is True
    # No labels drawn because annotate= was not passed
    assert len(ax.texts) == 0


def test_stacked_barplot_without_annotate_attaches_meta():
    """Cache must also be attached on the stacked/fill/gain path."""
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
        "hue": pd.Categorical(["x", "y", "x", "y"], categories=["x", "y"]),
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    ax = pp.barplot(data=df, x="cat", y="value", hue="hue", multiple="stack")
    assert isinstance(ax._publiplots_bar_meta, BarValueMeta)
    assert len(ax.texts) == 0


def test_barplot_annotate_true_attaches_meta_and_draws_labels():
    ax = pp.barplot(data=_simple_df(), x="category", y="value", annotate=True)
    assert isinstance(ax._publiplots_bar_meta, BarValueMeta)
    assert ax._publiplots_bar_meta.owner_is_publiplots is True
    texts = [t for t in ax.texts]
    assert len(texts) == 3


def test_barplot_annotate_dict_forwarded():
    ax = pp.barplot(data=_simple_df(), x="category", y="value",
                         annotate={"fmt": ".3f"})
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["1.000", "2.000", "3.000"]


def test_barplot_annotate_with_hue_has_hue_active():
    ax = pp.barplot(data=_grouped_df(), x="group", y="y", hue="cond",
                         annotate=True)
    meta = ax._publiplots_bar_meta
    assert meta.hue_active is True
    assert all(b.hue_color is not None for b in meta.bars)


def test_barplot_annotate_no_hue_hue_active_false():
    ax = pp.barplot(data=_simple_df(), x="category", y="value", annotate=True)
    assert ax._publiplots_bar_meta.hue_active is False


def test_barplot_annotate_expands_ylim():
    df = pd.DataFrame({
        "category": pd.Categorical(["A", "B"]),
        "value": [10.0, 10.0],
    })
    ax = pp.barplot(data=df, x="category", y="value", annotate=True)
    top = ax.get_ylim()[1]
    # seaborn's default would stop at ~10 + a small pad; with annotate the
    # label must fit above that, so top should be noticeably above 10.
    assert top > 10.0


def test_barplot_annotate_with_errorbars_anchors_past_cap():
    """With errorbar='se' and annotate=True, label y should sit above the cap."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "group": pd.Categorical(np.repeat(["A", "B"], 20)),
        "y": np.concatenate([
            rng.normal(1.0, 0.5, 20),
            rng.normal(2.0, 0.5, 20),
        ]),
    })
    ax = pp.barplot(data=df, x="group", y="y", errorbar="se", annotate=True)
    meta = ax._publiplots_bar_meta
    # Each bar has an err_high > value (standard error extends above mean).
    for bar in meta.bars:
        assert bar.err_high is not None
        assert bar.err_high > bar.value
    # Corresponding label y coordinates should be at or above err_high.
    for bar, text in zip(meta.bars, ax.texts):
        _, y = text.get_position()
        assert y >= bar.err_high


def test_barplot_annotate_with_capped_errorbars_anchors_past_cap():
    """capsize>0 path: the existing anchors_past_cap test uses the default
    capsize=0 (a clean 2-point errorbar Line2D). With capsize>0 matplotlib
    emits a single nan-separated Line2D; the cap extents must still be found
    so labels clear the cap rather than overlapping it."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "group": pd.Categorical(np.repeat(["A", "B"], 20)),
        "y": np.concatenate([
            rng.normal(1.0, 0.5, 20),
            rng.normal(2.0, 0.5, 20),
        ]),
    })
    ax = pp.barplot(data=df, x="group", y="y", errorbar="se",
                    capsize=3, annotate=True)
    meta = ax._publiplots_bar_meta
    for bar in meta.bars:
        assert bar.err_high is not None
        assert bar.err_high > bar.value
    for bar, text in zip(meta.bars, ax.texts):
        _, y = text.get_position()
        assert y >= bar.err_high


def test_pointplot_annotate_with_capped_errorbars_anchors_past_cap():
    """Same capsize>0 regression for pointplot, which shares the errorbar
    segment matcher."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "group": pd.Categorical(np.repeat(["A", "B"], 20)),
        "y": np.concatenate([
            rng.normal(1.0, 0.5, 20),
            rng.normal(2.0, 0.5, 20),
        ]),
    })
    ax = pp.pointplot(data=df, x="group", y="y", errorbar="se",
                      capsize=3, annotate=True)
    meta = ax._publiplots_point_meta
    for point in meta.points:
        assert point.err_high is not None
        assert point.err_high > point.value


def test_barplot_annotate_hue_labels_pair_to_correct_bars():
    """Labels must match bar heights, not swap across dodge groups.

    With asymmetric heights, a wrong loop order swaps labels silently.
    """
    import numpy as np
    df = pd.DataFrame({
        "g": pd.Categorical(["A", "B", "A", "B"]),
        "c": pd.Categorical(["ctrl", "ctrl", "trt", "trt"]),
        "y": [1.0, 2.0, 10.0, 20.0],
    })
    ax = pp.barplot(data=df, x="g", y="y", hue="c",
                         errorbar=None, annotate={"fmt": ".1f"})
    # Sort texts left-to-right by x position; seaborn draws hue-outer, cat-inner
    # so order is: ctrl-A=1, ctrl-B=2, trt-A=10, trt-B=20.
    by_x = sorted(ax.texts, key=lambda t: t.get_position()[0])
    labels = [t.get_text() for t in by_x]
    assert labels == ["1.0", "10.0", "2.0", "20.0"], (
        f"Got {labels}; bar-label pairing is broken."
    )


def test_barplot_annotate_errorbar_ci_pulls_from_drawn_artists():
    """For errorbar='ci' (bootstrap percentile), err_high should come
    from the drawn cap, not from a re-aggregated normal approx."""
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "g": pd.Categorical(np.repeat(["A", "B"], 50)),
        "y": np.concatenate([
            rng.normal(1.0, 0.5, 50),
            rng.normal(2.0, 0.5, 50),
        ]),
    })
    ax = pp.barplot(data=df, x="g", y="y", errorbar="ci", annotate=True)
    meta = ax._publiplots_bar_meta
    for bar in meta.bars:
        assert bar.err_high is not None
        # err_high must come from the drawn cap (not silent 0.0 fallback nor wrong math)
        assert bar.err_high > bar.value


def test_barplot_annotate_errorbar_none_all_none():
    df = pd.DataFrame({
        "g": pd.Categorical(["A", "B"]),
        "y": [1.0, 2.0],
    })
    ax = pp.barplot(data=df, x="g", y="y", errorbar=None, annotate=True)
    meta = ax._publiplots_bar_meta
    for bar in meta.bars:
        assert bar.err_low is None
        assert bar.err_high is None


def test_barplot_annotate_hue_and_hatch_pair_correctly():
    """Double-split (hue + hatch, distinct columns) produces n_cat × n_hue
    × n_hatch bars; every bar must get its own correctly-paired label.

    Regression: the original builder iterated hue × cat only, so double-split
    bars lost the hatch dimension and silently mispaired labels.
    """
    import math
    rng = np.random.default_rng(789)
    rows = []
    heights = {
        ("A", "Vehicle", "24h"): 95, ("A", "Vehicle", "48h"): 93,
        ("A", "Drug", "24h"): 75,    ("A", "Drug", "48h"): 60,
        ("B", "Vehicle", "24h"): 94, ("B", "Vehicle", "48h"): 92,
        ("B", "Drug", "24h"): 80,    ("B", "Drug", "48h"): 70,
    }
    for (cell, treat, time), mean in heights.items():
        for v in rng.normal(mean, 0.1, 10):  # tight std so labels are predictable
            rows.append({"cell": cell, "treat": treat, "time": time,
                         "y": float(v)})
    df = pd.DataFrame(rows)
    for c in ("cell", "treat", "time"):
        df[c] = df[c].astype("category")

    ax = pp.barplot(
        data=df, x="cell", y="y",
        hue="treat", hatch="time", errorbar="se",
        hatch_map={"24h": "", "48h": "///"},
        annotate={"fmt": ".0f"},
    )
    rects = [p for p in ax.patches if p.get_width() > 0 and p.get_height() > 0]
    assert len(rects) == 8, f"expected 8 bars, got {len(rects)}"
    assert len(ax.texts) == 8, f"expected 8 labels, got {len(ax.texts)}"
    for rect, text in zip(rects, ax.texts):
        expected = round(rect.get_height())
        assert int(text.get_text()) == expected, (
            f"label {text.get_text()!r} doesn't match bar height {rect.get_height():.2f}"
        )


def test_barplot_annotate_hue_equals_cat_axis_pairs_correctly():
    """hue == categorical axis: seaborn doesn't dodge; we emit one label
    per category (not n_cat × n_cat)."""
    df = pd.DataFrame({
        "c": pd.Categorical(["A", "B", "C"]),
        "v": [1.0, 2.0, 3.0],
    })
    ax = pp.barplot(data=df, x="c", y="v", hue="c",
                         errorbar=None, annotate=True)
    rects = [p for p in ax.patches if p.get_width() > 0 and p.get_height() > 0]
    assert len(rects) == 3
    assert len(ax.texts) == 3
    for rect, text in zip(rects, ax.texts):
        assert float(text.get_text()) == pytest.approx(rect.get_height())


def test_barplot_annotate_hatch_equals_cat_axis_pairs_correctly():
    """hatch == categorical axis, no hue: one label per category."""
    rng = np.random.default_rng(42)
    rows = []
    for g, mean in zip(("A", "B", "C"), (1.0, 2.0, 3.0)):
        for v in rng.normal(mean, 0.1, 10):
            rows.append({"c": g, "v": float(v)})
    df = pd.DataFrame(rows)
    df["c"] = df["c"].astype("category")
    ax = pp.barplot(
        data=df, x="c", y="v", hatch="c",
        hatch_map={"A": "", "B": "///", "C": "xx"},
        errorbar=None, annotate=True,
    )
    rects = [p for p in ax.patches if p.get_width() > 0 and p.get_height() > 0]
    assert len(rects) == 3
    assert len(ax.texts) == 3
    for rect, text in zip(rects, ax.texts):
        assert float(text.get_text()) == pytest.approx(rect.get_height(), abs=0.02)


def test_barplot_annotate_hue_equals_cat_plus_hatch_pairs_correctly():
    """hue == cat but a separate hatch is present: hatch causes dodging,
    so we have n_cat × n_hatch bars ordered hatch-outer / cat-inner."""
    rng = np.random.default_rng(42)
    rows = []
    for g in ("A", "B", "C"):
        for t in ("t1", "t2"):
            mean = {"A": 1.0, "B": 2.0, "C": 3.0}[g] + (0.0 if t == "t1" else 0.5)
            for v in rng.normal(mean, 0.1, 10):
                rows.append({"c": g, "t": t, "v": float(v)})
    df = pd.DataFrame(rows)
    df["c"] = df["c"].astype("category")
    df["t"] = df["t"].astype("category")
    ax = pp.barplot(
        data=df, x="c", y="v", hue="c", hatch="t",
        hatch_map={"t1": "", "t2": "///"},
        errorbar=None, annotate=True,
    )
    rects = [p for p in ax.patches if p.get_width() > 0 and p.get_height() > 0]
    assert len(rects) == 6
    assert len(ax.texts) == 6
    for rect, text in zip(rects, ax.texts):
        assert float(text.get_text()) == pytest.approx(rect.get_height(), abs=0.02)


def test_barplot_annotate_color_hue_without_hue_warns():
    """annotate={'color': 'hue'} on a no-hue barplot warns and falls back."""
    df = pd.DataFrame({
        "category": pd.Categorical(["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
    })
    with pytest.warns(UserWarning, match="plot has no hue"):
        ax = pp.barplot(data=df, x="category", y="value",
                             annotate={"color": "hue"})


def test_barplot_annotate_single_sample_groups_no_nan_positions():
    """Single-sample groups yield NaN SE; label positions must stay finite.

    Regression: seaborn emits errorbar segments with NaN endpoints for
    single-sample groups (std(ddof=1) is NaN when n=1). Those segments must be
    filtered in _match_errorbars, not propagated into err_high/err_low.
    """
    df = pd.DataFrame({
        "c": pd.Categorical(["A", "B", "C", "D", "E"]),
        "v": [0.2, 8.4, 0.5, 12.0, 3.1],
    })
    ax = pp.barplot(data=df, x="v", y="c",
                         annotate={"anchor": "inside", "fmt": ".1f"})
    for t in ax.texts:
        x, y = t.get_position()
        assert not math.isnan(float(x)), f"NaN x for label {t.get_text()!r}"
        assert not math.isnan(float(y)), f"NaN y for label {t.get_text()!r}"


def test_barplot_annotate_bar_custom_inline():
    """pp.barplot(data, x, y, annotate={'kind': 'bar_custom', 'labels': 'n'})."""
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
        "n": [12, 34, 56],
    })
    ax = pp.barplot(
        data=df, x="cat", y="value",
        annotate={"kind": "bar_custom", "labels": "n", "fmt": "n={}"},
    )
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["n=12", "n=34", "n=56"]
