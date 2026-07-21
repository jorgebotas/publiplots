"""Integration: pp.pointplot(..., annotate=...) cache-building + end-to-end."""
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


def test_pointplot_without_annotate_still_attaches_meta():
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B", "B"]),
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    ax = pp.pointplot(data=df, x="cat", y="value")
    meta = ax._publiplots_point_meta
    assert isinstance(meta, PointValueMeta)
    assert meta.source_frame is df
    assert meta.group_keys == ("cat",)
    assert meta.group_dims == ("cat",)
    for i, p in enumerate(meta.points):
        assert p.category in ("A", "B")
        assert p.draw_index == i
        assert p.frame_row_index is not None


def test_pointplot_annotate_kind_forwarded():
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B", "B"]),
        "value": [1.0, 2.0, 3.0, 4.0],
        "n":    [10, 11, 20, 21],
    })
    ax = pp.pointplot(
        data=df, x="cat", y="value",
        annotate={"kind": "point_custom", "labels": "n", "fmt": "n={}"},
    )
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["n=10", "n=20"]


# ---------------------------------------------------------------------------
# Estimator fix (issue #194 sibling): the label — and its position — must
# match the drawn marker for any estimator, not a recomputed group mean.
# ---------------------------------------------------------------------------

def _drawn_markers(ax):
    """Deduped drawn marker positions [(x, y), ...] in draw order.

    Skips errorbar lines (no marker) and collapses the double-layer marker
    copies that ``apply_double_layer_markers`` adds (same data, drawn thrice).
    NaN-aware so missing-group placeholder markers keep their grid slot.
    """
    def _key(arr):
        return tuple("nan" if math.isnan(v) else round(float(v), 6) for v in arr)

    seen = set()
    pts = []
    for ln in ax.lines:
        if ln.get_marker() in (None, "None", ""):
            continue
        xd = np.asarray(ln.get_xdata(), dtype=float)
        yd = np.asarray(ln.get_ydata(), dtype=float)
        if len(xd) == 0:
            continue
        key = (_key(xd), _key(yd))
        if key in seen:
            continue
        seen.add(key)
        for xx, yy in zip(xd, yd):
            pts.append((float(xx), float(yy)))
    return pts


def _skewed_groups(rng, cats=("A", "B", "C")):
    """Right-skewed groups where per-group median != mean."""
    rows = []
    for i, cat in enumerate(cats):
        for v in rng.exponential(1.0 + i, 200):
            rows.append({"g": cat, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = pd.Categorical(df["g"], categories=list(cats))
    return df


def test_pointplot_annotate_median_labels_the_median():
    """estimator='median': label matches the drawn marker (median), not mean."""
    df = _skewed_groups(np.random.default_rng(0))
    meds = {c: float(df[df.g == c].y.median()) for c in ("A", "B", "C")}
    means = {c: float(df[df.g == c].y.mean()) for c in ("A", "B", "C")}
    assert all(abs(meds[c] - means[c]) > 0.1 for c in meds), "median must != mean"

    ax = pp.pointplot(df, x="g", y="y", estimator="median", errorbar=None,
                      annotate={"fmt": ".3f"})
    # Each label's text must equal the drawn marker y at its x-position.
    drawn = {round(x, 3): y for x, y in _drawn_markers(ax)}
    assert len(ax.texts) == 3
    for t in ax.texts:
        tx, _ = t.get_position()
        assert float(t.get_text()) == pytest.approx(drawn[round(tx, 3)], abs=5e-4)
    # And the labeled values are the medians, not the means.
    labelled = sorted(float(t.get_text()) for t in ax.texts)
    assert labelled == pytest.approx(sorted(meds.values()), abs=5e-4)


def test_pointplot_annotate_median_hue_dodge_matches_markers():
    """estimator='median' + dodge: labels sit ON the dodged markers (x AND y).

    Covers both the estimator fix and the dodge x-position fix: the label
    must be centered on the drawn (dodged) marker, not at the integer
    category position.
    """
    rng = np.random.default_rng(1)
    rows = []
    for cat in ("A", "B", "C"):
        for cond in ("ctrl", "trt"):
            base = {"A": 1, "B": 3, "C": 5}[cat] + (0 if cond == "ctrl" else 2)
            for v in rng.exponential(base, 150):
                rows.append({"g": cat, "cond": cond, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = pd.Categorical(df["g"])
    df["cond"] = pd.Categorical(df["cond"])

    ax = pp.pointplot(df, x="g", y="y", hue="cond", estimator="median",
                      errorbar=None, dodge=True, annotate={"fmt": ".3f"})

    drawn = _drawn_markers(ax)
    assert len(ax.texts) == 6
    # At least one marker must be dodged off an integer position, else the
    # test wouldn't distinguish the bug.
    assert any(abs(x - round(x)) > 1e-3 for x, _ in drawn)
    # Every label must coincide with a drawn marker in BOTH x and y.
    for t in ax.texts:
        tx, _ = t.get_position()
        val = float(t.get_text())
        assert any(
            abs(tx - mx) < 1e-3 and abs(val - my) < 5e-4
            for mx, my in drawn
        ), f"label {val} at x={tx:.3f} matches no drawn marker"


def test_pointplot_annotate_callable_estimator_labels_drawn_marker():
    """A custom callable estimator (np.max) is honored by the label."""
    df = _skewed_groups(np.random.default_rng(2))
    ax = pp.pointplot(df, x="g", y="y", estimator=np.max, errorbar=None,
                      annotate={"fmt": ".3f"})
    drawn = {round(x, 3): y for x, y in _drawn_markers(ax)}
    maxes = {c: float(df[df.g == c].y.max()) for c in ("A", "B", "C")}
    for t in ax.texts:
        tx, _ = t.get_position()
        assert float(t.get_text()) == pytest.approx(drawn[round(tx, 3)], abs=5e-4)
    assert sorted(float(t.get_text()) for t in ax.texts) == pytest.approx(
        sorted(maxes.values()), abs=5e-4
    )


def test_pointplot_annotate_median_horizontal():
    """Horizontal pointplot (categorical y): label matches drawn marker x."""
    df = _skewed_groups(np.random.default_rng(3))
    meds = sorted(float(df[df.g == c].y.median()) for c in ("A", "B", "C"))
    ax = pp.pointplot(df, x="y", y="g", estimator="median", errorbar=None,
                      annotate={"fmt": ".3f"})
    # Value axis is x here; label text must equal the drawn marker x.
    drawn = {round(y, 3): x for x, y in _drawn_markers(ax)}
    assert len(ax.texts) == 3
    for t in ax.texts:
        _, ty = t.get_position()
        assert float(t.get_text()) == pytest.approx(drawn[round(ty, 3)], abs=5e-4)
    assert sorted(float(t.get_text()) for t in ax.texts) == pytest.approx(
        meds, abs=5e-4
    )


def test_pointplot_annotate_default_mean_still_correct():
    """Default (mean) estimator: label still matches the drawn marker."""
    df = _skewed_groups(np.random.default_rng(4))
    means = sorted(float(df[df.g == c].y.mean()) for c in ("A", "B", "C"))
    ax = pp.pointplot(df, x="g", y="y", errorbar=None, annotate={"fmt": ".3f"})
    drawn = {round(x, 3): y for x, y in _drawn_markers(ax)}
    for t in ax.texts:
        tx, _ = t.get_position()
        assert float(t.get_text()) == pytest.approx(drawn[round(tx, 3)], abs=5e-4)
    assert sorted(float(t.get_text()) for t in ax.texts) == pytest.approx(
        means, abs=5e-4
    )


def test_pointplot_annotate_custom_hue_order_bookkeeping_aligned():
    """A custom hue_order changes seaborn's draw order; the meta's hue_value
    and hue_color must stay aligned to the drawn marker, not the data's
    categorical order."""
    rng = np.random.default_rng(6)
    rows = []
    for cat in ("A", "B"):
        for cond, shift in (("lo", 0), ("hi", 10)):  # very different medians
            for v in rng.exponential(1 + shift, 150):
                rows.append({"g": cat, "cond": cond, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = pd.Categorical(df["g"])
    df["cond"] = pd.Categorical(df["cond"], categories=["lo", "hi"])

    # Draw order reversed relative to the categorical order.
    ax = pp.pointplot(df, x="g", y="y", hue="cond", hue_order=["hi", "lo"],
                      estimator="median", errorbar=None, dodge=True,
                      annotate={"fmt": ".2f"})
    meta = ax._publiplots_point_meta
    for p in meta.points:
        true_med = float(
            df[(df.g == p.category) & (df.cond == p.hue_value)].y.median()
        )
        # The point's own value (from the marker) must equal the median of
        # the group its bookkeeping claims it belongs to.
        assert p.value == pytest.approx(true_med, abs=5e-4), (
            f"{p.category}/{p.hue_value}: value {p.value} != median {true_med}"
        )


def test_pointplot_annotate_custom_order_bookkeeping_aligned():
    """A custom order= repositions categories; the meta's category key must
    stay aligned to the drawn marker, not the data's categorical order."""
    rng = np.random.default_rng(7)
    rows = []
    for cat, shift in (("A", 0), ("B", 5), ("C", 10)):  # very different medians
        for v in rng.exponential(1 + shift, 150):
            rows.append({"g": cat, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = pd.Categorical(df["g"], categories=["A", "B", "C"])

    ax = pp.pointplot(df, x="g", y="y", order=["C", "B", "A"],
                      estimator="median", errorbar=None, annotate={"fmt": ".2f"})
    meta = ax._publiplots_point_meta
    for p in meta.points:
        true_med = float(df[df.g == p.category].y.median())
        assert p.value == pytest.approx(true_med, abs=5e-4), (
            f"category {p.category}: value {p.value} != its median {true_med}"
        )


def test_pointplot_annotate_hue_equals_cat_axis_one_label_per_category():
    """hue == categorical axis: seaborn draws one NaN-padded marker series
    per category. Every category must still get exactly one correct label."""
    rng = np.random.default_rng(8)
    rows = []
    for cat, shift in (("A", 0), ("B", 3), ("C", 7)):
        for v in rng.exponential(1 + shift, 150):
            rows.append({"g": cat, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = pd.Categorical(df["g"], categories=["A", "B", "C"])
    meds = {c: float(df[df.g == c].y.median()) for c in ("A", "B", "C")}

    ax = pp.pointplot(df, x="g", y="y", hue="g", estimator="median",
                      errorbar=None, annotate={"fmt": ".3f"})
    assert len(ax.texts) == 3
    meta = ax._publiplots_point_meta
    for p in meta.points:
        assert p.value == pytest.approx(meds[p.category], abs=5e-4)
    assert sorted(float(t.get_text()) for t in ax.texts) == pytest.approx(
        sorted(meds.values()), abs=5e-4
    )


def test_pointplot_annotate_missing_group_no_spurious_label():
    """A (category, hue) combo with no rows draws a NaN marker; it must get
    no label, and the remaining labels must stay aligned to their markers."""
    rng = np.random.default_rng(5)
    rows = []
    for cat in ("A", "B", "C"):
        for cond in ("ctrl", "trt"):
            if cat == "C" and cond == "trt":
                continue  # missing group
            base = {"A": 1, "B": 3, "C": 5}[cat] + (0 if cond == "ctrl" else 2)
            for v in rng.exponential(base, 120):
                rows.append({"g": cat, "cond": cond, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = pd.Categorical(df["g"])
    df["cond"] = pd.Categorical(df["cond"])

    ax = pp.pointplot(df, x="g", y="y", hue="cond", estimator="median",
                      errorbar=None, dodge=True, annotate={"fmt": ".3f"})
    # 6 grid slots, 1 missing => 5 labels, none NaN.
    assert len(ax.texts) == 5
    for t in ax.texts:
        assert not math.isnan(float(t.get_text()))
    # Each label coincides with a finite drawn marker.
    finite = [(x, y) for x, y in _drawn_markers(ax) if not math.isnan(y)]
    for t in ax.texts:
        tx, _ = t.get_position()
        val = float(t.get_text())
        assert any(abs(tx - mx) < 1e-3 and abs(val - my) < 5e-4
                   for mx, my in finite)


# ---------------------------------------------------------------------------
# Draw-order alignment: numeric/bool hue and numeric category axes are drawn
# in SORTED order by seaborn (not first-occurrence). The builder must mirror
# that so category / hue_value / hue_color / frame_row_index bind to the
# right group. Regressions found by the review team.
# ---------------------------------------------------------------------------

def test_pointplot_annotate_numeric_hue_bookkeeping():
    """Numeric hue levels appear unsorted in the data but seaborn draws them
    sorted; the meta must attribute each drawn value to the right hue."""
    # hue values first appear as 3, 1 — seaborn sorts to [1, 3].
    df = pd.DataFrame({
        "g": pd.Categorical(list("aabb")),
        "h": [3, 1, 3, 1],
        "v": [10.0, 1.0, 30.0, 3.0],
    })
    truth = {("a", 3): 10.0, ("a", 1): 1.0, ("b", 3): 30.0, ("b", 1): 3.0}
    ax = pp.pointplot(df, x="g", y="v", hue="h", dodge=True, errorbar=None,
                      annotate={"fmt": ".1f"})
    meta = ax._publiplots_point_meta
    assert len(meta.points) == 4
    for p in meta.points:
        assert p.value == pytest.approx(truth[(p.category, p.hue_value)], abs=1e-6)


def test_pointplot_annotate_numeric_category_axis_bookkeeping():
    """A numeric categorical axis is drawn in sorted order; category keys
    must bind to the right position."""
    df = pd.DataFrame({
        "g": [3, 1, 3, 1, 2, 2],
        "v": [30.0, 10.0, 33.0, 11.0, 20.0, 22.0],
    })
    ax = pp.pointplot(df, x="g", y="v", estimator="median", errorbar=None,
                      annotate={"fmt": ".2f"})
    meta = ax._publiplots_point_meta
    for p in meta.points:
        med = float(df[df.g == p.category].v.median())
        assert p.value == pytest.approx(med, abs=1e-6)


def test_pointplot_annotate_numeric_hue_color_matches_marker():
    """hue_color must follow seaborn's sorted hue order, so the label color
    matches the drawn marker (not a first-occurrence-swapped palette)."""
    from matplotlib.colors import to_rgba
    df = pd.DataFrame({
        "g": pd.Categorical(list("aabb")),
        "h": [3, 1, 3, 1],
        "v": [10.0, 1.0, 30.0, 3.0],
    })
    pal = {1: "#1b9e77", 3: "#d95f02"}
    ax = pp.pointplot(df, x="g", y="v", hue="h", dodge=True, errorbar=None,
                      palette=pal, annotate={"fmt": ".1f"})
    for p in ax._publiplots_point_meta.points:
        assert p.hue_color == pytest.approx(to_rgba(pal[p.hue_value]))


def test_pointplot_annotate_large_dodge_no_misbinding():
    """dodge=1.0 puts markers on the category half-integer boundary; positional
    pairing (not coordinate rounding) must still bind every point correctly."""
    df = pd.DataFrame({
        "g": pd.Categorical(list("aabbcc")),
        "c": pd.Categorical(list("xyxyxy")),
        "v": [1.0, 5.0, 2.0, 6.0, 3.0, 7.0],
    })
    truth = {("a", "x"): 1.0, ("a", "y"): 5.0, ("b", "x"): 2.0,
             ("b", "y"): 6.0, ("c", "x"): 3.0, ("c", "y"): 7.0}
    ax = pp.pointplot(df, x="g", y="v", hue="c", dodge=1.0, errorbar=None,
                      annotate={"fmt": ".1f"})
    meta = ax._publiplots_point_meta
    assert len(meta.points) == 6
    for p in meta.points:
        assert p.value == pytest.approx(truth[(p.category, p.hue_value)], abs=1e-6)


def test_pointplot_annotate_markersize_zero_no_duplicate_labels():
    """markersize=0 makes the double-layer copies size-0 too; the builder must
    still recover one series per group (via the zorder guard), not triplicate."""
    rows = [{"g": c, "v": float(v)} for c in "abc" for v in (1.0, 2.0, 3.0)]
    df = pd.DataFrame(rows)
    df["g"] = pd.Categorical(df["g"])
    ax = pp.pointplot(df, x="g", y="v", markersize=0, errorbar=None,
                      annotate={"fmt": ".1f"})
    assert len(ax._publiplots_point_meta.points) == 3
    assert len(ax.texts) == 3


def test_pointplot_annotate_median_horizontal_hue_dodge():
    """Horizontal + hue + dodge + median: exercises the value-on-x branch with
    both fix axes at once. Labels match the drawn marker x."""
    rng = np.random.default_rng(11)
    rows = []
    for cat in ("A", "B", "C"):
        for cond in ("ctrl", "trt"):
            base = {"A": 1, "B": 3, "C": 5}[cat] + (0 if cond == "ctrl" else 2)
            for v in rng.exponential(base, 150):
                rows.append({"g": cat, "cond": cond, "y": float(v)})
    df = pd.DataFrame(rows)
    df["g"] = pd.Categorical(df["g"])
    df["cond"] = pd.Categorical(df["cond"])

    ax = pp.pointplot(df, x="y", y="g", hue="cond", estimator="median",
                      errorbar=None, dodge=True, annotate={"fmt": ".3f"})
    meta = ax._publiplots_point_meta
    assert len(meta.points) == 6
    for p in meta.points:
        med = float(df[(df.g == p.category) & (df.cond == p.hue_value)].y.median())
        assert p.value == pytest.approx(med, abs=5e-4)


def test_pointplot_annotate_two_call_form_median():
    """pp.annotate(ax, kind='point_values') after a median pointplot (no inline
    annotate=) must also label the drawn median."""
    df = _skewed_groups(np.random.default_rng(12))
    meds = sorted(float(df[df.g == c].y.median()) for c in ("A", "B", "C"))
    ax = pp.pointplot(df, x="g", y="y", estimator="median", errorbar=None)
    assert len(ax.texts) == 0  # nothing drawn yet
    pp.annotate(ax, kind="point_values", fmt=".3f")
    assert len(ax.texts) == 3
    assert sorted(float(t.get_text()) for t in ax.texts) == pytest.approx(
        meds, abs=5e-4
    )
