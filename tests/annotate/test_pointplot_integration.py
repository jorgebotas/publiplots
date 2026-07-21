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
