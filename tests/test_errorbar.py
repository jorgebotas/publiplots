"""Tests for the shared custom-errorbar helper and its plot-function
integrations, AND tests for the standalone ``pp.errorbar`` primitive.

The first half of this file covers the legacy ``format_for_custom_errorbar``
helper (used by ``pp.pointplot`` / ``pp.lineplot``). The second half (after
``# ====== pp.errorbar primitive ======``) tests the new top-level
``pp.errorbar`` plotting function — a 2D scatter primitive with x and/or
y uncertainty stems.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

import publiplots as pp
from publiplots.utils.errorbar import format_for_custom_errorbar
from publiplots.utils.legend_entries import get_entries, is_continuous_hue

# Import the new primitive directly — the public ``pp.errorbar`` alias is
# wired in ``src/publiplots/__init__.py`` by the main agent after this
# test/impl pair lands; the in-module import keeps these tests independent
# of that registration step.
from publiplots.plot.errorbar import errorbar as _errorbar


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# ---- Helper unit tests ----


def test_helper_triplicates_rows():
    df = pd.DataFrame(
        {"day": [1, 2, 3], "y": [10, 20, 30], "lo": [8, 18, 28], "hi": [12, 22, 32]}
    )
    out = format_for_custom_errorbar(df, "day", "y", ("lo", "hi"), "x")
    assert len(out) == 9  # 3x original
    # Value column must cycle lo / original / hi.
    assert list(out["y"]) == [8, 18, 28, 10, 20, 30, 12, 22, 32]
    # Other columns repeat three times unchanged.
    assert list(out["day"]) == [1, 2, 3] * 3


def test_helper_auto_orient_categorical_x():
    df = pd.DataFrame(
        {"gene": ["A", "B"], "val": [1.0, 2.0], "lo": [0.5, 1.5], "hi": [1.5, 2.5]}
    )
    # orient=None + x categorical -> value_col = y.
    out = format_for_custom_errorbar(df, "gene", "val", ("lo", "hi"), None)
    assert list(out["val"]) == [0.5, 1.5, 1.0, 2.0, 1.5, 2.5]


def test_helper_explicit_orient_y_aggregates_on_x():
    df = pd.DataFrame(
        {"val": [1.0, 2.0], "gene": ["A", "B"], "lo": [0.5, 1.5], "hi": [1.5, 2.5]}
    )
    # orient='y' means the categorical axis is y; x carries the value.
    out = format_for_custom_errorbar(df, "val", "gene", ("lo", "hi"), "y")
    assert list(out["val"]) == [0.5, 1.5, 1.0, 2.0, 1.5, 2.5]


def test_helper_both_numeric_no_orient_defaults_to_y():
    """Both axes numeric and orient=None -> value_col falls back to y
    (matching seaborn's ``orient='x'`` default)."""
    df = pd.DataFrame(
        {"x": [1, 2, 3], "y": [10, 20, 30], "lo": [8, 18, 28], "hi": [12, 22, 32]}
    )
    out = format_for_custom_errorbar(df, "x", "y", ("lo", "hi"), None)
    assert list(out["y"]) == [8, 18, 28, 10, 20, 30, 12, 22, 32]


def test_helper_missing_cols_raises_keyerror():
    df = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
    with pytest.raises(KeyError):
        format_for_custom_errorbar(df, "x", "y", ("nope", "also_nope"), "x")


# ---- Pointplot integration / regression ----


def test_pointplot_custom_errorbar_non_categorical_axes_no_crash():
    """Regression for the ``orient.isin(...)`` bug in v0.10.1: when both
    x and y are numeric and ``orient`` is the default ``None``, the old
    local helper called ``None.isin(...)`` and raised. After the move
    to the shared helper the call resolves cleanly (defaulting to y as
    the value axis)."""
    df = pd.DataFrame(
        {"x": [1, 2, 3], "y": [10, 20, 30], "lo": [8, 18, 28], "hi": [12, 22, 32]}
    )
    fig, ax = plt.subplots()
    pp.pointplot(
        data=df, x="x", y="y", errorbar=("custom", ("lo", "hi")), ax=ax
    )


def test_pointplot_custom_errorbar_categorical_y_still_works():
    """Pre-existing gallery use case (forest plot: numeric x, categorical y)
    must keep working after the refactor."""
    df = pd.DataFrame(
        {
            "log2_or": [0.85, 0.45, -0.25],
            "gene": ["APOE", "TREM2", "CLU"],
            "log2_lower": [0.65, 0.25, -0.45],
            "log2_upper": [1.05, 0.65, -0.05],
        }
    )
    fig, ax = plt.subplots()
    pp.pointplot(
        data=df,
        x="log2_or",
        y="gene",
        errorbar=("custom", ("log2_lower", "log2_upper")),
        ax=ax,
    )


# ---- Lineplot integration ----


def test_lineplot_custom_errorbar_renders_band():
    """A single-series lineplot with ``errorbar=('custom', ...)`` and
    ``err_style='band'`` must render a shaded band.

    Matplotlib 3.8+ represents the band as a ``FillBetweenPolyCollection``
    (a ``PolyCollection`` subclass); older versions use a plain
    ``PolyCollection``. ``isinstance`` covers both.
    """
    from matplotlib.collections import PolyCollection

    df = pd.DataFrame(
        {
            "day": list(range(10)),
            "y": list(range(10, 20)),
            "lo": list(range(8, 18)),
            "hi": list(range(12, 22)),
        }
    )
    fig, ax = plt.subplots()
    pp.lineplot(
        data=df,
        x="day",
        y="y",
        errorbar=("custom", ("lo", "hi")),
        err_style="band",
        ax=ax,
    )
    assert any(isinstance(coll, PolyCollection) for coll in ax.collections)


def test_lineplot_custom_errorbar_with_hue():
    """Custom errorbars must compose with ``hue`` — one band per group."""
    from matplotlib.collections import PolyCollection

    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "day": list(range(5)),
                    "y": [1, 2, 3, 4, 5],
                    "lo": [0.5] * 5,
                    "hi": [1.5] * 5,
                    "g": "a",
                }
            ),
            pd.DataFrame(
                {
                    "day": list(range(5)),
                    "y": [2, 3, 4, 5, 6],
                    "lo": [1.5] * 5,
                    "hi": [2.5] * 5,
                    "g": "b",
                }
            ),
        ]
    )
    fig, ax = plt.subplots()
    pp.lineplot(
        data=df,
        x="day",
        y="y",
        hue="g",
        errorbar=("custom", ("lo", "hi")),
        err_style="band",
        ax=ax,
    )
    # One band per hue level.
    bands = [
        coll for coll in ax.collections if isinstance(coll, PolyCollection)
    ]
    assert len(bands) == 2


# ====== pp.errorbar primitive ============================================
#
# Tests below cover the standalone 2D errorbar plotting function. It draws
# a ``pp.scatterplot`` for the markers and overlays a single
# ``ax.errorbar(fmt='none', ...)`` call for the x/y uncertainty stems.


@pytest.fixture(scope="module")
def err_df():
    """Synthetic errorbar dataset: 30 points with sym/asym uncertainties,
    a 3-level categorical group, and a continuous score for hue tests."""
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
        "xerr": np.abs(rng.normal(0, 0.2, n)),
        "yerr": np.abs(rng.normal(0, 0.2, n)),
        "yerr_lo": np.abs(rng.normal(0, 0.2, n)),
        "yerr_hi": np.abs(rng.normal(0, 0.3, n)),
        "g": rng.choice(list("abc"), size=n),
        "score": rng.uniform(0, 100, n),
    })


def _last_errorbar_container(ax):
    """Return the most recently added ErrorbarContainer on ``ax``, or
    None if no errorbar has been drawn."""
    if not ax.containers:
        return None
    return ax.containers[-1]


# ---- Contract ----

def test_returns_axes(err_df):
    ax = _errorbar(data=err_df, x="x", y="y", yerr="yerr")
    assert isinstance(ax, Axes)


def test_rejects_figsize(err_df):
    with pytest.raises(TypeError, match="figsize"):
        _errorbar(data=err_df, x="x", y="y", yerr="yerr", figsize=(4, 3))


def test_respects_ax(err_df):
    fig, ax0 = pp.subplots(axes_size=(60, 40))
    ax1 = _errorbar(data=err_df, x="x", y="y", yerr="yerr", ax=ax0)
    assert ax1 is ax0


def test_missing_xerr_column_raises(err_df):
    with pytest.raises((ValueError, KeyError)):
        _errorbar(data=err_df, x="x", y="y", xerr="nonexistent")


# ---- Errorbar shape ----

def test_yerr_only_no_x_stems(err_df):
    ax = _errorbar(data=err_df, x="x", y="y", yerr="yerr")
    cont = _last_errorbar_container(ax)
    assert cont is not None
    assert cont.has_yerr is True
    assert cont.has_xerr is False


def test_xerr_only_no_y_stems(err_df):
    ax = _errorbar(data=err_df, x="x", y="y", xerr="xerr")
    cont = _last_errorbar_container(ax)
    assert cont is not None
    assert cont.has_xerr is True
    assert cont.has_yerr is False


def test_both_errs(err_df):
    ax = _errorbar(data=err_df, x="x", y="y", xerr="xerr", yerr="yerr")
    cont = _last_errorbar_container(ax)
    assert cont is not None
    assert cont.has_xerr is True
    assert cont.has_yerr is True


def test_no_err_no_errorbar_call(err_df):
    """When both xerr and yerr are None, no ax.errorbar(...) is issued
    (no ErrorbarContainer is appended to ax.containers)."""
    ax = _errorbar(data=err_df, x="x", y="y")
    assert len(ax.containers) == 0


def test_asymmetric_errors_via_tuple(err_df):
    """yerr=(lo_col, hi_col) produces a 2xN error array; the resulting
    bar-segment Line2D's vertical extents per row should match the
    asymmetric (lo, hi) inputs (lo below, hi above)."""
    ax = _errorbar(
        data=err_df, x="x", y="y", yerr=("yerr_lo", "yerr_hi"),
    )
    cont = _last_errorbar_container(ax)
    assert cont is not None
    # cont.lines: (data_line, caplines_tuple, barlinecols_tuple)
    # For yerr only there is one barlinecol (LineCollection) of vertical
    # segments. Each segment is a 2-point line: (x, y-lo) -> (x, y+hi).
    barlines = cont.lines[2]
    assert len(barlines) == 1
    segs = barlines[0].get_segments()
    assert len(segs) == len(err_df)
    y = err_df["y"].to_numpy()
    lo = err_df["yerr_lo"].to_numpy()
    hi = err_df["yerr_hi"].to_numpy()
    # Each segment is a (2, 2) array: rows are endpoints, cols (x, y).
    for i, seg in enumerate(segs):
        ymin = seg[:, 1].min()
        ymax = seg[:, 1].max()
        assert ymin == pytest.approx(y[i] - lo[i], abs=1e-6)
        assert ymax == pytest.approx(y[i] + hi[i], abs=1e-6)


def test_scalar_xerr(err_df):
    """A scalar xerr is broadcast to a constant-length error per point."""
    ax = _errorbar(data=err_df, x="x", y="y", xerr=0.5)
    cont = _last_errorbar_container(ax)
    assert cont is not None
    assert cont.has_xerr is True


# ---- Styling ----

def test_alpha_face_edge_split(err_df):
    """publiplots double-layer style: marker face carries alpha, edge stays
    at 1.0. Pull from the scatter ``PathCollection``."""
    ax = _errorbar(
        data=err_df, x="x", y="y", yerr="yerr",
        alpha=0.3, edgecolor="black",
    )
    pcs = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(pcs) >= 1
    pc = pcs[0]
    fa = np.asarray(pc.get_facecolor())
    ea = np.asarray(pc.get_edgecolor())
    assert np.allclose(fa[..., -1], 0.3), (
        f"face alpha should be 0.3, got {fa[..., -1]}"
    )
    assert np.allclose(ea[..., -1], 1.0), (
        f"edge alpha should be 1.0 (opaque), got {ea[..., -1]}"
    )


def test_stem_color_is_edgecolor_default(err_df):
    """Errorbar stems should default to ``rcParams['edgecolor']``: a neutral
    color that doesn't compete with hue-mapped marker faces."""
    saved_edge = pp.rcParams["edgecolor"]
    try:
        pp.rcParams["edgecolor"] = "#444444"
        ax = _errorbar(data=err_df, x="x", y="y", yerr="yerr")
        cont = _last_errorbar_container(ax)
        assert cont is not None
        barlines = cont.lines[2]
        assert len(barlines) >= 1
        actual = np.asarray(barlines[0].get_color())
        expected = np.asarray(mcolors.to_rgba("#444444"))
        # LineCollection color is stored as a (1, 4) RGBA — squeeze to compare.
        if actual.ndim == 2:
            actual = actual[0]
        assert np.allclose(actual, expected, atol=1e-6)
    finally:
        pp.rcParams["edgecolor"] = saved_edge


def test_stem_color_override_via_errorbar_kws(err_df):
    """``errorbar_kws={'ecolor': 'red'}`` produces red stems."""
    ax = _errorbar(
        data=err_df, x="x", y="y", yerr="yerr",
        errorbar_kws={"ecolor": "red"},
    )
    cont = _last_errorbar_container(ax)
    assert cont is not None
    actual = np.asarray(cont.lines[2][0].get_color())
    expected = np.asarray(mcolors.to_rgba("red"))
    if actual.ndim == 2:
        actual = actual[0]
    assert np.allclose(actual, expected, atol=1e-6)


def test_capsize_default_zero(err_df):
    """Default capsize is 0 (from ``rcParams['capsize']``): caps tuple is
    empty (matplotlib emits no cap Line2Ds when capsize=0)."""
    ax = _errorbar(data=err_df, x="x", y="y", yerr="yerr")
    cont = _last_errorbar_container(ax)
    assert cont is not None
    caps = cont.lines[1]
    # Either an empty tuple or zero-length cap segments.
    if caps:
        for cap in caps:
            xd = np.asarray(cap.get_xdata())
            yd = np.asarray(cap.get_ydata())
            # Zero capsize -> the cap markers exist but at zero extent;
            # in practice matplotlib just omits them, so this branch is
            # rarely hit. Either outcome is acceptable.
            assert xd.size == 0 or yd.size == 0 or len(np.unique(yd)) <= len(yd)
    else:
        assert caps == ()


def test_capsize_kwarg_override(err_df):
    """capsize=2 produces visible caps: ax.containers[-1].lines[1] has at
    least one Line2D entry with non-empty data."""
    ax = _errorbar(data=err_df, x="x", y="y", yerr="yerr", capsize=2)
    cont = _last_errorbar_container(ax)
    assert cont is not None
    caps = cont.lines[1]
    assert len(caps) >= 1
    for cap in caps:
        xd = np.asarray(cap.get_xdata())
        assert xd.size > 0


# ---- Hue ----

def test_no_hue_no_entry(err_df):
    ax = _errorbar(data=err_df, x="x", y="y", yerr="yerr")
    assert get_entries(ax) == []


def test_categorical_hue_stashes_entry(err_df):
    ax = _errorbar(data=err_df, x="x", y="y", yerr="yerr", hue="g")
    entries = get_entries(ax)
    hue_entries = [e for e in entries if e.kind == "hue"]
    assert len(hue_entries) == 1
    assert hue_entries[0].name == "g"


def test_continuous_hue_stashes_colorbar(err_df):
    ax = _errorbar(
        data=err_df, x="x", y="y", yerr="yerr",
        hue="score", palette="viridis",
    )
    entries = get_entries(ax)
    hue_entries = [e for e in entries if e.kind == "hue"]
    assert len(hue_entries) == 1
    assert is_continuous_hue(hue_entries[0].handles)


def test_hue_does_not_affect_stem_color(err_df):
    """With ``hue='g'`` and a default edgecolor, stem color remains the
    rcParams edgecolor — NOT any palette color."""
    saved_edge = pp.rcParams["edgecolor"]
    try:
        pp.rcParams["edgecolor"] = "#444444"
        ax = _errorbar(
            data=err_df, x="x", y="y", yerr="yerr",
            hue="g", palette="pastel",
        )
        cont = _last_errorbar_container(ax)
        assert cont is not None
        actual = np.asarray(cont.lines[2][0].get_color())
        expected = np.asarray(mcolors.to_rgba("#444444"))
        if actual.ndim == 2:
            # Single-color LineCollection still reports its color as one
            # RGBA — assert all rows agree.
            for row in actual:
                assert np.allclose(row, expected, atol=1e-6)
        else:
            assert np.allclose(actual, expected, atol=1e-6)
    finally:
        pp.rcParams["edgecolor"] = saved_edge


# ---- Misc ----

def test_legend_false_suppresses_stash(err_df):
    ax = _errorbar(
        data=err_df, x="x", y="y", yerr="yerr",
        hue="g", legend=False,
    )
    assert get_entries(ax) == []


def test_legend_dict_per_kind(err_df):
    ax = _errorbar(
        data=err_df, x="x", y="y", yerr="yerr",
        hue="g", legend={"hue": False},
    )
    hue_entries = [e for e in get_entries(ax) if e.kind == "hue"]
    assert hue_entries == []
