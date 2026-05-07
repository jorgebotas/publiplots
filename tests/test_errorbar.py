"""Tests for the shared custom-errorbar helper and its plot-function
integrations."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp
from publiplots.utils.errorbar import format_for_custom_errorbar


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
