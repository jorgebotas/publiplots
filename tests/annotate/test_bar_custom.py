"""Tests for the bar_custom strategy."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate._cache import BarRecord


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _simple_df():
    return pd.DataFrame({
        "cat": pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"]),
        "value": [1.0, 2.0, 3.0],
        "n": [10, 20, 30],
    })


# -----------------------------------------------------------------------------
# Column-source path
# -----------------------------------------------------------------------------

def test_column_labels_from_cached_frame():
    ax = pp.barplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(ax, kind="bar_custom", labels="n")
    assert [t.get_text() for t in texts] == ["10", "20", "30"]


def test_column_labels_with_fmt():
    ax = pp.barplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(ax, kind="bar_custom", labels="n", fmt="n={:,}")
    assert [t.get_text() for t in texts] == ["n=10", "n=20", "n=30"]


def test_column_labels_with_data_override():
    ax = pp.barplot(data=_simple_df(), x="cat", y="value")
    other = _simple_df().assign(n=[99, 88, 77])
    texts = pp.annotate(ax, kind="bar_custom", labels="n", data=other)
    assert [t.get_text() for t in texts] == ["99", "88", "77"]


def test_column_labels_hue_split():
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
        "hue": pd.Categorical(["x", "y", "x", "y"], categories=["x", "y"]),
        "value": [1.0, 2.0, 3.0, 4.0],
        "n": [11, 22, 33, 44],
    })
    ax = pp.barplot(data=df, x="cat", y="value", hue="hue")
    texts = pp.annotate(ax, kind="bar_custom", labels="n")
    # Draw order: hue x -> (A, B), then hue y -> (A, B)
    assert [t.get_text() for t in texts] == ["11", "33", "22", "44"]


def test_column_labels_intra_group_variance_warns_and_uses_first():
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B"], categories=["A", "B"]),
        "value": [1.0, 2.0, 3.0],
        "n": [10, 999, 30],  # intra-group mismatch for cat=A
    })
    ax = pp.barplot(data=df, x="cat", y="value")
    with pytest.warns(UserWarning, match="varies within group"):
        texts = pp.annotate(ax, kind="bar_custom", labels="n")
    assert [t.get_text() for t in texts] == ["10", "30"]


# -----------------------------------------------------------------------------
# Callable-source path
# -----------------------------------------------------------------------------

def test_callable_receives_bar_record():
    seen = []

    def label(rec):
        assert isinstance(rec, BarRecord)
        seen.append(rec)
        return str(rec.category)

    ax = pp.barplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(ax, kind="bar_custom", labels=label)
    assert [t.get_text() for t in texts] == ["A", "B", "C"]
    assert len(seen) == 3


def test_callable_returning_none_skips_bar():
    ax = pp.barplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(
        ax, kind="bar_custom",
        labels=lambda r: None if r.category == "B" else str(r.category),
    )
    assert [t.get_text() for t in texts] == ["A", "C"]


def test_callable_returning_non_string_raises_typeerror():
    ax = pp.barplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(TypeError, match="draw_index=0"):
        pp.annotate(
            ax, kind="bar_custom",
            labels=lambda r: 42,  # not a string
        )


def test_fmt_warns_when_labels_is_callable():
    ax = pp.barplot(data=_simple_df(), x="cat", y="value")
    with pytest.warns(UserWarning, match="fmt is ignored"):
        pp.annotate(
            ax, kind="bar_custom",
            labels=lambda r: str(r.category),
            fmt="n={}",
        )


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------

def test_missing_labels_raises():
    ax = pp.barplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(ValueError, match="labels"):
        pp.annotate(ax, kind="bar_custom")


def test_column_without_cached_df_or_data_raises():
    # Use foreign axes (no pp.barplot) so there's no cached source_frame.
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0])
    fig.canvas.draw()
    with pytest.raises(ValueError, match="data"):
        pp.annotate(ax, kind="bar_custom", labels="n")


def test_column_not_in_frame_raises_keyerror():
    ax = pp.barplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(KeyError, match="not_a_col"):
        pp.annotate(ax, kind="bar_custom", labels="not_a_col")


def test_labels_wrong_type_raises():
    ax = pp.barplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(TypeError, match="labels must be"):
        pp.annotate(ax, kind="bar_custom", labels=123)


# -----------------------------------------------------------------------------
# Foreign axes
# -----------------------------------------------------------------------------

def test_foreign_axes_callable_works_via_draw_index():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0])
    fig.canvas.draw()
    ns = [10, 20, 30]
    texts = pp.annotate(
        ax, kind="bar_custom",
        labels=lambda r: f"n={ns[r.draw_index]}",
    )
    assert [t.get_text() for t in texts] == ["n=10", "n=20", "n=30"]


def test_foreign_axes_column_with_data_raises_notimplemented():
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [1.0, 2.0, 3.0])
    fig.canvas.draw()
    df = _simple_df()
    with pytest.raises(NotImplementedError, match="foreign"):
        pp.annotate(ax, kind="bar_custom", labels="n", data=df)


# -----------------------------------------------------------------------------
# NaN skip
# -----------------------------------------------------------------------------

def test_nan_value_bar_skipped():
    """Bars with NaN value are skipped just like in bar_values."""
    import math
    from publiplots.annotate._cache import BarValueMeta

    ax = pp.barplot(data=_simple_df(), x="cat", y="value")
    bars = ax._publiplots_bar_meta.bars
    # Replace middle bar with a NaN-value copy (frozen dataclass: rebuild)
    import dataclasses
    bars[1] = dataclasses.replace(bars[1], value=float("nan"))
    texts = pp.annotate(
        ax, kind="bar_custom",
        labels=lambda r: str(r.category),
    )
    assert [t.get_text() for t in texts] == ["A", "C"]


def test_column_labels_hatch_only_split():
    """Hatch-only splits (no hue) must also align column labels correctly."""
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
        "hatch": pd.Categorical(["h1", "h2", "h1", "h2"], categories=["h1", "h2"]),
        "value": [1.0, 2.0, 3.0, 4.0],
        "n": [11, 22, 33, 44],
    })
    ax = pp.barplot(data=df, x="cat", y="value", hatch="hatch")
    texts = pp.annotate(ax, kind="bar_custom", labels="n")
    # Draw order is hatch-outer / cat-inner: h1(A, B), h2(A, B)
    assert [t.get_text() for t in texts] == ["11", "33", "22", "44"]
