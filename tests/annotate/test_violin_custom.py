"""Tests for the violin_custom strategy."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate._cache import BoxStatsRecord


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _simple_df():
    # 10 samples per cat for stable KDE
    return pd.DataFrame({
        "cat": pd.Categorical(
            ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            categories=["A", "B", "C"],
        ),
        "value": list(range(1, 31)),
        "n":     [10] * 10 + [20] * 10 + [30] * 10,
    })


def test_column_labels_from_cached_frame():
    ax = pp.violinplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(ax, kind="violin_custom", labels="n")
    assert [t.get_text() for t in texts] == ["10", "20", "30"]


def test_column_labels_with_fmt():
    ax = pp.violinplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(ax, kind="violin_custom", labels="n", fmt="n={:,}")
    assert [t.get_text() for t in texts] == ["n=10", "n=20", "n=30"]


def test_column_labels_with_data_override():
    ax = pp.violinplot(data=_simple_df(), x="cat", y="value")
    other = _simple_df().assign(n=[99] * 10 + [88] * 10 + [77] * 10)
    texts = pp.annotate(ax, kind="violin_custom", labels="n", data=other)
    assert [t.get_text() for t in texts] == ["99", "88", "77"]


def test_column_labels_hue_split():
    df = pd.DataFrame({
        "cat": pd.Categorical(
            ["A"] * 20 + ["B"] * 20, categories=["A", "B"],
        ),
        "hue": pd.Categorical(
            (["x"] * 10 + ["y"] * 10) * 2, categories=["x", "y"],
        ),
        "value": list(range(1, 41)),
        "n": [11] * 10 + [22] * 10 + [33] * 10 + [44] * 10,
    })
    ax = pp.violinplot(data=df, x="cat", y="value", hue="hue")
    texts = pp.annotate(ax, kind="violin_custom", labels="n")
    assert [t.get_text() for t in texts] == ["11", "33", "22", "44"]


def test_column_labels_intra_group_variance_warns_and_uses_first():
    df = pd.DataFrame({
        "cat": pd.Categorical(
            ["A"] * 10 + ["B"] * 10, categories=["A", "B"],
        ),
        "value": list(range(1, 21)),
        "n": [10] * 9 + [999] + [20] * 10,
    })
    ax = pp.violinplot(data=df, x="cat", y="value")
    with pytest.warns(UserWarning, match="varies within group"):
        texts = pp.annotate(ax, kind="violin_custom", labels="n")
    assert [t.get_text() for t in texts] == ["10", "20"]


def test_callable_receives_box_record():
    seen = []
    def label(rec):
        assert isinstance(rec, BoxStatsRecord)
        seen.append(rec)
        return str(rec.category)
    ax = pp.violinplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(ax, kind="violin_custom", labels=label)
    assert [t.get_text() for t in texts] == ["A", "B", "C"]
    assert len(seen) == 3


def test_callable_returning_none_skips_violin():
    ax = pp.violinplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(
        ax, kind="violin_custom",
        labels=lambda r: None if r.category == "B" else str(r.category),
    )
    assert [t.get_text() for t in texts] == ["A", "C"]


def test_callable_returning_non_string_raises_typeerror():
    ax = pp.violinplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(TypeError, match="draw_index=0"):
        pp.annotate(ax, kind="violin_custom", labels=lambda r: 42)


def test_fmt_warns_when_labels_is_callable():
    ax = pp.violinplot(data=_simple_df(), x="cat", y="value")
    with pytest.warns(UserWarning, match="fmt is ignored"):
        pp.annotate(
            ax, kind="violin_custom",
            labels=lambda r: str(r.category),
            fmt="n={}",
        )


def test_missing_labels_raises():
    ax = pp.violinplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(ValueError, match="labels"):
        pp.annotate(ax, kind="violin_custom")


def test_foreign_axes_warns_and_returns_empty():
    fig, ax = plt.subplots()
    ax.violinplot([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    with pytest.warns(UserWarning, match="publiplots-owned axes"):
        texts = pp.annotate(ax, kind="violin_custom", labels="n")
    assert texts == []


def test_column_not_in_frame_raises_keyerror():
    ax = pp.violinplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(KeyError, match="not_a_col"):
        pp.annotate(ax, kind="violin_custom", labels="not_a_col")


def test_labels_wrong_type_raises():
    ax = pp.violinplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(TypeError, match="labels must be"):
        pp.annotate(ax, kind="violin_custom", labels=123)


def test_end_to_end_via_pp_violinplot_annotate():
    ax = pp.violinplot(
        data=_simple_df(), x="cat", y="value",
        annotate={"kind": "violin_custom", "labels": "n", "fmt": "n={}"},
    )
    assert [t.get_text() for t in ax.texts] == ["n=10", "n=20", "n=30"]
