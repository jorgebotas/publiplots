"""Integration: pp.boxplot + pp.violinplot cache-building + end-to-end."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate._cache import BoxStatsMeta


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _box_df():
    return pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "A", "B", "B", "B", "C", "C", "C"]),
        "value": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0],
        "n":     [10, 10, 10, 20, 20, 20, 30, 30, 30],
    })


def test_boxplot_without_annotate_still_attaches_meta():
    df = _box_df()
    ax = pp.boxplot(data=df, x="cat", y="value")
    meta = ax._publiplots_box_meta
    assert isinstance(meta, BoxStatsMeta)
    assert meta.source_frame is df
    assert meta.group_keys == ("cat",)


def test_violinplot_without_annotate_still_attaches_meta():
    df = _box_df()
    ax = pp.violinplot(data=df, x="cat", y="value")
    meta = ax._publiplots_box_meta
    assert isinstance(meta, BoxStatsMeta)
    assert meta.source_frame is df


@pytest.mark.skip(reason="strategy registered in Task 7")
def test_boxplot_annotate_kind_forwarded():
    df = _box_df()
    ax = pp.boxplot(
        data=df, x="cat", y="value",
        annotate={"kind": "box_custom", "labels": "n", "fmt": "n={}"},
    )
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["n=10", "n=20", "n=30"]


@pytest.mark.skip(reason="strategy registered in Task 8")
def test_violinplot_annotate_kind_forwarded():
    df = _box_df()
    ax = pp.violinplot(
        data=df, x="cat", y="value",
        annotate={"kind": "violin_custom", "labels": "n", "fmt": "n={}"},
    )
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["n=10", "n=20", "n=30"]
