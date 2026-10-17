"""Integration: pp.boxplot + pp.violinplot cache-building + end-to-end."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def test_boxplot_annotate_kind_forwarded():
    df = _box_df()
    ax = pp.boxplot(
        data=df, x="cat", y="value",
        annotate={"kind": "box_custom", "labels": "n", "fmt": "n={}"},
    )
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["n=10", "n=20", "n=30"]


def test_violinplot_annotate_kind_forwarded():
    df = _box_df()
    ax = pp.violinplot(
        data=df, x="cat", y="value",
        annotate={"kind": "violin_custom", "labels": "n", "fmt": "n={}"},
    )
    labels = [t.get_text() for t in ax.texts]
    assert labels == ["n=10", "n=20", "n=30"]


def test_boxplot_nan_group_gets_a_finite_median_label():
    df = _box_df()
    df.loc[0, "value"] = np.nan   # group A: median of [2, 3] is 2.5
    ax = pp.boxplot(data=df, x="cat", y="value", annotate=True)
    assert [t.get_text() for t in ax.texts] == ["2.50", "3.00", "4.00"]


def test_boxplot_all_nan_group_is_skipped_like_an_empty_one():
    df = _box_df()
    df.loc[df["cat"] == "B", "value"] = np.nan
    ax = pp.boxplot(data=df, x="cat", y="value", annotate=True)
    boxes = ax._publiplots_box_meta.boxes
    assert [b.category for b in boxes] == ["A", "C"]
    assert [b.center_pos for b in boxes] == pytest.approx([0, 2], abs=1e-9)
    assert [t.get_text() for t in ax.texts] == ["2.00", "4.00"]
