"""Integration: pp.pointplot(..., annotate=...) cache-building + end-to-end."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
