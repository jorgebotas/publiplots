"""Tests for the point_custom strategy."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate._cache import PointRecord


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


def test_column_labels_from_cached_frame():
    ax = pp.pointplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(ax, kind="point_custom", labels="n")
    assert [t.get_text() for t in texts] == ["10", "20", "30"]


def test_column_labels_with_fmt():
    ax = pp.pointplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(ax, kind="point_custom", labels="n", fmt="n={:,}")
    assert [t.get_text() for t in texts] == ["n=10", "n=20", "n=30"]


def test_column_labels_with_data_override():
    ax = pp.pointplot(data=_simple_df(), x="cat", y="value")
    other = _simple_df().assign(n=[99, 88, 77])
    texts = pp.annotate(ax, kind="point_custom", labels="n", data=other)
    assert [t.get_text() for t in texts] == ["99", "88", "77"]


def test_column_labels_hue_split():
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
        "hue": pd.Categorical(["x", "y", "x", "y"], categories=["x", "y"]),
        "value": [1.0, 2.0, 3.0, 4.0],
        "n": [11, 22, 33, 44],
    })
    ax = pp.pointplot(data=df, x="cat", y="value", hue="hue")
    texts = pp.annotate(ax, kind="point_custom", labels="n")
    assert [t.get_text() for t in texts] == ["11", "33", "22", "44"]


def test_column_labels_intra_group_variance_warns_and_uses_first():
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "A", "B"], categories=["A", "B"]),
        "value": [1.0, 2.0, 3.0],
        "n": [10, 999, 30],
    })
    ax = pp.pointplot(data=df, x="cat", y="value")
    with pytest.warns(UserWarning, match="varies within group"):
        texts = pp.annotate(ax, kind="point_custom", labels="n")
    assert [t.get_text() for t in texts] == ["10", "30"]


def test_callable_receives_point_record():
    seen = []
    def label(rec):
        assert isinstance(rec, PointRecord)
        seen.append(rec)
        return str(rec.category)
    ax = pp.pointplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(ax, kind="point_custom", labels=label)
    assert [t.get_text() for t in texts] == ["A", "B", "C"]
    assert len(seen) == 3


def test_callable_returning_none_skips_point():
    ax = pp.pointplot(data=_simple_df(), x="cat", y="value")
    texts = pp.annotate(
        ax, kind="point_custom",
        labels=lambda r: None if r.category == "B" else str(r.category),
    )
    assert [t.get_text() for t in texts] == ["A", "C"]


def test_callable_returning_non_string_raises_typeerror():
    ax = pp.pointplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(TypeError, match="draw_index=0"):
        pp.annotate(ax, kind="point_custom", labels=lambda r: 42)


def test_fmt_warns_when_labels_is_callable():
    ax = pp.pointplot(data=_simple_df(), x="cat", y="value")
    with pytest.warns(UserWarning, match="fmt is ignored"):
        pp.annotate(
            ax, kind="point_custom",
            labels=lambda r: str(r.category),
            fmt="n={}",
        )


def test_missing_labels_raises():
    ax = pp.pointplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(ValueError, match="labels"):
        pp.annotate(ax, kind="point_custom")


def test_foreign_axes_warns_and_returns_empty():
    """point_custom has introspect=None; foreign axes emit a UserWarning
    and return an empty list (matching _custom.py's behavior)."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1.0, 2.0, 3.0], "o")
    with pytest.warns(UserWarning, match="publiplots-owned axes"):
        texts = pp.annotate(ax, kind="point_custom", labels="n")
    assert texts == []


def test_column_not_in_frame_raises_keyerror():
    ax = pp.pointplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(KeyError, match="not_a_col"):
        pp.annotate(ax, kind="point_custom", labels="not_a_col")


def test_labels_wrong_type_raises():
    ax = pp.pointplot(data=_simple_df(), x="cat", y="value")
    with pytest.raises(TypeError, match="labels must be"):
        pp.annotate(ax, kind="point_custom", labels=123)


def test_point_record_is_publicly_importable():
    from publiplots.annotate import PointRecord as Public
    from publiplots.annotate._cache import PointRecord as Private
    assert Public is Private


def test_end_to_end_via_pp_pointplot_annotate():
    ax = pp.pointplot(
        data=_simple_df(), x="cat", y="value",
        annotate={"kind": "point_custom", "labels": "n", "fmt": "n={}"},
    )
    assert [t.get_text() for t in ax.texts] == ["n=10", "n=20", "n=30"]
