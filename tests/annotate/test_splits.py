"""Tests for the shared BarSplitSpec — the single source of truth for
which dimensions cause bar dodging.
"""
import pandas as pd
import pytest

from publiplots.annotate._splits import BarSplitSpec


def _df():
    return pd.DataFrame({
        "c": pd.Categorical(["A", "B", "A", "B"]),
        "h": pd.Categorical(["x", "x", "y", "y"]),
        "t": pd.Categorical(["p", "q", "p", "q"]),
        "v": [1.0, 2.0, 3.0, 4.0],
    })


def test_split_no_hue_no_hatch():
    spec = BarSplitSpec.resolve(x="c", y="v", hue=None, hatch=None,
                                categorical_axis="c")
    assert spec.split_hue is None
    assert spec.split_hatch is None
    assert spec.orient == "v"
    assert spec.n_dodge_dims == 0


def test_split_hue_equals_cat_collapses():
    spec = BarSplitSpec.resolve(x="c", y="v", hue="c", hatch=None,
                                categorical_axis="c")
    assert spec.split_hue is None


def test_split_hatch_equals_cat_collapses():
    spec = BarSplitSpec.resolve(x="c", y="v", hue=None, hatch="c",
                                categorical_axis="c")
    assert spec.split_hatch is None


def test_split_hatch_equals_hue_collapses():
    spec = BarSplitSpec.resolve(x="c", y="v", hue="h", hatch="h",
                                categorical_axis="c")
    assert spec.split_hue == "h"
    assert spec.split_hatch is None


def test_split_full_double():
    spec = BarSplitSpec.resolve(x="c", y="v", hue="h", hatch="t",
                                categorical_axis="c")
    assert spec.split_hue == "h"
    assert spec.split_hatch == "t"
    assert spec.n_dodge_dims == 2


def test_split_horizontal_orient():
    spec = BarSplitSpec.resolve(x="v", y="c", hue=None, hatch=None,
                                categorical_axis="c")
    assert spec.orient == "h"


def test_iter_draw_order_hue_outer_cat_inner():
    """With just hue: all bars of hue-x, then all of hue-y — each containing
    cats A then B."""
    df = _df()
    spec = BarSplitSpec.resolve(x="c", y="v", hue="h", hatch=None,
                                categorical_axis="c")
    order = list(spec.iter_draw_order(df))
    assert order == [
        ("A", "x", None), ("B", "x", None),
        ("A", "y", None), ("B", "y", None),
    ]


def test_iter_draw_order_double_split():
    """hue outer, hatch middle, cat inner."""
    df = _df()
    # Change df so all hue × hatch × cat combos exist
    df = pd.DataFrame({
        "c": pd.Categorical(["A", "B"] * 4),
        "h": pd.Categorical(["x", "x", "x", "x", "y", "y", "y", "y"]),
        "t": pd.Categorical(["p", "p", "q", "q"] * 2),
        "v": list(range(8)),
    })
    spec = BarSplitSpec.resolve(x="c", y="v", hue="h", hatch="t",
                                categorical_axis="c")
    order = list(spec.iter_draw_order(df))
    assert order == [
        ("A", "x", "p"), ("B", "x", "p"),
        ("A", "x", "q"), ("B", "x", "q"),
        ("A", "y", "p"), ("B", "y", "p"),
        ("A", "y", "q"), ("B", "y", "q"),
    ]


def test_iter_draw_order_skips_empty_combinations():
    """When hue == cat, hue_val and cat agree → only diagonal yields rows."""
    df = pd.DataFrame({
        "c": pd.Categorical(["A", "B", "C"]),
        "v": [1.0, 2.0, 3.0],
    })
    spec = BarSplitSpec.resolve(x="c", y="v", hue="c", hatch=None,
                                categorical_axis="c")
    # hue collapsed, so iter yields one tuple per cat with hue_value=None
    order = list(spec.iter_draw_order(df))
    assert order == [("A", None, None), ("B", None, None), ("C", None, None)]
