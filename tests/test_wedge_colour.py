"""Hue resolution and depth/sibling colour derivation for wedge charts."""
from __future__ import annotations

import pandas as pd
import pytest

from publiplots.plot.wedge.colour import blend, derive_colors, resolve_hue_values
from publiplots.plot.wedge.geometry import build_nodes


def _nested():
    return pd.DataFrame({
        "k": ["A", "A", "B", "B"],
        "s": ["x", "y", "z", "w"],
        "region": ["north", "north", "south", "south"],
        "score": [1.0, 2.0, 3.0, 4.0],
        "n": [10.0, 20.0, 30.0, 40.0],
    })


def test_blend_toward_white_and_black():
    assert blend("#000000", 0.5) == pytest.approx((0.5, 0.5, 0.5))
    assert blend("#ffffff", -0.5) == pytest.approx((0.5, 0.5, 0.5))
    assert blend("#336699", 0.0) == pytest.approx((0.2, 0.4, 0.6), abs=1e-6)


def test_hue_defaults_to_the_root_level():
    df, levels = _nested(), ["k", "s"]
    nodes = build_nodes(df, levels, "n")
    values, continuous = resolve_hue_values(df, levels, None, nodes)
    assert continuous is False
    assert values[("A",)] == "A"
    assert values[("A", "x")] == "A"        # inherited from the root


def test_hue_may_name_a_column_outside_the_path():
    df, levels = _nested(), ["k", "s"]
    nodes = build_nodes(df, levels, "n")
    values, continuous = resolve_hue_values(df, levels, "region", nodes)
    assert continuous is False
    assert values[("A",)] == "north"
    assert values[("B", "z")] == "south"


def test_a_hue_that_varies_within_a_wedge_raises():
    df = pd.DataFrame({
        "k": ["A", "A"], "s": ["x", "y"],
        "region": ["north", "south"], "n": [1.0, 1.0],
    })
    nodes = build_nodes(df, ["k"], "n")
    with pytest.raises(ValueError, match=r"region.*\bA\b.*north.*south"):
        resolve_hue_values(df, ["k"], "region", nodes)


def test_numeric_hue_is_continuous_and_aggregates_by_mean():
    df, levels = _nested(), ["k", "s"]
    nodes = build_nodes(df, levels, "n")
    values, continuous = resolve_hue_values(df, levels, "score", nodes)
    assert continuous is True
    assert values[("A",)] == pytest.approx(1.5)     # mean of 1.0 and 2.0
    assert values[("A", "x")] == pytest.approx(1.0)


def test_one_level_colours_each_wedge_by_its_own_category():
    df = pd.DataFrame({"k": list("ABC"), "n": [1.0, 1.0, 1.0]})
    nodes = build_nodes(df, ["k"], "n")
    values, _ = resolve_hue_values(df, ["k"], None, nodes)
    colors = derive_colors(nodes, values, {"A": "#ff0000", "B": "#00ff00",
                                           "C": "#0000ff"},
                           depth_range=0.45, sibling_range=0.14, n_levels=1)
    assert colors[("A",)] == pytest.approx((1.0, 0.0, 0.0))
    assert colors[("B",)] == pytest.approx((0.0, 1.0, 0.0))


def test_knobs_are_inert_on_one_ring():
    df = pd.DataFrame({"k": list("AB"), "n": [1.0, 1.0]})
    nodes = build_nodes(df, ["k"], "n")
    values, _ = resolve_hue_values(df, ["k"], None, nodes)
    flat = derive_colors(nodes, values, {"A": "#336699", "B": "#993366"},
                         depth_range=0.0, sibling_range=0.0, n_levels=1)
    loud = derive_colors(nodes, values, {"A": "#336699", "B": "#993366"},
                         depth_range=0.9, sibling_range=0.9, n_levels=1)
    assert flat == loud


def test_depth_range_is_the_total_span_not_a_per_ring_step():
    """The rim lands at depth_range regardless of how deep the tree is."""
    shallow = pd.DataFrame({"a": ["A", "A"], "b": ["x", "y"], "n": [1.0, 1.0]})
    deep = pd.DataFrame({
        "a": ["A"] * 2, "b": ["x"] * 2, "c": ["p"] * 2,
        "d": ["q"] * 2, "e": ["m", "n"], "n": [1.0, 1.0],
    })
    for df, levels in ((shallow, ["a", "b"]), (deep, ["a", "b", "c", "d", "e"])):
        nodes = build_nodes(df, levels, "n")
        values, _ = resolve_hue_values(df, levels, None, nodes)
        colors = derive_colors(nodes, values, {"A": "#000000"},
                               depth_range=0.5, sibling_range=0.0,
                               n_levels=len(levels))
        rim = [c for p, c in colors.items() if len(p) == len(levels)]
        for channel in rim[0]:
            assert channel == pytest.approx(0.5, abs=1e-9)


def test_negative_depth_range_darkens_outward():
    df = pd.DataFrame({"a": ["A", "A"], "b": ["x", "y"], "n": [1.0, 1.0]})
    nodes = build_nodes(df, ["a", "b"], "n")
    values, _ = resolve_hue_values(df, ["a", "b"], None, nodes)
    colors = derive_colors(nodes, values, {"A": "#ffffff"},
                           depth_range=-0.5, sibling_range=0.0, n_levels=2)
    assert colors[("A", "x")][0] == pytest.approx(0.5)


def test_siblings_vary_within_their_parent():
    df = pd.DataFrame({
        "a": ["A"] * 3, "b": ["x", "y", "z"], "n": [1.0, 1.0, 1.0],
    })
    nodes = build_nodes(df, ["a", "b"], "n")
    values, _ = resolve_hue_values(df, ["a", "b"], None, nodes)
    colors = derive_colors(nodes, values, {"A": "#000000"},
                           depth_range=0.0, sibling_range=0.3, n_levels=2)
    kids = [colors[("A", k)][0] for k in ("x", "y", "z")]
    assert kids[0] < kids[1] < kids[2]
    assert kids[2] == pytest.approx(0.3)


def test_an_only_child_takes_no_sibling_offset():
    df = pd.DataFrame({"a": ["A"], "b": ["x"], "n": [1.0]})
    nodes = build_nodes(df, ["a", "b"], "n")
    values, _ = resolve_hue_values(df, ["a", "b"], None, nodes)
    colors = derive_colors(nodes, values, {"A": "#000000"},
                           depth_range=0.0, sibling_range=0.5, n_levels=2)
    assert colors[("A", "x")] == pytest.approx((0.0, 0.0, 0.0))
