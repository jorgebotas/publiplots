"""Hue resolution and depth/sibling colour derivation for wedge charts."""
from __future__ import annotations

import pandas as pd
import pytest

import publiplots as pp
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
    # A deeper node takes its root's value: the column is a root attribute.
    assert values[("A", "x")] == pytest.approx(1.5)


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


def test_object_dtype_numeric_hue_is_continuous():
    """An object-dtype column of plain numbers is still a continuous hue --
    seaborn's ``all_numeric`` fallback (#262), not a dtype check alone."""
    df = pd.DataFrame({
        "k": list("ABC"),
        "score": pd.Series([1, 2, 3], dtype=object),
        "n": [1.0, 1.0, 1.0],
    })
    nodes = build_nodes(df, ["k"], "n")
    values, continuous = resolve_hue_values(df, ["k"], "score", nodes)
    assert continuous is True
    assert values[("A",)] == pytest.approx(1.0)


def test_bool_hue_is_categorical():
    """Unlike seaborn's own numeric rule, a boolean hue stays categorical
    here -- a colorbar over {False, True} would be nonsense."""
    df = pd.DataFrame({
        "k": list("ABC"),
        "flag": [True, False, True],
        "n": [1.0, 1.0, 1.0],
    })
    nodes = build_nodes(df, ["k"], "n")
    values, continuous = resolve_hue_values(df, ["k"], "flag", nodes)
    assert continuous is False
    assert values[("A",)] is True


def test_hue_naming_a_non_root_level_raises():
    df, levels = _nested(), ["k", "s"]
    nodes = build_nodes(df, levels, "n")
    with pytest.raises(ValueError, match=r"^pieplot: hue='s' names ring 1"):
        resolve_hue_values(df, levels, "s", nodes)


def test_missing_hue_value_warns_and_falls_back_to_grey():
    """A NaN in the hue column drops that node from resolve_hue_values;
    derive_colors must still colour it, but say so once per call."""
    df = pd.DataFrame({
        "a": ["A", "B"], "b": ["x", "y"],
        "region": [None, "north"], "n": [1.0, 1.0],
    })
    nodes = build_nodes(df, ["a", "b"], "n")
    values, _ = resolve_hue_values(df, ["a", "b"], "region", nodes)
    assert ("A", "x") not in values

    with pytest.warns(UserWarning, match=r"region"):
        colors = derive_colors(nodes, values, {"north": "#000000"},
                               depth_range=0.0, sibling_range=0.0,
                               n_levels=2, hue_column="region")
    import matplotlib.colors as mcolors
    assert colors[("A", "x")] == pytest.approx(mcolors.to_rgb("#cccccc"))


def test_sunburst_rcparams_defaults():
    assert pp.rcParams["sunburst.depth_range"] == 0.45
    assert pp.rcParams["sunburst.sibling_range"] == 0.14


def test_check_shading_bounds_the_total_span():
    import pytest
    from publiplots.plot.wedge.colour import check_shading
    check_shading(0.6, 0.4, owner="sunburstplot")
    with pytest.raises(ValueError, match=r"^sunburstplot: depth_range"):
        check_shading(0.7, -0.4, owner="sunburstplot")


def test_continuous_hue_is_the_per_root_row_mean_to_the_last_bit():
    """One level must reproduce the old per-node mean exactly: the pie
    gallery's output bytes hang on it, and GroupBy.mean sums in a different
    order than Series.mean (it differs in the last bits on this data)."""
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(50):
        df = pd.DataFrame({
            "k": rng.choice(list("ABCDE"), 40),
            "score": rng.normal(size=40) * 10.0 ** rng.integers(-3, 4, 40),
            "n": 1.0,
        })
        nodes = build_nodes(df, ["k"], "n")
        values, continuous = resolve_hue_values(df, ["k"], "score", nodes)
        assert continuous is True
        for node in nodes:
            expected = float(df.loc[df["k"] == node.path[0], "score"].mean())
            assert values[node.path] == expected


def test_not_constant_within_names_the_root_and_the_values_found():
    df = _nested().assign(region=["north", "south", "south", "south"])
    nodes = build_nodes(df, ["k", "s"], "n")
    with pytest.raises(ValueError) as err:
        resolve_hue_values(df, ["k", "s"], "region", nodes, owner="sunburstplot")
    msg = str(err.value)
    assert msg.startswith("sunburstplot: hue='region' is not constant within "
                          "the wedge ('A',)")
    assert "'north'" in msg and "'south'" in msg


def test_every_message_begins_with_the_owner():
    df, levels = _nested(), ["k", "s"]
    nodes = build_nodes(df, levels, "n")
    with pytest.raises(ValueError, match=r"^sunburstplot: hatch='zzz'"):
        resolve_hue_values(df, levels, "zzz", nodes, param_name="hatch",
                           owner="sunburstplot")
    with pytest.raises(ValueError, match=r"^sunburstplot: hue='s' names ring 1"):
        resolve_hue_values(df, levels, "s", nodes, owner="sunburstplot")


def test_allow_level_hands_ring_k_values_to_ring_k_and_deeper_only():
    df, levels = _nested(), ["k", "s"]
    nodes = build_nodes(df, levels, "n")
    values, continuous = resolve_hue_values(df, levels, "s", nodes,
                                            allow_level=True)
    assert continuous is False
    assert values == {("A", "x"): "x", ("A", "y"): "y",
                      ("B", "z"): "z", ("B", "w"): "w"}


def test_a_root_attribute_is_inherited_by_every_node_of_the_root():
    df, levels = _nested(), ["k", "s"]
    nodes = build_nodes(df, levels, "n")
    values, _ = resolve_hue_values(df, levels, "region", nodes)
    assert values == {("A",): "north", ("A", "x"): "north",
                      ("A", "y"): "north", ("B",): "south",
                      ("B", "z"): "south", ("B", "w"): "south"}
