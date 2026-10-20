"""Tree building and angular layout for the wedge renderer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from publiplots.plot.wedge.geometry import (
    build_nodes,
    has_children,
    layout,
    total_value,
)


def _flat():
    return pd.DataFrame({"kind": list("ABC"), "n": [50.0, 30.0, 20.0]})


def _nested():
    return pd.DataFrame({
        "k": ["A", "A", "A", "B", "B"],
        "s": ["x", "x", "y", "z", "w"],
        "n": [10.0, 5.0, 15.0, 20.0, 10.0],
    })


def test_flat_tree_has_one_node_per_category():
    nodes = build_nodes(_flat(), ["kind"], "n")
    assert [n.path for n in nodes] == [("A",), ("B",), ("C",)]
    assert [n.value for n in nodes] == pytest.approx([50.0, 30.0, 20.0])
    assert {n.depth for n in nodes} == {0}


def test_value_omitted_counts_rows():
    df = pd.DataFrame({"kind": list("AABBB")})
    nodes = build_nodes(df, ["kind"], None)
    assert [(n.path, n.value) for n in nodes] == [(("A",), 2.0), (("B",), 3.0)]


def test_parent_is_the_sum_of_its_children():
    nodes = build_nodes(_nested(), ["k", "s"], "n")
    by_path = {n.path: n.value for n in nodes}
    assert by_path[("A",)] == pytest.approx(30.0)   # 10 + 5 + 15
    assert by_path[("B",)] == pytest.approx(30.0)   # 20 + 10
    assert by_path[("A", "x")] == pytest.approx(15.0)


def test_nodes_are_depth_first_root_ring_first():
    nodes = build_nodes(_nested(), ["k", "s"], "n")
    assert [n.path for n in nodes] == [
        ("A",), ("A", "x"), ("A", "y"),
        ("B",), ("B", "z"), ("B", "w"),
    ]


def test_sibling_index_and_count_are_within_parent():
    nodes = build_nodes(_nested(), ["k", "s"], "n")
    kids = [n for n in nodes if n.parent == ("A",)]
    assert [(n.sibling_index, n.sibling_count) for n in kids] == [(0, 2), (1, 2)]


def test_explicit_order_wins():
    nodes = build_nodes(_flat(), ["kind"], "n", order={"kind": ["C", "A", "B"]})
    assert [n.path[0] for n in nodes] == ["C", "A", "B"]


def test_order_naming_a_subset_filters_rather_than_truncates():
    nodes = build_nodes(_flat(), ["kind"], "n", order={"kind": ["B", "C"]})
    assert [n.path[0] for n in nodes] == ["B", "C"]


def test_order_naming_an_absent_category_drops_it():
    nodes = build_nodes(_flat(), ["kind"], "n", order={"kind": ["A", "ZZ", "B"]})
    assert [n.path[0] for n in nodes] == ["A", "B"]


def test_numeric_levels_sort():
    df = pd.DataFrame({"dose": [3, 1, 2], "n": [1.0, 2.0, 3.0]})
    nodes = build_nodes(df, ["dose"], "n")
    assert [n.path[0] for n in nodes] == [1, 2, 3]


def test_sort_orders_by_value_within_each_parent():
    """Values chosen so nothing ties -- a test that accepts two answers
    pins nothing. Equal values keep first-resolved order via stable sort,
    which needs no separate test."""
    df = pd.DataFrame({
        "k": ["A", "A", "A", "B", "B"],
        "s": ["x", "x", "y", "z", "w"],
        "n": [10.0, 5.0, 40.0, 20.0, 10.0],
    })
    nodes = build_nodes(df, ["k", "s"], "n", sort=True)
    assert [n.path for n in nodes] == [
        ("A",), ("A", "y"), ("A", "x"),   # 40 then 15
        ("B",), ("B", "z"), ("B", "w"),   # 20 then 10
    ]


def test_sort_is_per_parent_not_global():
    df = pd.DataFrame({
        "k": ["A", "A", "B", "B"],
        "s": ["small", "big", "small", "big"],
        "n": [1.0, 100.0, 2.0, 50.0],
    })
    nodes = build_nodes(df, ["k", "s"], "n", sort=True)
    # Every child stays under its own parent; B's big child does not migrate
    # above A's small one.
    assert [n.path for n in nodes] == [
        ("A",), ("A", "big"), ("A", "small"),
        ("B",), ("B", "big"), ("B", "small"),
    ]


def test_sort_with_order_raises():
    with pytest.raises(ValueError, match="sort=True cannot be combined"):
        build_nodes(_flat(), ["kind"], "n", order={"kind": ["B", "A"]}, sort=True)


def test_nan_values_are_dropped():
    df = pd.DataFrame({"kind": list("ABC"), "n": [1.0, float("nan"), 3.0]})
    nodes = build_nodes(df, ["kind"], "n")
    assert [n.path[0] for n in nodes] == ["A", "C"]


def test_zero_valued_categories_are_dropped():
    df = pd.DataFrame({"kind": list("ABC"), "n": [1.0, 0.0, 3.0]})
    nodes = build_nodes(df, ["kind"], "n")
    assert [n.path[0] for n in nodes] == ["A", "C"]


def test_negative_values_raise():
    df = pd.DataFrame({"kind": list("AB"), "n": [1.0, -2.0]})
    with pytest.raises(ValueError, match="negative"):
        build_nodes(df, ["kind"], "n")


def test_total_value_sums_the_root_ring_only():
    nodes = build_nodes(_nested(), ["k", "s"], "n")
    assert total_value(nodes) == pytest.approx(60.0)


def test_layout_spans_sum_to_360_per_ring():
    nodes = build_nodes(_nested(), ["k", "s"], "n")
    places = layout(nodes, n_levels=2)
    for depth in (0, 1):
        spans = [p.theta1 - p.theta0 for p in places if p.node.depth == depth]
        assert sum(spans) == pytest.approx(360.0)


def test_layout_starts_at_twelve_oclock_going_clockwise():
    nodes = build_nodes(_flat(), ["kind"], "n")
    places = layout(nodes, start_angle=90.0, direction="clockwise", n_levels=1)
    # A is 50% of 100, so it runs from 90 degrees clockwise to -90 degrees.
    a = places[0]
    assert a.theta1 == pytest.approx(90.0)
    assert a.theta0 == pytest.approx(-90.0)


def test_layout_respects_a_non_default_start_angle():
    nodes = build_nodes(_flat(), ["kind"], "n")
    places = layout(nodes, start_angle=0.0, direction="clockwise", n_levels=1)
    # A is 50% of 100, so it runs from 0 degrees clockwise to -180 degrees.
    a = places[0]
    assert a.theta1 == pytest.approx(0.0)
    assert a.theta0 == pytest.approx(-180.0)


def test_layout_counterclockwise_reverses_the_sweep():
    nodes = build_nodes(_flat(), ["kind"], "n")
    places = layout(nodes, start_angle=90.0, direction="counterclockwise",
                    n_levels=1)
    a = places[0]
    assert a.theta0 == pytest.approx(90.0)
    assert a.theta1 == pytest.approx(270.0)


def test_layout_theta0_is_always_below_theta1():
    nodes = build_nodes(_nested(), ["k", "s"], "n")
    for p in layout(nodes, n_levels=2):
        assert p.theta0 < p.theta1


def test_donut_sets_the_inner_radius_of_the_first_ring():
    nodes = build_nodes(_flat(), ["kind"], "n")
    places = layout(nodes, donut=0.4, n_levels=1)
    assert places[0].r_inner == pytest.approx(0.4)
    assert places[0].r_outer == pytest.approx(1.0)


def test_rings_divide_the_radial_span_evenly():
    nodes = build_nodes(_nested(), ["k", "s"], "n")
    places = layout(nodes, donut=0.0, n_levels=2)
    inner = next(p for p in places if p.node.depth == 0)
    outer = next(p for p in places if p.node.depth == 1)
    assert (inner.r_inner, inner.r_outer) == pytest.approx((0.0, 0.5))
    assert (outer.r_inner, outer.r_outer) == pytest.approx((0.5, 1.0))


def test_child_angles_nest_inside_the_parent():
    nodes = build_nodes(_nested(), ["k", "s"], "n")
    places = {p.node.path: p for p in layout(nodes, n_levels=2)}
    for path, child in places.items():
        if child.node.parent is None:
            continue
        parent = places[child.node.parent]
        assert child.theta0 >= parent.theta0 - 1e-9
        assert child.theta1 <= parent.theta1 + 1e-9


def test_unknown_direction_raises():
    nodes = build_nodes(_flat(), ["kind"], "n")
    with pytest.raises(ValueError, match="direction"):
        layout(nodes, direction="widdershins", n_levels=1)


def test_build_nodes_names_its_owner_in_errors():
    import pandas as pd
    import pytest
    from publiplots.plot.wedge.geometry import build_nodes
    frame = pd.DataFrame({"k": ["a", "b"], "v": [1.0, -1.0]})
    with pytest.raises(ValueError, match=r"^sunburstplot: negative values"):
        build_nodes(frame, ["k"], "v", owner="sunburstplot")
    with pytest.raises(ValueError, match=r"^pieplot: negative values"):
        build_nodes(frame, ["k"], "v")


def _by_path(nodes):
    return {n.path: n for n in nodes}


def test_ragged_rows_end_their_branch_and_still_count():
    df = pd.DataFrame({
        "a": ["P", "P", "P", "Q", "Q"],
        "b": ["p1", "p1", None, "q1", "q2"],
        "c": ["x", "y", None, "u", None],
        "n": [10.0, 5.0, 15.0, 20.0, 40.0],
    })
    nodes = build_nodes(df, ["a", "b", "c"], "n")
    by = _by_path(nodes)
    assert by[("P",)].value == 30.0 and by[("P",)].own_value == 15.0
    assert by[("Q",)].value == 60.0 and by[("Q",)].own_value == 0.0
    assert by[("Q", "q2")].value == 40.0 and by[("Q", "q2")].own_value == 40.0
    assert ("Q", "q2") not in has_children(nodes)
    assert ("P",) in has_children(nodes)
    # Children are listed first in draw order; P's own share is not a node.
    assert [n.path for n in nodes if n.parent == ("P",)] == [("P", "p1")]


def test_a_skipped_ring_raises_naming_the_row():
    df = pd.DataFrame({"a": ["P"], "b": [None], "c": ["x"], "n": [1.0]},
                      index=["r7"])
    with pytest.raises(ValueError, match=r"^sunburstplot: .*'r7'"):
        build_nodes(df, ["a", "b", "c"], "n", owner="sunburstplot")


def test_categorical_columns_create_only_observed_nodes():
    cats = [f"k{i}" for i in range(200)]
    df = pd.DataFrame({
        "a": pd.Categorical(["A", "B"], categories=["A", "B", "C"]),
        "b": pd.Categorical(["k1", "k2"], categories=cats),
        "n": [1.0, 2.0],
    })
    nodes = build_nodes(df, ["a", "b"], "n")
    assert [n.path for n in nodes] == [("A",), ("A", "k1"), ("B",), ("B", "k2")]


def test_one_level_matches_the_previous_walk_exactly():
    df = pd.DataFrame({"k": pd.Categorical(list("dcab"),
                                           categories=list("abcd")),
                       "v": [4.0, 3.0, 1.0, 2.0]})
    for kw in [{}, {"sort": True}, {"order": {"k": list("bdac")}}]:
        nodes = build_nodes(df, ["k"], "v", **kw)
        assert all(n.own_value == n.value for n in nodes)
    assert [n.path[0] for n in build_nodes(df, ["k"], "v")] == list("abcd")


def test_build_nodes_is_fast_on_large_categorical_trees():
    import time
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": pd.Categorical(rng.choice([f"a{i}" for i in range(10)], 100_000)),
        "b": pd.Categorical(rng.choice([f"b{i}" for i in range(100)], 100_000)),
        "c": pd.Categorical(rng.choice([f"c{i}" for i in range(1000)], 100_000)),
        "n": rng.random(100_000),
    })
    t0 = time.perf_counter()
    build_nodes(df, ["a", "b", "c"], "n")
    elapsed = time.perf_counter() - t0
    print(f"build_nodes elapsed: {elapsed:.3f}s")
    assert elapsed < 10.0
