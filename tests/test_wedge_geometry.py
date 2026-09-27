"""Tree building and angular layout for the wedge renderer."""
from __future__ import annotations

import pandas as pd
import pytest

from publiplots.plot.wedge.geometry import (
    WedgeNode,
    build_nodes,
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
    parent = places[("A",)]
    for child_path in (("A", "x"), ("A", "y")):
        child = places[child_path]
        assert child.theta0 >= parent.theta0 - 1e-9
        assert child.theta1 <= parent.theta1 + 1e-9


def test_unknown_direction_raises():
    nodes = build_nodes(_flat(), ["kind"], "n")
    with pytest.raises(ValueError, match="direction"):
        layout(nodes, direction="widdershins", n_levels=1)
