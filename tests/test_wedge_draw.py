"""Wedge artist construction, framing and meta emission."""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.patches import Wedge

import publiplots as pp
from publiplots.plot.wedge.draw import (
    apply_cell_limits,
    build_wedge_meta,
    draw_wedges,
    pull_offset,
)
from publiplots.plot.wedge.geometry import build_nodes, layout, total_value


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _places():
    df = pd.DataFrame({"k": list("AB"), "n": [3.0, 1.0]})
    nodes = build_nodes(df, ["k"], "n")
    return nodes, layout(nodes, n_levels=1), total_value(nodes)


def _places_donut():
    df = pd.DataFrame({"k": list("AB"), "n": [3.0, 1.0]})
    nodes = build_nodes(df, ["k"], "n")
    return nodes, layout(nodes, n_levels=1, donut=0.3), total_value(nodes)


def _places_multilevel():
    """Two levels, six nodes, whose values are deliberately NOT in
    descending order at either level -- appearance order is A (small)
    before B (large), and each parent's own children are ascending. This
    means any reordering by value or by depth produces a draw order
    different from the one `build_nodes`/`layout` actually assigned, so a
    pairing test against this fixture catches mispairings that a
    two-node, already-descending fixture would not.
    """
    df = pd.DataFrame({
        "level1": ["A", "A", "B", "B"],
        "level2": ["A1", "A2", "B1", "B2"],
        "n": [0.3, 0.7, 2.0, 3.0],
    })
    nodes = build_nodes(df, ["level1", "level2"], "n")
    return nodes, layout(nodes, n_levels=2), total_value(nodes)


def _style(nodes):
    colors = {n.path: (0.5, 0.5, 0.5) for n in nodes}
    hatches = {n.path: None for n in nodes}
    return colors, hatches


def test_pull_offset_is_zero_without_pull():
    _, places, _ = _places()
    assert pull_offset(places[0], 0.0) == (0.0, 0.0)


def test_pull_offset_points_along_the_mid_angle():
    _, places, _ = _places()
    dx, dy = pull_offset(places[0], 0.5)
    mid = math.radians((places[0].theta0 + places[0].theta1) / 2.0)
    assert (dx, dy) == pytest.approx((0.5 * math.cos(mid),
                                      0.5 * math.sin(mid)))


def test_draw_wedges_adds_one_patch_per_placement():
    nodes, places, _ = _places()
    colors, hatches = _style(nodes)
    fig, ax = pp.subplots()
    patches = draw_wedges(ax, places, colors, hatches, center=(0.0, 0.0),
                          pull={}, alpha=1.0, edgecolor="black",
                          linewidth=0.75)
    assert len(patches) == 2
    assert all(isinstance(p, Wedge) for p in patches)
    assert [p for p in ax.patches if isinstance(p, Wedge)] == patches


def test_build_meta_pairs_each_placement_with_its_own_patch():
    nodes, places, total = _places()
    colors, hatches = _style(nodes)
    fig, ax = pp.subplots()
    patches = draw_wedges(ax, places, colors, hatches, center=(0.0, 0.0),
                          pull={}, alpha=1.0, edgecolor="black",
                          linewidth=0.75)
    meta = build_wedge_meta(places, patches, colors, hatches, {},
                            total=total, donut=0.0, center=(0.0, 0.0),
                            levels=["k"], hue_active=True,
                            hue_continuous=False, source_frame=None)
    for record, place, patch in zip(meta.wedges, places, patches):
        assert record.patch is patch
        assert record.path == place.node.path
        assert record.theta0 == pytest.approx(place.theta0)


def test_build_meta_pairs_placements_in_order_when_values_are_not_sorted():
    nodes, places, total = _places_multilevel()
    colors, hatches = _style(nodes)
    fig, ax = pp.subplots()
    patches = draw_wedges(ax, places, colors, hatches, center=(0.0, 0.0),
                          pull={}, alpha=1.0, edgecolor="black",
                          linewidth=0.75)
    meta = build_wedge_meta(places, patches, colors, hatches, {},
                            total=total, donut=0.0, center=(0.0, 0.0),
                            levels=["level1", "level2"], hue_active=True,
                            hue_continuous=False, source_frame=None)
    # The fixture's values are ascending within each parent and the parent
    # totals themselves are ascending in appearance order (A=1.0 before
    # B=5.0), so a pairing that silently re-sorted by value or by depth
    # would not land back on the identity permutation here.
    assert [r.path for r in meta.wedges] == [p.node.path for p in places]
    for record, patch in zip(meta.wedges, patches):
        assert record.patch is patch


def test_draw_wedges_applies_pull_offset_to_patch_center():
    nodes, places, _ = _places()
    colors, hatches = _style(nodes)
    fig, ax = pp.subplots()
    pull = {places[0].node.path: 0.4}
    patches = draw_wedges(ax, places, colors, hatches, center=(0.0, 0.0),
                          pull=pull, alpha=1.0, edgecolor="black",
                          linewidth=0.75)
    expected = pull_offset(places[0], 0.4)
    assert patches[0].center == pytest.approx(expected)
    # The other wedge has no pull entry, so it stays at the given center.
    assert patches[1].center == pytest.approx((0.0, 0.0))


def test_draw_wedges_applies_style_kwargs_and_honors_ring_width():
    nodes, places, _ = _places_donut()
    colors = {n.path: (0.1, 0.2, 0.3) for n in nodes}
    hatches = {places[0].node.path: "//", places[1].node.path: None}
    fig, ax = pp.subplots()
    patches = draw_wedges(ax, places, colors, hatches, center=(0.0, 0.0),
                          pull={}, alpha=0.42, edgecolor=(0.9, 0.1, 0.1, 1.0),
                          linewidth=2.5)
    for place, patch in zip(places, patches):
        assert patch.get_facecolor()[:3] == pytest.approx(
            colors[place.node.path])
        assert patch.get_alpha() == pytest.approx(0.42)
        assert patch.get_linewidth() == pytest.approx(2.5)
        assert patch.get_edgecolor()[:3] == pytest.approx((0.9, 0.1, 0.1))
        # donut=0.3 means r_inner > 0: a ring, not a solid disc.
        assert patch.width == pytest.approx(place.r_outer - place.r_inner)
        assert patch.width < place.r_outer
    assert patches[0].get_hatch() == "//"
    assert patches[1].get_hatch() is None


def test_build_meta_raises_when_an_ancestor_is_missing():
    nodes, places, total = _places_multilevel()
    colors, hatches = _style(nodes)
    fig, ax = pp.subplots()
    patches = draw_wedges(ax, places, colors, hatches, center=(0.0, 0.0),
                          pull={}, alpha=1.0, edgecolor="black",
                          linewidth=0.75)
    # Drop the depth-0 rings (A, B), keeping only their children -- each
    # remaining node's parent is now absent from the placements passed in.
    leaf_places = [p for p in places if p.node.depth == 1]
    leaf_patches = [patch for p, patch in zip(places, patches)
                     if p.node.depth == 1]
    with pytest.raises(ValueError, match="parent"):
        build_wedge_meta(leaf_places, leaf_patches, colors, hatches, {},
                         total=total, donut=0.0, center=(0.0, 0.0),
                         levels=["level1", "level2"], hue_active=True,
                         hue_continuous=False, source_frame=None)


def test_build_meta_reports_fractions():
    nodes, places, total = _places()
    colors, hatches = _style(nodes)
    fig, ax = pp.subplots()
    patches = draw_wedges(ax, places, colors, hatches, center=(0.0, 0.0),
                          pull={}, alpha=1.0, edgecolor="black",
                          linewidth=0.75)
    meta = build_wedge_meta(places, patches, colors, hatches, {},
                            total=total, donut=0.0, center=(0.0, 0.0),
                            levels=["k"], hue_active=True,
                            hue_continuous=False, source_frame=None)
    assert [r.fraction_of_total for r in meta.wedges] == pytest.approx(
        [0.75, 0.25])
    # At depth 0 the parent is the whole chart, so the two agree.
    assert [r.fraction_of_parent for r in meta.wedges] == pytest.approx(
        [0.75, 0.25])


def test_build_meta_refuses_a_length_mismatch():
    nodes, places, total = _places()
    colors, hatches = _style(nodes)
    fig, ax = pp.subplots()
    patches = draw_wedges(ax, places, colors, hatches, center=(0.0, 0.0),
                          pull={}, alpha=1.0, edgecolor="black",
                          linewidth=0.75)
    with pytest.raises(AssertionError, match="placements"):
        build_wedge_meta(places, patches[:1], colors, hatches, {},
                         total=total, donut=0.0, center=(0.0, 0.0),
                         levels=["k"], hue_active=True,
                         hue_continuous=False, source_frame=None)


def test_apply_cell_limits_makes_a_non_square_cell_render_round():
    fig, ax = pp.subplots(axes_size=(70, 50))
    apply_cell_limits(ax, radius=1.0, aspect="cell")
    fig.canvas.draw()
    (x0, y0), (x1, y1) = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
    assert (x1 - x0) == pytest.approx(y1 - y0, rel=1e-6)


def test_apply_cell_limits_keeps_the_allocated_width():
    fig, ax = pp.subplots(axes_size=(70, 50))
    apply_cell_limits(ax, radius=1.0, aspect="cell")
    fig.canvas.draw()
    assert ax.get_position().width == pytest.approx(
        ax.get_position(original=True).width, rel=1e-9)


def test_apply_cell_limits_equal_uses_matplotlib_aspect():
    fig, ax = pp.subplots(axes_size=(70, 50))
    apply_cell_limits(ax, radius=1.0, aspect="equal")
    assert ax.get_aspect() == 1.0


def test_apply_cell_limits_rejects_an_unknown_aspect():
    fig, ax = pp.subplots()
    with pytest.raises(ValueError, match="aspect"):
        apply_cell_limits(ax, radius=1.0, aspect="square")
