import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import publiplots as pp

MM = 25.4


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# Three rings, full depth, 2 + 4 + 6 nodes.
TREE = pd.DataFrame({
    "a": ["P", "P", "P", "Q", "Q", "Q"],
    "b": ["p1", "p1", "p2", "q1", "q1", "q2"],
    "c": ["x", "y", "z", "u", "v", "w"],
    "n": [10.0, 5.0, 15.0, 20.0, 10.0, 40.0],
})

# Ragged: Q's "q2" row ends at ring 1; P is a parent-leaf with a row of its
# own.
RAGGED = pd.DataFrame({
    "a": ["P", "P", "P", "Q", "Q"],
    "b": ["p1", "p1", None, "q1", "q2"],
    "c": ["x", "y", None, "u", None],
    "n": [10.0, 5.0, 15.0, 20.0, 40.0],
})


def _mm_per_unit(ax):
    ax.apply_aspect()
    inv = ax.transData.inverted()
    y0 = inv.transform((0.0, 0.0))[1]
    y1 = inv.transform((0.0, ax.figure.dpi / MM))[1]   # 1 mm in pixels
    return 1.0 / abs(y1 - y0)


from publiplots.plot.wedge.render import render_wedge_chart


def _draw(df, levels, **overrides):
    fig, ax = pp.subplots(axes_size=(50, 50))
    kwargs = dict(
        owner="sunburstplot", source_frame=df, y="n", hue=None,
        order_map=None, sort=False, hue_order=None, start_angle=90.0,
        direction="clockwise", donut=0.0, pull=None, gap=0.0, palette=None,
        hatch=None, hatch_map=None, hatch_order=None, color="#5d83c3",
        edgecolor=None, linewidth=0.75, alpha=0.1, center_label=None,
        aspect="cell", legend=False, legend_kws=None, title="",
        depth_range=0.45, sibling_range=0.14,
    )
    kwargs.update(overrides)
    meta = render_wedge_chart(ax, df, levels, **kwargs)
    fig.canvas.draw()
    return ax, meta


def _radial_gaps_mm(ax, meta):
    """Gap between each wedge's rim and each child's inner edge, in mm."""
    mm = _mm_per_unit(ax)
    by = {r.path: r for r in meta.wedges}
    gaps = []
    for r in meta.wedges:
        if r.depth > 0:
            parent = by[r.path[:-1]]
            gaps.append((r.r_inner - parent.r_outer) * mm)
    return gaps


def test_explode_puts_ring_k_at_its_stated_radii():
    ax, meta = _draw(TREE, ["a", "b", "c"], pull=0.05)
    t = 0.05 * 1.0
    w = 1.0 / 3
    for r in meta.wedges:
        k = r.depth
        assert r.r_inner == pytest.approx((2 * k + 1) * t + k * w)
        assert r.r_outer == pytest.approx((2 * k + 1) * t + (k + 1) * w)


@pytest.mark.parametrize("gap", [0.5, 1.0])
def test_gap_separates_rings_by_gap_mm(gap):
    ax, meta = _draw(TREE, ["a", "b", "c"], gap=gap)
    assert _radial_gaps_mm(ax, meta) == pytest.approx(
        [gap] * len(_radial_gaps_mm(ax, meta)), abs=0.01)


def test_explode_and_gap_open_every_ring_boundary_to_two_t_plus_gap():
    ax, meta = _draw(TREE, ["a", "b", "c"], pull=0.05, gap=1.0)
    gaps = _radial_gaps_mm(ax, meta)
    two_t = 2 * 0.05 * _mm_per_unit(ax)
    assert gaps == pytest.approx([1.0 + two_t] * len(gaps), abs=0.01)


def test_a_ragged_leaf_rim_is_not_inset():
    ax, meta = _draw(RAGGED, ["a", "b", "c"], gap=1.0)
    by = {r.path: r for r in meta.wedges}
    w = 1.0 / 3
    assert by[("Q", "q2")].r_outer == pytest.approx(2 * w)   # not moved in
    assert by[("P",)].r_outer < w                            # parent-leaf: inset


def test_seams_are_collinear_through_rings_even_where_capped():
    # A 1% sliver caps its side of the ray; its parent must share the inset.
    df = pd.DataFrame({"a": ["P", "P", "Q"], "b": ["s", "t", "u"],
                       "n": [1.0, 49.0, 50.0]})
    ax, meta = _draw(df, ["a", "b"], gap=2.0)
    by = {r.path: r for r in meta.wedges}
    child, parent = by[("P", "s")], by[("P",)]
    # Same side of the shared ray at P's leading edge.
    assert child.theta1 == pytest.approx(parent.theta1)
    assert (child.patch.get_drawn_insets()[1]
            == pytest.approx(parent.patch.get_drawn_insets()[1]))


# --- Beyond the brief ------------------------------------------------------

@pytest.mark.parametrize("a, b", [
    (-1e-13, 360.0 - 1e-13),
    (-1e-13, 0.0),
    (0.0, 360.0),
    (720.0, -360.0),
    (1e-13, -1e-13),
    (90.0 + 1e-12, -270.0 - 1e-12),
])
def test_a_ray_key_folds_whole_turns_and_float_noise(a, b):
    from publiplots.plot.wedge.draw import ray_key
    assert ray_key(a) == ray_key(b)
    assert 0.0 <= ray_key(a) < 360.0


def test_a_ray_key_keeps_distinct_rays_apart():
    from publiplots.plot.wedge.draw import ray_key
    assert ray_key(10.0) != ray_key(10.001)
    assert ray_key(359.999) != ray_key(0.0)


def _offset_from_ray(patch, angle):
    """Smallest perpendicular distance from the ray at ``angle`` (degrees)
    of the patch's drawn vertices lying along that ray, in data units."""
    v = patch.get_path().vertices - np.asarray(patch.center)
    d = np.array([math.cos(math.radians(angle)),
                  math.sin(math.radians(angle))])
    along = v @ d
    across = np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])
    return float(across[along > 0.05].min())


@pytest.mark.parametrize("gap", [0.0, 1.0])
def test_an_exploded_sliver_and_its_parent_share_the_capped_seam(gap):
    # Under explode the edge inset is min(h + t, cap): t on top of the cap
    # would re-cap the sliver on its own and step the seam at its parent.
    # With no gap at all the explode alone still needs the shared cap.
    df = pd.DataFrame({"a": ["P", "P", "Q"], "b": ["s", "t", "u"],
                       "n": [1.0, 49.0, 50.0]})
    ax, meta = _draw(df, ["a", "b"], pull=0.05, gap=gap)
    by = {r.path: r for r in meta.wedges}
    child, parent = by[("P", "s")], by[("P",)]
    assert child.theta1 == pytest.approx(parent.theta1)
    cap = child.patch.r * math.sin(math.radians(child.theta1
                                                - child.theta0) / 4.0)
    h = (gap / 2.0) / _mm_per_unit(ax)
    t = 0.05
    assert cap < h + t                       # the sliver really is capped
    c1, p1 = child.patch.get_drawn_insets()[1], parent.patch.get_drawn_insets()[1]
    assert c1 == pytest.approx(cap)
    assert p1 == pytest.approx(cap)
    # On the drawn outlines: both edges run at the same offset from the ray.
    assert (_offset_from_ray(child.patch, child.theta1)
            == pytest.approx(_offset_from_ray(parent.patch, parent.theta1)))
    assert _offset_from_ray(child.patch, child.theta1) == pytest.approx(cap)
    # The parent's other edge is uncapped, at h + t, and so is the far
    # side of the capped ray: Q and u, whose edge there sits at -270.
    assert parent.patch.get_drawn_insets()[0] == pytest.approx(h + t)
    for path in [("Q",), ("Q", "u")]:
        assert by[path].theta0 == pytest.approx(-270.0)
        assert by[path].patch.get_drawn_insets()[0] == pytest.approx(h + t)


def test_a_thin_parent_caps_its_seam_from_its_inset_rim():
    # P's rim moves in for the ring gap; the cap must be read from the rim
    # as drawn, or P's own cap re-bites below the child's inset.
    df = pd.DataFrame({"a": ["P", "Q"], "b": ["s", "u"], "n": [1.0, 99.0]})
    ax, meta = _draw(df, ["a", "b"], gap=2.0)
    by = {r.path: r for r in meta.wedges}
    child, parent = by[("P", "s")], by[("P",)]
    assert parent.r_outer < 0.5                        # rim inset
    cap = parent.r_outer * math.sin(math.radians(3.6) / 4.0)
    assert cap < 1.0 / _mm_per_unit(ax)                # and capped
    assert parent.patch.get_drawn_insets() == pytest.approx((cap, cap))
    assert child.patch.get_drawn_insets() == pytest.approx((cap, cap))
    for record, angle in [(child, child.theta0), (child, child.theta1),
                          (parent, parent.theta0), (parent, parent.theta1)]:
        assert _offset_from_ray(record.patch, angle) == pytest.approx(cap)


def test_ray_side_caps_count_only_wedges_of_at_most_180_degrees():
    from publiplots.plot.wedge.draw import GappedWedge, ray_side_caps
    thin = GappedWedge((0, 0), 1.0, 0.0, 20.0, width=0.5)
    half_disc = GappedWedge((0, 0), 0.8, 20.0, 200.0, width=0.5)
    wedge160 = GappedWedge((0, 0), 0.5, 200.0, 360.0, width=0.5)
    reflex = GappedWedge((0, 0), 0.4, -160.0, 20.001, width=0.4)
    full = GappedWedge((0, 0), 0.3, 0.0, 360.0, width=0.3)
    caps = ray_side_caps([thin, half_disc, wedge160, full])
    assert caps[(0.0, 1)] == pytest.approx(1.0 * math.sin(math.radians(5)))
    assert caps[(20.0, -1)] == pytest.approx(1.0 * math.sin(math.radians(5)))
    assert caps[(20.0, 1)] == pytest.approx(0.8 * math.sin(math.radians(45)))
    assert caps[(200.0, -1)] == pytest.approx(0.8 * math.sin(math.radians(45)))
    # The 160-degree wedge from 200 to 360 counts; the full circle does not.
    assert caps[(200.0, 1)] == pytest.approx(0.5 * math.sin(math.radians(40)))
    assert caps[(0.0, -1)] == pytest.approx(0.5 * math.sin(math.radians(40)))
    # Just past 180 degrees a wedge no longer caps its sides.
    caps = ray_side_caps([reflex, full])
    assert caps[(200.0, 1)] == math.inf
    assert caps[(20.001, -1)] == math.inf
    assert caps[(0.0, 1)] == math.inf


def test_an_uncapped_seam_is_half_the_gap_plus_the_explode():
    ax, meta = _draw(TREE, ["a", "b", "c"], pull=0.05, gap=1.0)
    h = 0.5 / _mm_per_unit(ax)
    for r in meta.wedges:
        assert r.patch.get_drawn_insets() == pytest.approx((h + 0.05,
                                                            h + 0.05))


def test_a_band_is_never_radially_inset_past_half_its_width():
    # gap/2 far exceeds w/2: one-sided bands keep w/2, two-sided keep w/2
    # between w/4 insets.
    fig, ax = pp.subplots(axes_size=(20, 20))
    meta = render_wedge_chart(
        ax, TREE, ["a", "b", "c"], owner="sunburstplot", source_frame=TREE,
        y="n", hue=None, order_map=None, sort=False, hue_order=None,
        start_angle=90.0, direction="clockwise", donut=0.0, pull=None,
        gap=10.0, palette=None, hatch=None, hatch_map=None, hatch_order=None,
        color="#5d83c3", edgecolor=None, linewidth=0.75, alpha=0.1,
        center_label=None, aspect="cell", legend=False, legend_kws=None,
        title="", depth_range=0.45, sibling_range=0.14)
    assert 5.0 / _mm_per_unit(ax) > 1.0 / 6       # h > w/2
    w = 1.0 / 3
    expected = {0: (0.0, w / 2), 1: (w + w / 4, 2 * w - w / 4),
                2: (2 * w + w / 2, 1.0)}
    for r in meta.wedges:
        assert (r.r_inner, r.r_outer) == pytest.approx(expected[r.depth])


def test_no_gap_and_no_pull_draws_the_plain_rings():
    ax, meta = _draw(TREE, ["a", "b", "c"])
    w = 1.0 / 3
    for r in meta.wedges:
        assert (r.r_inner, r.r_outer) == pytest.approx(
            (r.depth * w, (r.depth + 1) * w))
        assert r.patch.get_drawn_insets() == (0.0, 0.0)


def test_the_frame_leaves_room_for_the_outer_ring_under_explode():
    ax, meta = _draw(TREE, ["a", "b", "c"], pull=0.05)
    rim = max(r.r_outer for r in meta.wedges)
    assert rim == pytest.approx(1.0 + 5 * 0.05)
    assert ax.get_ylim()[1] == pytest.approx(rim + 0.05)


def test_apply_gap_refuses_nodes_that_do_not_match_the_patches():
    from publiplots.plot.wedge.draw import apply_gap
    from publiplots.plot.wedge.geometry import build_nodes
    ax, meta = _draw(TREE, ["a", "b", "c"])
    patches = [r.patch for r in meta.wedges]
    nodes = build_nodes(TREE, ["a", "b", "c"], "n")
    with pytest.raises(AssertionError, match="nodes against"):
        apply_gap(ax, patches[:-1], 1.0, nodes=nodes, n_levels=3)
    with pytest.raises(AssertionError, match="deeper than the 2 rings"):
        apply_gap(ax, patches, 1.0, nodes=nodes, n_levels=2)


def test_pulling_a_parent_moves_its_subtree_rigidly():
    ax, meta = _draw(TREE, ["a", "b", "c"], pull={"P": 0.1})
    by = {r.path: r for r in meta.wedges}
    vec = np.asarray(by[("P",)].patch.center)
    assert np.linalg.norm(vec) == pytest.approx(0.1)
    for path, rec in by.items():
        expected = vec if path[0] == "P" else np.zeros(2)
        assert np.asarray(rec.patch.center) == pytest.approx(expected)


def test_pull_vectors_add_down_the_tree():
    ax, meta = _draw(TREE, ["a", "b", "c"],
                     pull={"P": 0.1, ("P", "p1"): 0.05})
    by = {r.path: r for r in meta.wedges}
    p, p1 = by[("P",)], by[("P", "p1")]
    mid = math.radians((p1.theta0 + p1.theta1) / 2)
    expected = np.asarray(p.patch.center) + 0.05 * np.array([math.cos(mid), math.sin(mid)])
    assert np.asarray(p1.patch.center) == pytest.approx(expected)
    assert np.asarray(by[("P", "p1", "x")].patch.center) == pytest.approx(expected)


@pytest.mark.parametrize("bad,exc", [
    ({"Z": 0.1}, ValueError), ({"P": True}, TypeError),
    ({"P": "x"}, TypeError), ({"P": -0.1}, ValueError),
    ({"P": float("nan")}, ValueError),
])
def test_bad_pull_dicts_raise(bad, exc):
    with pytest.raises(exc, match=r"^sunburstplot: pull="):
        _draw(TREE, ["a", "b", "c"], pull=bad)


# --- Beyond the brief: dict pull ---------------------------------------------

@pytest.mark.parametrize("bad,exc", [
    ({("P", "zz"): 0.1}, ValueError),            # a tuple naming no node
    ({("P", "p1", "x", "deeper"): 0.1}, ValueError),
    ({("P", "p1"): float("inf")}, ValueError),   # checked on tuple keys too
    ({("P", "p1"): None}, TypeError),
])
def test_bad_path_keys_and_values_raise(bad, exc):
    with pytest.raises(exc, match=r"^sunburstplot: pull="):
        _draw(TREE, ["a", "b", "c"], pull=bad)


def test_a_root_key_and_its_one_tuple_name_the_same_wedge():
    _, short = _draw(TREE, ["a", "b", "c"], pull={"Q": 0.1})
    _, long = _draw(TREE, ["a", "b", "c"], pull={("Q",): 0.1})
    assert ([tuple(r.patch.center) for r in short.wedges]
            == [tuple(r.patch.center) for r in long.wedges])
    assert np.linalg.norm(short.wedges[-1].patch.center) == pytest.approx(0.1)


def test_a_key_equal_to_a_tuple_root_category_is_that_category():
    # The root categories are themselves tuples, so ("P", "p1") is one of
    # them -- and also a path prefix it could be mistaken for.
    from publiplots.plot.wedge.geometry import build_nodes, layout
    from publiplots.plot.wedge.render import resolve_pull_vectors
    df = pd.DataFrame({"a": [("P", "p1"), "P"], "b": ["s", "p1"],
                       "n": [1.0, 3.0]})
    nodes = build_nodes(df, ["a", "b"], "n")
    places = layout(nodes, start_angle=90.0, direction="clockwise",
                    donut=0.0, n_levels=2)
    vectors = resolve_pull_vectors({("P", "p1"): 0.1}, nodes, places,
                                   owner="sunburstplot")
    assert np.hypot(*vectors[(("P", "p1"),)]) == pytest.approx(0.1)
    assert vectors[("P", "p1")] == (0.0, 0.0)


def test_records_keep_the_pull_asked_at_each_node():
    _, meta = _draw(TREE, ["a", "b", "c"], pull={"P": 0.1, ("P", "p1"): 0.05})
    asked = {r.path: r.pull for r in meta.wedges}
    assert asked[("P",)] == 0.1
    assert asked[("P", "p1")] == 0.05
    assert all(v == 0.0 for k, v in asked.items()
               if k not in {("P",), ("P", "p1")})


def test_the_frame_leaves_room_for_the_summed_pull():
    # Two pulls in nearly the same direction: the reach is their sum, not
    # the largest of them.
    ax, meta = _draw(TREE, ["a", "b", "c"], pull={"P": 0.1, ("P", "p1"): 0.05})
    reach = max(np.hypot(*r.patch.center) for r in meta.wedges)
    assert reach > 0.1 + 0.04
    assert ax.get_ylim()[1] == pytest.approx(1.0 + reach + 0.05)


def test_a_rigidly_pulled_subtree_keeps_its_capped_seam_under_gap():
    # Per-ray caps stay grouped by angle and side under a dict pull: P and
    # its sliver s move together, so their shared seam stays one line.
    df = pd.DataFrame({"a": ["P", "P", "Q"], "b": ["s", "t", "u"],
                       "n": [1.0, 49.0, 50.0]})
    ax, meta = _draw(df, ["a", "b"], pull={"P": 0.1}, gap=2.0)
    by = {r.path: r for r in meta.wedges}
    child, parent = by[("P", "s")], by[("P",)]
    assert child.theta1 == pytest.approx(parent.theta1)
    assert np.asarray(child.patch.center) == pytest.approx(
        np.asarray(parent.patch.center))
    assert np.hypot(*parent.patch.center) == pytest.approx(0.1)
    cap = child.patch.r * math.sin(math.radians(child.theta1
                                                - child.theta0) / 4.0)
    assert cap < 1.0 / _mm_per_unit(ax)                # really capped
    assert child.patch.get_drawn_insets()[1] == pytest.approx(cap)
    assert parent.patch.get_drawn_insets()[1] == pytest.approx(cap)
    assert (_offset_from_ray(child.patch, child.theta1)
            == pytest.approx(_offset_from_ray(parent.patch, parent.theta1)))


def test_the_rim_blends_depth_range_toward_white():
    from matplotlib.colors import to_rgb
    ax, meta = _draw(TREE, ["a", "b", "c"], palette={"P": "#000000", "Q": "#000000"},
                     sibling_range=0.0)
    for rec in meta.wedges:
        expected = 0.45 * rec.depth / 2
        assert rec.hue_color == pytest.approx((expected,) * 3, abs=1e-9)


def test_siblings_spread_by_sibling_range():
    ax, meta = _draw(TREE, ["a", "b", "c"], palette={"P": "#000000", "Q": "#000000"},
                     depth_range=0.0)
    by = {r.path: r for r in meta.wedges}
    assert by[("P", "p1")].hue_color == pytest.approx((0.0,) * 3)
    assert by[("P", "p2")].hue_color == pytest.approx((0.14,) * 3)


def test_negative_depth_range_darkens():
    ax, meta = _draw(TREE, ["a", "b", "c"], palette={"P": "#ffffff", "Q": "#ffffff"},
                     depth_range=-0.45, sibling_range=0.0)
    rim = [r for r in meta.wedges if r.depth == 2][0]
    assert rim.hue_color == pytest.approx((0.55,) * 3)


def test_hue_must_be_a_root_attribute():
    with pytest.raises(ValueError, match=r"^sunburstplot: hue='b'"):
        _draw(TREE, ["a", "b", "c"], hue="b")
    df = TREE.assign(region=["N", "S", "N", "S", "S", "S"])
    with pytest.raises(ValueError, match="not constant within"):
        _draw(df, ["a", "b", "c"], hue="region")


def test_hatch_on_ring_one_is_inherited_outward_only():
    ax, meta = _draw(TREE, ["a", "b", "c"], hatch="b")
    by = {r.path: r for r in meta.wedges}
    assert by[("P",)].hatch is None
    assert by[("P", "p1")].hatch is not None
    assert by[("P", "p1", "x")].hatch == by[("P", "p1")].hatch


def test_a_ring_hatch_legend_lists_each_value_once_in_draw_order():
    from publiplots.utils.legend_entries import get_entries
    df = pd.DataFrame({
        "a": ["P", "P", "Q", "Q"],
        "b": ["s2", "s1", "s1", "s2"],
        "n": [1.0, 1.0, 1.0, 1.0],
    })
    ax, meta = _draw(df, ["a", "b"], hatch="b", legend=True)
    entry = {e.kind: e for e in get_entries(ax)}["hatch"]
    assert entry.name == "b"
    assert list(entry.labels) == ["s2", "s1"]
    by = {r.path: r for r in meta.wedges}
    assert by[("P", "s1")].hatch == by[("Q", "s1")].hatch
    assert by[("P", "s1")].hatch != by[("P", "s2")].hatch
    assert by[("P",)].hatch is None and by[("Q",)].hatch is None


def test_a_root_attribute_hue_colours_every_ring_of_a_root_alike():
    df = TREE.assign(region=["N", "N", "N", "S", "S", "S"])
    ax, meta = _draw(df, ["a", "b", "c"], hue="region",
                     palette={"N": "#000000", "S": "#ffffff"},
                     depth_range=0.0, sibling_range=0.0)
    for rec in meta.wedges:
        want = 0.0 if rec.path[0] == "P" else 1.0
        assert rec.hue_color == pytest.approx((want,) * 3)


# --- pp.sunburstplot --------------------------------------------------------

def test_one_ring_sunburst_draws_what_pieplot_draws():
    df = TREE.groupby("a", as_index=False)["n"].sum()
    for kw in [{}, {"donut": 0.5}, {"gap": 1.0}, {"pull": 0.05}]:
        _, a = pp.subplots(axes_size=(40, 40))
        pp.pieplot(data=df, x="a", y="n", legend=False, ax=a, **kw)
        _, b = pp.subplots(axes_size=(40, 40))
        pp.sunburstplot(data=df, x="a", y="n", legend=False, ax=b, **kw)
        assert len(a.patches) == len(b.patches) == 2
        for pa, pb in zip(a.patches, b.patches):
            assert pa.get_path().vertices.tolist() == pb.get_path().vertices.tolist()
            assert tuple(pa.get_facecolor()) == tuple(pb.get_facecolor())


def test_spans_close_in_each_full_ring():
    ax = pp.sunburstplot(data=TREE, x=["a", "b", "c"], y="n", legend=False)
    meta = ax._publiplots_wedge_meta
    for depth in range(3):
        total = sum(r.theta1 - r.theta0 for r in meta.wedges if r.depth == depth)
        assert total == pytest.approx(360.0)


def test_a_parent_leafs_remainder_trails_its_children():
    ax = pp.sunburstplot(data=RAGGED, x=["a", "b", "c"], y="n", legend=False)
    by = {r.path: r for r in ax._publiplots_wedge_meta.wedges}
    p, p1 = by[("P",)], by[("P", "p1")]
    # Clockwise from twelve o'clock: the child starts at P's leading edge
    # (theta1, the larger angle) and P's own share is the trailing rest.
    assert p1.theta1 == pytest.approx(p.theta1)
    assert p1.theta1 - p1.theta0 == pytest.approx((p.theta1 - p.theta0) * 15 / 30)


def test_only_root_categories_are_stashed():
    from publiplots.utils.legend_entries import get_entries
    ax = pp.sunburstplot(data=TREE, x=["a", "b", "c"], y="n", legend=False)
    (entry,) = [e for e in get_entries(ax) if e.kind == "hue"]
    assert list(entry.labels) == ["P", "Q"]


@pytest.mark.parametrize("kwargs,exc", [
    ({"min_share": 0.1}, TypeError),
    ({"order": ["P", "Q"]}, TypeError),
    ({"order": {"zz": ["P"]}}, ValueError),
    ({"depth_range": 0.9, "sibling_range": 0.2}, ValueError),
    ({"annotate": True}, NotImplementedError),
    ({"sort": True, "order": {"a": ["P", "Q"]}}, ValueError),
])
def test_sunburstplot_refusals(kwargs, exc):
    with pytest.raises(exc):
        pp.sunburstplot(data=TREE, x=["a", "b", "c"], y="n", **kwargs)


def test_pieplot_names_sunburstplot_for_several_columns():
    with pytest.raises(ValueError, match="sunburstplot"):
        pp.pieplot(data=TREE, x=["a", "b"], y="n")


def test_repeated_saves_are_stable():
    import io, warnings
    fig, axes = pp.subplots(ncols=2, axes_size=(40, 40))
    pp.sunburstplot(data=TREE, x=["a", "b", "c"], y="n", ax=axes[0], gap=1.0)
    pp.sunburstplot(data=RAGGED, x=["a", "b", "c"], y="n", ax=axes[1],
                    pull=0.05, aspect="equal")
    sizes = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", pp.LayoutConvergenceWarning)
        for _ in range(3):
            fig.savefig(io.BytesIO(), format="png")
            sizes.append(tuple(fig.get_size_inches()))
    assert sizes[0] == sizes[1] == sizes[2]


def test_wedge_meta_records_which_function_drew_the_chart():
    df = TREE.groupby("a", as_index=False)["n"].sum()
    pie = pp.pieplot(data=df, x="a", y="n", legend=False)
    sun = pp.sunburstplot(data=df, x="a", y="n", legend=False)
    assert pie._publiplots_wedge_meta.chart == "pie"
    assert sun._publiplots_wedge_meta.chart == "sunburst"


def test_render_wedge_chart_chart_is_explicit_not_derived_from_owner():
    _, meta = _draw(TREE, ["a", "b", "c"], chart="pie")
    assert meta.chart == "pie"
    _, meta = _draw(TREE, ["a", "b", "c"])
    assert meta.chart == "pie"          # the default; owner is not consulted
    _, meta = _draw(TREE, ["a", "b", "c"], chart="sunburst")
    assert meta.chart == "sunburst"


def test_order_omission_warning_points_at_the_callers_line():
    import warnings
    df = TREE.groupby("a", as_index=False)["n"].sum()
    with pytest.warns(UserWarning, match="pieplot: order= omits 1 category") as rec:
        pp.pieplot(data=df, x="a", y="n", order=["P"], legend=False)
    assert [w.filename for w in rec if "omits" in str(w.message)] == [__file__]
    with pytest.warns(UserWarning,
                      match="sunburstplot: order= omits 1 category") as rec:
        pp.sunburstplot(data=df, x="a", y="n", order={"a": ["P"]}, legend=False)
    assert [w.filename for w in rec if "omits" in str(w.message)] == [__file__]


def test_order_omission_warning_covers_every_ordered_column():
    with pytest.warns(UserWarning, match="sunburstplot: order= omits 1 category"
                      ) as rec:
        pp.sunburstplot(data=TREE, x=["a", "b"], y="n", legend=False,
                        order={"a": ["P", "Q"], "b": ["p1", "p2", "q1"]})
    assert "'q2'" in str(rec[0].message)


def test_shading_knobs_reach_the_renderer():
    def colours(**kw):
        ax = pp.sunburstplot(data=TREE, x=["a", "b", "c"], y="n",
                             legend=False, **kw)
        return {r.path: r.hue_color for r in ax._publiplots_wedge_meta.wedges}

    flat = colours(depth_range=0.0, sibling_range=0.0)
    assert flat[("P", "p1", "x")] == flat[("P",)] == flat[("P", "p2")]
    deep = colours(depth_range=0.4, sibling_range=0.0)
    assert deep[("P", "p1", "x")] != deep[("P",)]
    sibs = colours(depth_range=0.0, sibling_range=0.2)
    assert sibs[("P", "p1")] != sibs[("P", "p2")]
    # Left unset, the rcParams defaults (0.45, 0.14) apply.
    default = colours()
    assert default[("P", "p1", "x")] != default[("P",)]
    assert default[("P", "p1")] != default[("P", "p2")]


def test_a_refused_call_draws_no_figure():
    plt.close("all")
    with pytest.raises(ValueError, match="sunburstplot"):
        pp.sunburstplot(data=TREE, x=["a", "b", "c"], y="n",
                        depth_range=0.9, sibling_range=0.2)
    assert plt.get_fignums() == []


def test_render_wedge_chart_refuses_an_unknown_chart_kind():
    with pytest.raises(ValueError, match="chart must be"):
        _draw(TREE, ["a", "b", "c"], chart="donut")
