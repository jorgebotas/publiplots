"""Crowded outside wedge labels are moved apart: ``avoid_overlap``.

Every geometry check reads the rendered figure after a full ``savefig``,
in millimetres, so it sees what the page shows once the layout has
settled -- not the data-unit arithmetic the strategy did.
"""
from __future__ import annotations

import io
import itertools
import math
import random

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp
from publiplots.annotate import _wedge


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def restore_pie_rcparams():
    keys = ["pie.avoid_overlap", "pie.label_spacing", "pie.leader"]
    saved = {key: pp.rcParams[key] for key in keys}
    yield
    for key, value in saved.items():
        pp.rcParams[key] = value


TOL_MM = 0.01
MM_PER_INCH = 25.4
LEADERS = ["radial", "elbow", None]

# The long tail the pass exists for: four slivers beside one another.
TAIL = pd.DataFrame({"kind": list("ABCDEFGH"),
                     "n": [40.0, 30.0, 20.0, 5.0, 3.0, 1.0, 0.5, 0.5]})
# The wide wedges, whose labels crowd nothing.
WIDE = ("A", "B", "C")
# Nothing crowded at all.
ROOMY = pd.DataFrame({"kind": list("ABCD"), "n": [50.0, 25.0, 15.0, 10.0]})

# (start_angle, direction): the slivers at twelve, three, a little past
# seven, and the other way round.
TURNS = [(90, "clockwise"), (0, "clockwise"), (200, "clockwise"),
         (90, "counterclockwise"), (0, "counterclockwise"),
         (200, "counterclockwise")]

GEOMETRIES = {
    "donut": {"donut": 0.55},
    "pull_dict": {"pull": {"F": 0.2, "D": 0.1}},
    "pull_scalar": {"pull": 0.05},
    "gap": {"gap": 1.0},
}


def _settle(fig):
    fig.canvas.draw()
    for ax in fig.axes:
        ax.apply_aspect()
    return fig.canvas.get_renderer()


def _mm(ax, px):
    return px * MM_PER_INCH / ax.figure.dpi


def _pie(frame=TAIL, size=40.0, pie_kws=None, **annotate_kws):
    """A pie with every label outside unless asked otherwise, saved."""
    fig, ax = pp.subplots(axes_size=(size, size))
    pp.pieplot(data=frame, x="kind", y="n", legend=False, ax=ax,
               **(pie_kws or {}))
    annotate_kws.setdefault("anchor", "outside")
    pp.annotate(ax, "wedge_values", **annotate_kws)
    fig.savefig(io.BytesIO(), format="png")
    return ax, _settle(fig)


def _boxes(ax, renderer):
    return [tuple(text.get_window_extent(renderer).extents)
            for text in ax.texts]


def _gap(a, b):
    """Straight-line gap between two display boxes; minus the overlap."""
    gx = max(a[0] - b[2], b[0] - a[2])
    gy = max(a[1] - b[3], b[1] - a[3])
    if gx < 0 and gy < 0:
        return max(gx, gy)
    return math.hypot(max(gx, 0.0), max(gy, 0.0))


def _min_gap_mm(ax, renderer):
    boxes = _boxes(ax, renderer)
    return min(_mm(ax, _gap(a, b))
               for a, b in itertools.combinations(boxes, 2))


def _leaders(ax):
    return [ax.transData.transform(list(zip(line.get_xdata(),
                                            line.get_ydata())))
            for line in ax.lines]


def _point_box(p, box):
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return math.hypot(dx, dy)


def _segment_box(a, b, box):
    """Display distance from segment ``a``-``b`` to a box, by sampling.

    Sampled finely rather than solved, so it checks the strategy's own
    closed-form geometry independently.
    """
    return min(_point_box((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])),
                          box)
               for t in (i / 4000.0 for i in range(4001)))


def _crosses(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (orient(q1, q2, p1) * orient(q1, q2, p2) < 0
            and orient(p1, p2, q1) * orient(p1, p2, q2) < 0)


def _rim_display(ax, record):
    cx, cy = record.patch.center
    mid = math.radians((record.theta0 + record.theta1) / 2.0)
    return ax.transData.transform((cx + record.r_outer * math.cos(mid),
                                   cy + record.r_outer * math.sin(mid)))


# -----------------------------------------------------------------------------
# No overlap, and label_spacing kept
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("size", [25.0, 40.0, 80.0])
@pytest.mark.parametrize("start, direction", TURNS)
@pytest.mark.parametrize("leader", LEADERS)
def test_no_two_outside_labels_come_within_label_spacing(
    leader, start, direction, size,
):
    """Every pair, not only neighbours, measured on the rendered boxes."""
    ax, renderer = _pie(size=size, leader=leader,
                        pie_kws={"start_angle": start,
                                 "direction": direction})
    assert _min_gap_mm(ax, renderer) >= pp.rcParams["pie.label_spacing"] \
        - TOL_MM


@pytest.mark.parametrize("size", [25.0, 80.0])
@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
@pytest.mark.parametrize("start", [90, 200])
@pytest.mark.parametrize("leader", LEADERS)
def test_label_spacing_holds_on_a_donut_with_pull_and_gap(
    leader, start, geometry, size,
):
    ax, renderer = _pie(size=size, leader=leader,
                        pie_kws={"start_angle": start,
                                 **GEOMETRIES[geometry]})
    assert _min_gap_mm(ax, renderer) >= pp.rcParams["pie.label_spacing"] \
        - TOL_MM


def test_the_long_tail_overlaps_without_the_pass():
    """The case is a real one: unmoved, the slivers' labels overlap."""
    for leader in LEADERS:
        ax, renderer = _pie(leader=leader, avoid_overlap=False)
        assert _min_gap_mm(ax, renderer) < 0.0, leader


@pytest.mark.parametrize("spacing", [0.0, 1.5])
@pytest.mark.parametrize("leader", LEADERS)
def test_label_spacing_is_the_least_gap_kept(leader, spacing):
    """The gap kept is the one asked for, and is reached: the pass moves
    labels no further apart than it must, so the closest pair sits at
    ``label_spacing`` itself."""
    ax, renderer = _pie(leader=leader, label_spacing=spacing)
    assert _min_gap_mm(ax, renderer) == pytest.approx(spacing, abs=TOL_MM)


# -----------------------------------------------------------------------------
# Nothing moves that need not
# -----------------------------------------------------------------------------

def _layout(ax):
    return ([tuple(text.get_position()) for text in ax.texts],
            [(tuple(line.get_xdata()), tuple(line.get_ydata()))
             for line in ax.lines])


@pytest.mark.parametrize("leader", LEADERS)
def test_labels_that_crowd_nothing_do_not_move_at_all(leader):
    """The wide wedges' labels, and their leaders, are exactly where
    ``avoid_overlap=False`` puts them -- equal, not approximately."""
    off, _ = _pie(leader=leader, avoid_overlap=False)
    on, _ = _pie(leader=leader)
    names = [record.path[0] for record in on._publiplots_wedge_meta.wedges]
    (off_texts, off_lines), (on_texts, on_lines) = _layout(off), _layout(on)
    for i, name in enumerate(names):
        if name in WIDE:
            assert on_texts[i] == off_texts[i], name
            if leader is not None:
                assert on_lines[i] == off_lines[i], name
    assert on_texts != off_texts, "the slivers' labels must have moved"


@pytest.mark.parametrize("leader", LEADERS)
def test_a_chart_with_nothing_crowded_is_drawn_identically(leader):
    """No conflict anywhere: the rendered page is the same, to the byte."""
    pages = []
    for avoid in (False, True):
        fig, ax = pp.subplots(axes_size=(60.0, 60.0))
        pp.pieplot(data=ROOMY, x="kind", y="n", legend=False, ax=ax,
                   annotate={"anchor": "outside", "leader": leader,
                             "avoid_overlap": avoid})
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        pages.append(buffer.getvalue())
    assert pages[0] == pages[1]


def test_avoid_overlap_false_never_runs_a_pass(monkeypatch):
    """Off means the layout as it always was: neither pass is reached."""
    def refuse(*args, **kwargs):
        raise AssertionError("a de-collision pass ran")

    monkeypatch.setattr(_wedge, "_spread_on_circle", refuse)
    monkeypatch.setattr(_wedge, "_spread_in_columns", refuse)
    for leader in LEADERS:
        _pie(leader=leader, avoid_overlap=False)


def test_inside_labels_are_never_moved():
    """Under ``auto`` the labels that fit stay exactly where they were."""
    off, _ = _pie(size=60.0, anchor="auto", avoid_overlap=False)
    on, _ = _pie(size=60.0, anchor="auto")
    inside = [i for i, (record, text) in enumerate(
                  zip(on._publiplots_wedge_meta.wedges, on.texts))
              if math.dist(text.get_position(), record.patch.center)
              < record.r_outer]
    assert inside, "the wide wedges' labels fit inside at 60mm"
    for i in inside:
        assert on.texts[i].get_position() == off.texts[i].get_position()
    assert len(on.lines) < len(on.texts), "some labels stayed inside"
    assert _min_gap_mm(*_pie(size=60.0, anchor="auto")) >= 0.4 - TOL_MM


# -----------------------------------------------------------------------------
# Order, leaders and the disc
# -----------------------------------------------------------------------------

def _angle_order(points, centre):
    """Indices sorted anticlockwise from the largest empty sector."""
    angles = [math.atan2(y - centre[1], x - centre[0]) % (2 * math.pi)
              for x, y in points]
    order = sorted(range(len(points)), key=lambda i: angles[i])
    gaps = [(angles[order[(k + 1) % len(order)]] - angles[order[k]])
            % (2 * math.pi) for k in range(len(order))]
    cut = gaps.index(max(gaps))
    return order[cut + 1:] + order[:cut + 1]


@pytest.mark.parametrize("start, direction", TURNS)
@pytest.mark.parametrize("leader", ["radial", None])
def test_labels_keep_their_order_round_the_circle(leader, start, direction):
    ax, renderer = _pie(leader=leader,
                        pie_kws={"start_angle": start,
                                 "direction": direction})
    centre = ax.transData.transform((0.0, 0.0))
    mids = [_rim_display(ax, record)
            for record in ax._publiplots_wedge_meta.wedges]
    labels = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
              for b in _boxes(ax, renderer)]
    assert _angle_order(labels, centre) == _angle_order(mids, centre)


@pytest.mark.parametrize("start, direction", TURNS)
def test_elbow_labels_keep_their_order_down_each_side(start, direction):
    kws = {"start_angle": start, "direction": direction}
    off, _ = _pie(leader="elbow", avoid_overlap=False, pie_kws=kws)
    on, _ = _pie(leader="elbow", pie_kws=kws)
    for ha in ("left", "right"):
        side = [i for i, text in enumerate(on.texts) if text.get_ha() == ha]
        before = sorted(side, key=lambda i: off.texts[i].get_position()[1])
        after = sorted(side, key=lambda i: on.texts[i].get_position()[1])
        assert before == after, ha


@pytest.mark.parametrize("size", [25.0, 40.0, 80.0])
@pytest.mark.parametrize("start", [90, 0, 200])
@pytest.mark.parametrize("leader", ["radial", "elbow"])
def test_every_leader_still_starts_on_its_rim_at_its_mid_ray(
    leader, start, size,
):
    ax, _ = _pie(size=size, leader=leader, pie_kws={"start_angle": start})
    for record, points in zip(ax._publiplots_wedge_meta.wedges, _leaders(ax)):
        assert _mm(ax, math.dist(points[0], _rim_display(ax, record))) == \
            pytest.approx(0.0, abs=1e-6), record.path


@pytest.mark.parametrize("size", [25.0, 40.0, 80.0])
@pytest.mark.parametrize("start", [90, 0, 200])
def test_a_radial_leader_keeps_leader_gap_to_its_own_label(start, size):
    """However far a label moved, no point of its leaning leader comes
    nearer its box than ``leader_gap``, and the nearest is at it."""
    ax, renderer = _pie(size=size, leader="radial",
                        pie_kws={"start_angle": start})
    expected = pp.rcParams["pie.leader_gap"]
    moved = 0
    off, _ = _pie(size=size, leader="radial", avoid_overlap=False,
                  pie_kws={"start_angle": start})
    for i, ((a, b), box) in enumerate(zip(_leaders(ax),
                                          _boxes(ax, renderer))):
        assert _mm(ax, _segment_box(a, b, box)) == pytest.approx(
            expected, abs=TOL_MM), ax.texts[i].get_text()
        moved += ax.texts[i].get_position() != off.texts[i].get_position()
    assert moved >= 2


@pytest.mark.parametrize("size", [25.0, 40.0, 80.0])
@pytest.mark.parametrize("start", [90, 0, 200])
def test_a_moved_elbow_keeps_a_level_tail_leader_gap_from_its_label(
    start, size,
):
    """The elbow keeps its x and rises or drops with the label; the
    tail stays level, ``elbow_length`` long, and ends ``leader_gap``
    short of the label, at its mid-height."""
    kws = {"start_angle": start}
    ax, renderer = _pie(size=size, leader="elbow", pie_kws=kws)
    off, _ = _pie(size=size, leader="elbow", avoid_overlap=False,
                  pie_kws=kws)
    moved = 0
    for i, (points, unmoved, box) in enumerate(zip(
            _leaders(ax), _leaders(off), _boxes(ax, renderer))):
        rim, elbow, tail = points
        assert elbow[0] == pytest.approx(unmoved[1][0], abs=1e-6)
        assert _mm(ax, abs(tail[1] - elbow[1])) == pytest.approx(0.0,
                                                                abs=1e-6)
        assert _mm(ax, abs(tail[0] - elbow[0])) == pytest.approx(
            pp.rcParams["pie.elbow_length"], abs=TOL_MM)
        gap = box[0] - tail[0] if ax.texts[i].get_ha() == "left" \
            else tail[0] - box[2]
        assert _mm(ax, gap) == pytest.approx(pp.rcParams["pie.leader_gap"],
                                             abs=TOL_MM)
        assert _mm(ax, (box[1] + box[3]) / 2.0 - tail[1]) == pytest.approx(
            0.0, abs=TOL_MM)
        moved += abs(elbow[1] - unmoved[1][1]) > 1e-6
    assert moved >= 3


@pytest.mark.parametrize("size", [25.0, 40.0, 80.0])
@pytest.mark.parametrize("start", [90, 0, 200])
def test_an_unled_moved_label_keeps_offset_from_the_rim_on_its_new_ray(
    start, size,
):
    """Placed on its new ray as an unmoved label is on its mid-ray."""
    ax, renderer = _pie(size=size, leader=None, offset=1.5,
                        pie_kws={"start_angle": start})
    centre = ax.transData.transform((0.0, 0.0))
    radius = ax.transData.transform((1.0, 0.0))[0] - centre[0]
    for box in _boxes(ax, renderer):
        phi = math.atan2((box[1] + box[3]) / 2.0 - centre[1],
                         (box[0] + box[2]) / 2.0 - centre[0])
        rim = (centre[0] + radius * math.cos(phi),
               centre[1] + radius * math.sin(phi))
        assert _mm(ax, _point_box(rim, box)) == pytest.approx(1.5,
                                                              abs=TOL_MM)


def _touches_a_wedge(ax, box):
    from matplotlib.transforms import Bbox
    bbox = Bbox.from_extents(*box)
    return [record.path for record in ax._publiplots_wedge_meta.wedges
            if record.patch.get_transform().transform_path(
                record.patch.get_path()).intersects_bbox(bbox, filled=True)]


# Every pairing but elbow labels beside a dict `pull`: there a pulled
# wedge can sit in the column above a crowded run of labels and its
# neighbour's leg below, leaving no height that clears both and keeps the
# labels apart. The column pass then keeps the labels apart and may set
# one on the pulled wedge (`_column_bounds`); at 80mm with the slivers at
# twelve o'clock it does.
DISC_CASES = [(leader, geometry) for leader in LEADERS
              for geometry in ["plain"] + sorted(GEOMETRIES)
              if not (leader == "elbow" and geometry == "pull_dict")]


@pytest.mark.parametrize("size", [25.0, 40.0, 80.0])
@pytest.mark.parametrize("start, direction", TURNS)
@pytest.mark.parametrize("leader, geometry", DISC_CASES)
def test_no_label_is_moved_onto_the_disc(leader, geometry, start, direction,
                                         size):
    """No outside label touches a wedge after the pass that did not
    before it. Without a dict ``pull`` none touches one at all."""
    kws = {"start_angle": start, "direction": direction,
           **GEOMETRIES.get(geometry, {})}
    on, renderer = _pie(size=size, leader=leader, pie_kws=kws)
    off, off_renderer = _pie(size=size, leader=leader, avoid_overlap=False,
                             pie_kws=kws)
    for i, (box, before) in enumerate(zip(_boxes(on, renderer),
                                          _boxes(off, off_renderer))):
        touched = set(_touches_a_wedge(on, box))
        if geometry != "pull_dict":
            assert not touched, (on.texts[i].get_text(), touched)
        else:
            assert touched <= set(_touches_a_wedge(off, before)), \
                on.texts[i].get_text()


@pytest.mark.parametrize("size", [25.0, 40.0, 80.0])
@pytest.mark.parametrize("geometry", ["plain"] + sorted(set(GEOMETRIES)
                                                       - {"pull_dict"}))
@pytest.mark.parametrize("start, direction", TURNS)
@pytest.mark.parametrize("leader", ["radial", "elbow"])
def test_no_two_leaders_cross_and_none_runs_through_a_label(
    leader, start, direction, geometry, size,
):
    kws = {"start_angle": start, "direction": direction,
           **GEOMETRIES.get(geometry, {})}
    ax, renderer = _pie(size=size, leader=leader, pie_kws=kws)
    lines = _leaders(ax)
    segments = [(i, p, q) for i, points in enumerate(lines)
                for p, q in zip(points[:-1], points[1:])]
    for (i, p1, p2), (j, q1, q2) in itertools.combinations(segments, 2):
        if i != j:
            assert not _crosses(p1, p2, q1, q2), (i, j)
    from matplotlib.path import Path
    from matplotlib.transforms import Bbox
    for (i, p, q), (j, box) in itertools.product(
            segments, enumerate(_boxes(ax, renderer))):
        if i != j:
            assert not Path([p, q]).intersects_bbox(
                Bbox.from_extents(*box), filled=False), (i, j)


# -----------------------------------------------------------------------------
# Defaults, overrides and validation
# -----------------------------------------------------------------------------

def test_avoid_overlap_defaults_to_the_rcparam(restore_pie_rcparams):
    off, _ = _pie(avoid_overlap=False)
    on, _ = _pie(avoid_overlap=True)
    assert _layout(_pie()[0]) == _layout(on)
    pp.rcParams["pie.avoid_overlap"] = False
    assert _layout(_pie()[0]) == _layout(off)
    assert _layout(_pie(avoid_overlap=True)[0]) == _layout(on), \
        "a per-call value overrides the rcParam"


def test_label_spacing_defaults_to_the_rcparam(restore_pie_rcparams):
    pp.rcParams["pie.label_spacing"] = 1.2
    assert _min_gap_mm(*_pie()) == pytest.approx(1.2, abs=TOL_MM)
    assert _min_gap_mm(*_pie(label_spacing=0.3)) == pytest.approx(
        0.3, abs=TOL_MM), "a per-call value overrides the rcParam"


def test_the_options_reach_the_strategy_through_pieplot():
    fig, ax = pp.subplots(axes_size=(40.0, 40.0))
    pp.pieplot(data=TAIL, x="kind", y="n", legend=False, ax=ax,
               annotate={"anchor": "outside", "label_spacing": 1.0})
    fig.savefig(io.BytesIO(), format="png")
    assert _min_gap_mm(ax, _settle(fig)) == pytest.approx(1.0, abs=TOL_MM)


@pytest.mark.parametrize("value", [1, 0, "yes", 1.0, [True]])
def test_a_non_bool_avoid_overlap_raises_type_error(value):
    fig, ax = pp.subplots()
    pp.pieplot(data=ROOMY, x="kind", y="n", legend=False, ax=ax)
    with pytest.raises(TypeError, match=r"^wedge_values: avoid_overlap= must "
                       rf"be True or False; got {type(value).__name__} "):
        pp.annotate(ax, "wedge_values", avoid_overlap=value)


def test_a_numpy_bool_avoid_overlap_is_accepted():
    import numpy as np
    off, _ = _pie(avoid_overlap=np.False_)
    assert _layout(off) == _layout(_pie(avoid_overlap=False)[0])


def test_a_bad_avoid_overlap_rcparam_is_refused(restore_pie_rcparams):
    pp.rcParams["pie.avoid_overlap"] = "no"
    fig, ax = pp.subplots()
    pp.pieplot(data=ROOMY, x="kind", y="n", legend=False, ax=ax)
    with pytest.raises(TypeError, match=r"^wedge_values: avoid_overlap="):
        pp.annotate(ax, "wedge_values")


@pytest.mark.parametrize("value, error", [
    (True, TypeError), (False, TypeError), ("wide", TypeError),
    ([1.0], TypeError), (-0.5, ValueError), (float("nan"), ValueError),
    (float("inf"), ValueError),
])
def test_a_bad_label_spacing_is_refused_like_leader_gap(value, error):
    fig, ax = pp.subplots()
    pp.pieplot(data=ROOMY, x="kind", y="n", legend=False, ax=ax)
    with pytest.raises(error, match=r"^wedge_values: label_spacing= must be "):
        pp.annotate(ax, "wedge_values", label_spacing=value)


def test_a_bad_label_spacing_rcparam_is_refused(restore_pie_rcparams):
    fig, ax = pp.subplots()
    pp.pieplot(data=ROOMY, x="kind", y="n", legend=False, ax=ax)
    pp.rcParams["pie.label_spacing"] = -1.0
    with pytest.raises(ValueError, match=r"^wedge_values: label_spacing= "):
        pp.annotate(ax, "wedge_values")


@pytest.mark.parametrize("option", [{"avoid_overlap": True},
                                    {"avoid_overlap": False},
                                    {"label_spacing": 0.4}])
def test_other_kinds_reject_the_options(option):
    from publiplots.annotate._dispatcher import _STRATEGIES, _WEDGE_KINDS

    (name,) = option
    for kind in sorted(set(_STRATEGIES) - _WEDGE_KINDS):
        fig, ax = pp.subplots()
        with pytest.raises(TypeError, match=rf"{name}= is only supported"):
            pp.annotate(ax, kind, **option)


# -----------------------------------------------------------------------------
# The solver
# -----------------------------------------------------------------------------

def _brute_force(desired, gaps, lower, upper):
    """The optimum by enumerating every split of the chain into blocks.

    At the optimum, a maximal run of equal ``z = y - C`` is one block,
    and its value minimises its own squares inside its tightest bounds
    -- otherwise it could move a little alone. So the optimum is the
    cheapest feasible way to split the chain into consecutive blocks
    each at its clipped mean.
    """
    n = len(desired)
    offsets = [0.0]
    for gap in gaps:
        offsets.append(offsets[-1] + gap)
    w = [d - c for d, c in zip(desired, offsets)]
    lo = [b - c for b, c in zip(lower, offsets)]
    hi = [b - c for b, c in zip(upper, offsets)]
    best, best_cost = None, math.inf
    for cuts in itertools.product([False, True], repeat=n - 1):
        z, start = [None] * n, 0
        for end in [k + 1 for k, cut in enumerate(cuts) if cut] + [n]:
            low, high = max(lo[start:end]), min(hi[start:end])
            if low > high:
                break
            mean = sum(w[start:end]) / (end - start)
            z[start:end] = [min(max(mean, low), high)] * (end - start)
            start = end
        else:
            if all(z[k] <= z[k + 1] + 1e-12 for k in range(n - 1)):
                cost = sum((a - b) ** 2 for a, b in zip(z, w))
                if cost < best_cost:
                    best_cost = cost
                    best = [a + c for a, c in zip(z, offsets)]
    return best, best_cost


@pytest.mark.parametrize("seed", range(300))
def test_the_solver_finds_the_brute_force_optimum(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 7)
    desired = sorted(rng.uniform(-3.0, 3.0) for _ in range(n))
    gaps = [rng.choice([0.0, rng.uniform(0.0, 2.0)]) for _ in range(n - 1)]
    lower = [-math.inf] * n
    upper = [math.inf] * n
    # Bounds as the column pass sets them: each admits its own desired
    # value; the labels below the chart, first in the column, may get a
    # ceiling and those above it, last, a floor. That is always feasible
    # -- the first can always go down and the last up. A third of the
    # instances have no bounds at all.
    if seed % 3:
        split = rng.randint(0, n)
        for i in range(n):
            if rng.random() < 0.5:
                step = rng.uniform(0.0, 1.0)
                if i < split:
                    upper[i] = desired[i] + step
                else:
                    lower[i] = desired[i] - step
    got, _ = _wedge._pav_with_gaps(desired, gaps, lower, upper)
    want, cost = _brute_force(desired, gaps, lower, upper)
    assert want is not None
    assert got == pytest.approx(want, abs=1e-9)
    assert sum((a - b) ** 2 for a, b in zip(got, desired)) == pytest.approx(
        cost, abs=1e-9)
    for k in range(n - 1):
        assert got[k + 1] - got[k] >= gaps[k] - 1e-9
    for i in range(n):
        assert lower[i] - 1e-9 <= got[i] <= upper[i] + 1e-9


@pytest.mark.parametrize("seed", range(50))
def test_the_solver_moves_nothing_when_nothing_conflicts(seed):
    """Feasible already: the exact input comes back, not a rounding of it."""
    rng = random.Random(seed)
    n = rng.randint(1, 8)
    gaps = [rng.uniform(0.0, 1.0) for _ in range(n - 1)]
    desired = [rng.uniform(-1.0, 1.0)]
    for gap in gaps:
        desired.append(desired[-1] + gap + rng.choice([0.0, 0.3]))
    got, moved = _wedge._pav_with_gaps(desired, gaps)
    assert got == desired
    assert moved == [False] * n


def test_the_solver_refuses_a_wrong_number_of_gaps():
    with pytest.raises(ValueError, match="need 2 gaps"):
        _wedge._pav_with_gaps([0.0, 1.0, 2.0], [0.5])
