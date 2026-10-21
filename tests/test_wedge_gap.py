"""Tests for pieplot's round wedge joins and its uniform ``gap=``.

Gaps are measured on the rendered geometry: each wedge's path is put
through the transform it is drawn with, after a draw, and converted from
display pixels to millimetres.
"""
from __future__ import annotations

import io
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Wedge
from matplotlib.path import Path
from PIL import Image

import publiplots as pp
from publiplots.plot.wedge.draw import GappedWedge

MM_PER_INCH = 25.4


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def df():
    """The gallery's data: two wide wedges and two thin ones."""
    return pd.DataFrame({
        "kind": ["symptom", "admin", "lay_dx", "none"],
        "n": [52.2, 41.8, 6.0, 4.0],
    })


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _edges_mm(patch):
    """The two straight edges of a wedge's rendered path, in mm.

    Returns ``(leading, trailing)``: the edge on the ``theta1`` side
    (the smaller angle) and the one on the ``theta2`` side, each as its
    two endpoints. The path runs outer arc, ``LINETO`` down the trailing
    edge, then the inner arc or the apex, and closes up the leading edge.
    """
    fig = patch.figure
    path = patch.get_path()
    vertices = patch.get_transform().transform(path.vertices)
    vertices = vertices / fig.dpi * MM_PER_INCH
    codes = path.codes
    k = int(np.flatnonzero(codes == Path.LINETO)[0])
    last = len(vertices) - (2 if codes[-1] == Path.CLOSEPOLY else 1)
    return ((vertices[last], vertices[0]), (vertices[k - 1], vertices[k]))


def _distance_to_line(a, b, points):
    unit = (b - a) / np.linalg.norm(b - a)
    return np.abs(_cross(unit, np.asarray(points) - a))


def _neighbour_gaps_mm(ax):
    """Perpendicular distance between every pair of neighbouring edges.

    For each shared ray, the distance from both endpoints of one wedge's
    edge to the line through its neighbour's edge, and the reverse: four
    numbers per pair, all equal only if the edges are parallel and that
    far apart.
    """
    ax.figure.canvas.draw()
    # The layout resets the axes position after a draw; under
    # aspect="equal" the drawn box is the square apply_aspect makes.
    ax.apply_aspect()
    records = ax._publiplots_wedge_meta.wedges
    gaps = {}
    for i, here in enumerate(records):
        # Clockwise: this wedge's leading edge is the next one's trailing.
        nxt = records[(i + 1) % len(records)]
        leading, _ = _edges_mm(here.patch)
        _, trailing = _edges_mm(nxt.patch)
        gaps[(here.path[0], nxt.path[0])] = np.concatenate([
            _distance_to_line(*leading, np.array(trailing)),
            _distance_to_line(*trailing, np.array(leading)),
        ])
    return gaps


def _pie(data, *, axes_size=(50, 50), **kwargs):
    fig, ax = pp.subplots(axes_size=axes_size)
    pp.pieplot(data=data, x="kind", y="n", legend=False, ax=ax, **kwargs)
    return ax


# ---------------------------------------------------------------- joins


def test_every_wedge_is_drawn_with_round_joins(df):
    """A miter join at a thin wedge's apex spikes past the centre."""
    ax = _pie(df)
    joins = [p.get_joinstyle() for p in ax.patches
             if isinstance(p, Wedge)]
    assert joins == ["round"] * 4


def test_round_joins_survive_a_gap(df):
    ax = _pie(df, gap=1.0, donut=0.5)
    assert {p.get_joinstyle() for p in ax.patches
            if isinstance(p, Wedge)} == {"round"}


# ------------------------------------------------------- the gap, in mm


@pytest.mark.parametrize("kwargs", [
    pytest.param(dict(gap=1.0), id="pie"),
    pytest.param(dict(gap=0.4), id="pie-narrow"),
    pytest.param(dict(gap=1.0, donut=0.55), id="donut"),
    pytest.param(dict(gap=2.0, donut=0.55), id="donut-wide"),
    # The thin wedges' offset edges meet outside this hole, so they end
    # in an apex rather than an inner arc.
    pytest.param(dict(gap=2.0, donut=0.1), id="donut-small-hole"),
    pytest.param(dict(gap=1.0, axes_size=(40, 70)), id="tall-cell"),
    pytest.param(dict(gap=1.0, aspect="equal", axes_size=(70, 40)),
                 id="equal-wide"),
    pytest.param(dict(gap=1.0, aspect="equal", axes_size=(40, 70)),
                 id="equal-tall"),
    pytest.param(dict(gap=1.0, hatch="kind", alpha=1.0, edgecolor="black",
                      linewidth=1.5), id="hatch-alpha-edge-width"),
])
def test_every_pair_of_neighbours_is_gap_apart(df, kwargs):
    """Wide (52%, 42%) and thin (6%, 4%) wedges alike, including the
    pair across twelve o'clock."""
    gap = kwargs["gap"]
    gaps = _neighbour_gaps_mm(_pie(df, **kwargs))
    assert len(gaps) == 4
    for pair, measured in gaps.items():
        assert measured == pytest.approx([gap] * 4, abs=0.01), pair


@pytest.mark.parametrize("donut", [0.0, 0.03, 0.4])
def test_a_reflex_wedge_keeps_the_gap(donut):
    """A 70% wedge spans 252 degrees: its half-span is past 90, so its
    apex sits past the centre on its own side."""
    two = pd.DataFrame({"kind": ["a", "b"], "n": [70.0, 30.0]})
    gaps = _neighbour_gaps_mm(_pie(two, gap=1.5, donut=donut))
    for pair, measured in gaps.items():
        assert measured == pytest.approx([1.5] * 4, abs=0.01), pair


@pytest.mark.parametrize("donut, gap", [(0.55, 1.0), (0.55, 2.0),
                                        (0.1, 2.0), (0.4, 1.5)])
def test_a_donut_keeps_its_hole_and_its_inner_radius(df, donut, gap):
    """The inset moves the straight edges only: no wedge reaches into
    the hole, and a wedge whose offset edges meet inside the hole ends
    in an arc on the inner radius rather than an apex."""
    ax = _pie(df, gap=gap, donut=donut)
    fig = ax.figure
    fig.canvas.draw()
    to_mm = MM_PER_INCH / fig.dpi
    centre = ax.transData.transform((0.0, 0.0)) * to_mm
    r_in = (ax.transData.transform((donut, 0.0)) * to_mm - centre)[0]
    h = gap / 2.0
    for record in ax._publiplots_wedge_meta.wedges:
        patch = record.patch
        path = patch.get_path()
        drawn = path.vertices[path.codes != Path.CLOSEPOLY]
        radii = np.linalg.norm(
            patch.get_transform().transform(drawn) * to_mm - centre, axis=1)
        assert radii.min() >= r_in - 0.01, record.path
        half = math.radians(record.theta1 - record.theta0) / 2.0
        if h / math.sin(half) < r_in:
            assert radii.min() == pytest.approx(r_in, abs=0.01), record.path


def test_a_pulled_wedge_leaves_its_unpulled_neighbours_gap_apart(df):
    ax = _pie(df, gap=1.0, pull={"lay_dx": 0.15})
    gaps = _neighbour_gaps_mm(ax)
    for pair in [("symptom", "admin"), ("none", "symptom")]:
        assert gaps[pair] == pytest.approx([1.0] * 4, abs=0.01), pair
    # The pulled wedge moved along its own mid-ray, so its edges stay
    # parallel to its neighbours' and the gap only widens.
    for pair in [("admin", "lay_dx"), ("lay_dx", "none")]:
        assert np.ptp(gaps[pair]) < 0.01
        assert gaps[pair].min() > 1.0 + 0.1


def test_gap_defaults_to_the_rcparam(df):
    previous = pp.rcParams["pie.gap"]
    pp.rcParams["pie.gap"] = 0.8
    try:
        gaps = _neighbour_gaps_mm(_pie(df))
    finally:
        pp.rcParams["pie.gap"] = previous
    for pair, measured in gaps.items():
        assert measured == pytest.approx([0.8] * 4, abs=0.01), pair


def test_the_painted_band_is_the_gap_less_one_outline(df):
    """Read off the pixels at 1200 dpi: the unpainted band between two
    solid wedges is gap minus the outline width, to within the two
    antialiased pixels at its edges (0.042mm)."""
    dpi = 1200
    ax = _pie(df, gap=1.0, alpha=1.0)
    fig = ax.figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = np.asarray(Image.open(buf).convert("RGB"))
    ax.apply_aspect()
    theta = math.radians(ax._publiplots_wedge_meta.wedges[1].theta0)
    normal = np.array([-math.sin(theta), math.cos(theta)])
    step = 0.25  # px
    outline = 0.75 / 72 * MM_PER_INCH
    for frac in (0.3, 0.6, 0.9):
        point = (frac * math.cos(theta), frac * math.sin(theta))
        at = ax.transData.transform(point) * dpi / fig.dpi
        samples = at + np.arange(-200, 200)[:, None] * step * normal
        rows = (img.shape[0] - samples[:, 1]).astype(int)
        pixels = img[rows, samples[:, 0].astype(int)]
        blank = (pixels.min(axis=1) >= 250).sum() * step / dpi * MM_PER_INCH
        assert blank == pytest.approx(1.0 - outline, abs=2 / dpi * 25.4)


# ------------------------------------------------- a scalar pull, evenly


def _radius_mm(ax):
    """The unexploded outer radius, 1 data unit, in rendered mm."""
    fig = ax.figure
    (x0, _), (x1, _) = ax.transData.transform([(0.0, 0.0), (1.0, 0.0)])
    return (x1 - x0) / fig.dpi * MM_PER_INCH


def _curve_radii_mm(ax, patch):
    """Distance from the chart centre of points along the drawn outline.

    Sampled on the path's own Bezier segments rather than read off its
    vertices: an arc's control points sit outside the arc.
    """
    fig = ax.figure
    points = np.vstack([bezier(np.linspace(0.0, 1.0, 64))
                        for bezier, _ in patch.get_path().iter_bezier()])
    to_mm = MM_PER_INCH / fig.dpi
    centre = ax.transData.transform((0.0, 0.0)) * to_mm
    return np.linalg.norm(ax.transData.transform(points) * to_mm - centre,
                          axis=1)


def _expected_insets_mm(ax, h_mm):
    """Each wedge's drawn edge inset, in mm, by the rule rather than read
    off the patch: the full ``h_mm``, capped for a wedge of at most 180
    degrees at ``R * sin(half-span / 2)``, ``R`` its drawn outer radius,
    so its rim keeps half its span."""
    scale = _radius_mm(ax)
    insets = {}
    for record in ax._publiplots_wedge_meta.wedges:
        span = record.theta1 - record.theta0
        cap = (record.r_outer * scale * math.sin(math.radians(span) / 4.0)
               if span <= 180.0 else math.inf)
        insets[record.path[0]] = min(h_mm, cap)
    return insets


@pytest.mark.parametrize("kwargs", [
    pytest.param(dict(pull=0.05), id="pie"),
    # The 4% wedge is capped from here on: see _expected_insets_mm.
    pytest.param(dict(pull=0.12), id="pie-wide"),
    pytest.param(dict(pull=0.05, gap=1.0), id="pie-gap"),
    pytest.param(dict(pull=0.05, donut=0.55), id="donut"),
    pytest.param(dict(pull=0.05, donut=0.55, gap=1.0), id="donut-gap"),
    pytest.param(dict(pull=0.05, axes_size=(40, 70)), id="tall-cell"),
    pytest.param(dict(pull=0.08, aspect="equal", axes_size=(70, 40)),
                 id="equal-wide"),
    pytest.param(dict(pull=0.08, gap=0.5, aspect="equal",
                      axes_size=(40, 70)), id="equal-tall-gap"),
])
def test_a_scalar_pull_parts_every_pair_of_neighbours_by_two_t(df, kwargs):
    """Every pair -- the 52/42 one, the 6/4 one and the one across
    twelve o'clock -- is the same 2t (+ gap) apart, thin wedges included.
    A translation along each mid-ray would part the 4% and 6% slivers
    by a fraction of what it parts the two wide wedges by."""
    ax = _pie(df, **kwargs)
    gaps = _neighbour_gaps_mm(ax)
    t = kwargs["pull"] * _radius_mm(ax)
    insets = _expected_insets_mm(ax, t + kwargs.get("gap", 0.0) / 2.0)
    assert len(gaps) == 4
    for (a, b), measured in gaps.items():
        assert measured == pytest.approx([insets[a] + insets[b]] * 4,
                                         abs=0.01), (a, b)
    # The pair of wide wedges is never capped, so it is exactly 2t + gap.
    assert gaps[("symptom", "admin")] == pytest.approx(
        [2.0 * t + kwargs.get("gap", 0.0)] * 4, abs=0.01)


def test_a_scalar_pull_parts_a_reflex_wedge_evenly_too():
    two = pd.DataFrame({"kind": ["a", "b"], "n": [70.0, 30.0]})
    ax = _pie(two, pull=0.06, gap=0.6)
    gaps = _neighbour_gaps_mm(ax)
    expected = 2.0 * 0.06 * _radius_mm(ax) + 0.6
    for pair, measured in gaps.items():
        assert measured == pytest.approx([expected] * 4, abs=0.01), pair


@pytest.mark.parametrize("kwargs", [
    pytest.param(dict(pull=0.05), id="pie"),
    pytest.param(dict(pull=0.05, gap=1.0), id="pie-gap"),
    pytest.param(dict(pull=0.05, donut=0.55), id="donut"),
    pytest.param(dict(pull=0.05, donut=0.6, gap=0.5), id="donut-gap"),
    pytest.param(dict(pull=0.08, aspect="equal", axes_size=(70, 40)),
                 id="equal"),
])
def test_a_scalar_pull_moves_every_rim_and_inner_boundary_out_by_t(
    df, kwargs,
):
    """Rims at r_out + t, nothing inside r_in + t.

    In a donut every wedge here keeps an inner arc, which lies on
    r_in + t exactly. In a pie the wedges end where their inset edges
    meet; the 52% wedge is a hair past a half-disc, so with no gap its
    edge comes down to the hole itself, at t.
    """
    ax = _pie(df, **kwargs)
    ax.figure.canvas.draw()
    ax.apply_aspect()
    radius = _radius_mm(ax)
    t = kwargs["pull"] * radius
    donut = kwargs.get("donut", 0.0)
    hole = donut * radius + t
    nearest = []
    for record in ax._publiplots_wedge_meta.wedges:
        radii = _curve_radii_mm(ax, record.patch)
        assert radii.max() == pytest.approx(radius + t, abs=0.01), \
            record.path
        assert radii.min() >= hole - 0.01, record.path
        if donut:
            assert radii.min() == pytest.approx(hole, abs=0.01), record.path
        nearest.append(radii.min())
    if not donut and not kwargs.get("gap"):
        assert min(nearest) == pytest.approx(t, abs=0.01)


def test_a_scalar_pull_keeps_every_wedge_and_the_hole_on_the_centre(df):
    """Nothing is translated, so the hole -- and the centre label in it
    -- stays centred on the chart."""
    ax = _pie(df, pull=0.05, donut=0.55, center_label="total")
    for record in ax._publiplots_wedge_meta.wedges:
        assert tuple(record.patch.center) == (0.0, 0.0), record.path
    (label,) = [t for t in ax.texts if t.get_text() == "total"]
    assert label.get_position() == (0.0, 0.0)
    assert (label.get_ha(), label.get_va()) == ("center", "center")


def test_a_scalar_pull_leaves_room_in_the_frame_for_the_moved_rim(df):
    ax = _pie(df, pull=0.2)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert min(-x0, x1, -y0, y1) > 1.2


@pytest.mark.parametrize("donut", [0.0, 0.5])
def test_a_scalar_pull_on_a_single_category_draws_a_full_ring(donut):
    """No edges to inset and no neighbour to part from: the lone wedge
    moves out like any other, to a ring from r_in + t to r_out + t."""
    one = pd.DataFrame({"kind": ["a"], "n": [1.0]})
    ax = _pie(one, pull=0.1, gap=1.0, donut=donut)
    (patch,) = [p for p in ax.patches if isinstance(p, Wedge)]
    ring = Wedge((0.0, 0.0), 1.1, 90.0 - 360.0, 90.0, width=1.0 - donut)
    assert np.allclose(patch.get_path().vertices, ring.get_path().vertices,
                       atol=1e-12, rtol=0)
    (record,) = ax._publiplots_wedge_meta.wedges
    assert (record.r_inner, record.r_outer) == pytest.approx(
        (donut + 0.1, 1.1))


def test_a_dict_pull_still_translates_the_wedges_it_names(df):
    """Exactly what it did before the even explode: the named wedge's
    shape is unchanged and moved along its own mid-ray, the others stay
    put, and the records keep the drawn radii."""
    ax = _pie(df, pull={"lay_dx": 0.15, "none": 0.05})
    for record in ax._publiplots_wedge_meta.wedges:
        patch = record.patch
        pull = {"lay_dx": 0.15, "none": 0.05}.get(record.path[0], 0.0)
        mid = math.radians((record.theta0 + record.theta1) / 2.0)
        assert patch.center == pytest.approx(
            (pull * math.cos(mid), pull * math.sin(mid)), abs=1e-12)
        assert (record.r_inner, record.r_outer, patch.r) == (0.0, 1.0, 1.0)
        assert patch.get_inset() == 0.0
        assert record.pull == pull
        plain = Wedge(patch.center, 1.0, record.theta0, record.theta1,
                      width=1.0)
        assert np.array_equal(patch.get_path().vertices,
                              plain.get_path().vertices)


@pytest.mark.parametrize("kwargs", [dict(), dict(donut=0.55),
                                    dict(gap=1.0), dict(annotate=True)])
def test_a_zero_scalar_pull_draws_exactly_what_no_pull_does(df, kwargs):
    def render(**extra):
        ax = _pie(df, **kwargs, **extra)
        buf = io.BytesIO()
        ax.figure.savefig(buf, format="raw", dpi=200)
        records = [(r.r_inner, r.r_outer, tuple(r.patch.center),
                    r.patch.get_inset())
                   for r in ax._publiplots_wedge_meta.wedges]
        return buf.getvalue(), records, ax.get_xlim(), ax.get_ylim()
    assert render(pull=0.0) == render()


@pytest.mark.parametrize("bad", [-0.05, float("nan"), float("inf")])
def test_a_negative_or_non_finite_scalar_pull_is_refused(df, bad):
    with pytest.raises(ValueError, match=r"pull="):
        _pie(df, pull=bad)


@pytest.mark.parametrize("bad", ["0.1", True, [0.1]])
def test_a_non_numeric_scalar_pull_is_refused(df, bad):
    with pytest.raises(TypeError, match=r"pull="):
        _pie(df, pull=bad)


# ------------------------------------------------- the geometry, exactly


@pytest.mark.parametrize("span", [8.0, 40.0, 89.0, 90.0, 180.0, 181.0,
                                  250.0, 340.0])
def test_the_apex_sits_on_the_mid_ray_at_h_over_sin_half(span):
    """For convex and reflex wedges alike: the apex is where both offset
    edges meet, h from each original edge line."""
    theta1 = 17.0
    centre = np.array([0.3, -0.2])
    patch = GappedWedge(tuple(centre), 1.0, theta1, theta1 + span,
                        width=1.0, inset=0.05)
    # The 8-degree wedge is capped so its rim keeps half its span.
    h = min(0.05, math.sin(math.radians(span) / 4.0)) if span <= 180 \
        else 0.05
    assert patch.get_drawn_inset() == pytest.approx(h, abs=1e-15)
    assert (h < 0.05) == (span == 8.0)
    path = patch.get_path()
    k = int(np.flatnonzero(path.codes == Path.LINETO)[0])
    apex = path.vertices[k] - centre
    half = math.radians(span / 2.0)
    mid = math.radians(theta1) + half
    expected = h / math.sin(half) * np.array([math.cos(mid), math.sin(mid)])
    assert apex == pytest.approx(expected, abs=1e-12)
    # Each edge's outer corner is on the rim and h from its edge line.
    corners = (path.vertices[0] - centre, path.vertices[k - 1] - centre)
    for theta, corner in zip((theta1, theta1 + span), corners):
        ray = np.array([math.cos(math.radians(theta)),
                        math.sin(math.radians(theta))])
        assert abs(_cross(ray, apex)) == pytest.approx(h, abs=1e-12)
        assert abs(_cross(ray, corner)) == pytest.approx(h, abs=1e-12)
        assert np.linalg.norm(corner) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("pull", np.linspace(0.01, 0.2, 39),
                         ids=lambda p: f"{p:.3f}")
def test_an_evenly_exploded_reflex_wedge_always_ends_in_its_inner_arc(pull):
    """Under an even explode a pie wedge's inner radius equals its inset,
    ``t``, in exact arithmetic; floating point puts the computed one a
    hair either side. Whichever side, the 95% wedge of a 95/3/2 pie is
    drawn alike: fully inset by ``t``, ending in an arc on the radius-``t``
    hole -- never in the apex far out on its mid-ray (a deep notch), and
    never uninset as though its rim had vanished."""
    frame = pd.DataFrame({"kind": list("abc"), "n": [95, 3, 2]})
    ax = _pie(frame, axes_size=(40, 40), pull=float(pull))
    patch = ax._publiplots_wedge_meta.wedges[0].patch
    assert patch.theta2 - patch.theta1 > 180.0
    t = float(pull)
    assert patch.get_drawn_inset() == pytest.approx(t, rel=1e-9)
    path = patch.get_path()
    drawn = path.codes != Path.CLOSEPOLY
    radii = np.linalg.norm(path.vertices[drawn] - patch.center, axis=1)
    assert radii.min() >= t * (1.0 - 1e-9)
    # The outline closes from an inner arc's last curve vertex on the
    # hole, not from a LINETO in to the apex.
    assert path.codes[-2] == Path.CURVE4
    assert radii[-1] == pytest.approx(t, rel=1e-9)


@pytest.mark.parametrize("width", [1.0, 0.5])
def test_a_full_circle_is_not_inset(width):
    plain = Wedge((0.0, 0.0), 1.0, 90.0, 450.0, width=width)
    gapped = GappedWedge((0.0, 0.0), 1.0, 90.0, 450.0, width=width,
                         inset=0.1)
    assert np.array_equal(gapped.get_path().vertices,
                          plain.get_path().vertices)


@pytest.mark.parametrize("donut", [0.0, 0.5])
def test_a_single_category_pie_draws_its_full_disc(donut):
    one = pd.DataFrame({"kind": ["a"], "n": [1.0]})
    ax = _pie(one, gap=2.0, donut=donut)
    (patch,) = [p for p in ax.patches if isinstance(p, Wedge)]
    plain = Wedge(patch.center, patch.r, patch.theta1, patch.theta2,
                  width=patch.width)
    assert np.array_equal(patch.get_path().vertices,
                          plain.get_path().vertices)


# ---------------------------------------------------------- gap == 0


@pytest.mark.parametrize("kwargs", [
    dict(), dict(donut=0.55), dict(pull={"lay_dx": 0.15}),
    dict(aspect="equal", axes_size=(70, 40)),
])
def test_a_zero_gap_draws_matplotlibs_own_wedge(df, kwargs):
    ax = _pie(df, gap=0.0, **kwargs)
    for patch in [p for p in ax.patches if isinstance(p, Wedge)]:
        plain = Wedge(patch.center, patch.r, patch.theta1, patch.theta2,
                      width=patch.width)
        assert np.array_equal(patch.get_path().vertices,
                              plain.get_path().vertices)
        assert np.array_equal(patch.get_path().codes,
                              plain.get_path().codes)


def test_an_explicit_zero_gap_overrides_the_rcparam(df):
    """gap=0 is a value, not "unset": with rcParams["pie.gap"] at 1 it
    must still render exactly what the default rcParam of 0 does."""
    def render(**kwargs):
        ax = _pie(df, donut=0.4, annotate=True, **kwargs)
        buf = io.BytesIO()
        ax.figure.savefig(buf, format="raw", dpi=200)
        return buf.getvalue()

    assert pp.rcParams["pie.gap"] == 0.0
    baseline = render()
    previous = pp.rcParams["pie.gap"]
    pp.rcParams["pie.gap"] = 1.0
    try:
        assert render() != baseline, "the rcParam must reach the drawing"
        assert render(gap=0.0) == baseline
    finally:
        pp.rcParams["pie.gap"] = previous


# --------------------------------- too thin for the inset: capped, kept


@pytest.fixture
def sliver():
    """A 0.1% wedge: on a 40mm pie its rim is 0.04mm wide."""
    return pd.DataFrame({"kind": ["a", "b", "c"], "n": [60.0, 39.9, 0.1]})


THIN = [
    pytest.param([60.0, 39.9, 0.1], (40, 40), id="0.1%-40mm"),
    pytest.param([60.0, 39.5, 0.5], (20, 20), id="0.5%-20mm"),
    # The gallery's 52.2/41.8/6/4 pie: 4 of 104 is the "3.8%" wedge.
    pytest.param([52.2, 41.8, 6.0, 4.0], (40, 40), id="3.8%-40mm"),
]
INSETS = [pytest.param(dict(gap=1.0), id="gap=1"),
          pytest.param(dict(pull=0.1, gap=1.0), id="pull=0.1,gap=1")]


def _thin_pie(shares, size, **kwargs):
    frame = pd.DataFrame({"kind": list("abcd"[:len(shares)]), "n": shares})
    ax = _pie(frame, axes_size=size, **kwargs)
    ax.figure.canvas.draw()
    ax.apply_aspect()
    return ax


def _rim_mm(ax, patch):
    """The drawn outer arc's chord and angular extent, in mm and radians.

    The path starts on the outer arc and the first ``LINETO`` leaves it.
    """
    path = patch.get_path()
    k = int(np.flatnonzero(path.codes == Path.LINETO)[0])
    to_mm = MM_PER_INCH / ax.figure.dpi
    start, end = ax.transData.transform([path.vertices[0],
                                         path.vertices[k - 1]]) * to_mm
    centre = ax.transData.transform(patch.center) * to_mm
    a, b = start - centre, end - centre
    angle = abs(math.atan2(_cross(a, b), float(np.dot(a, b))))
    return float(np.linalg.norm(end - start)), angle


@pytest.mark.parametrize("kwargs", INSETS)
@pytest.mark.parametrize("shares, size", THIN)
def test_a_wedge_too_thin_for_the_inset_keeps_half_its_rim(
    shares, size, kwargs,
):
    """It is capped, not dropped: its rim keeps at least half its span,
    and so at least half its chord, at the radius it is drawn at."""
    ax = _thin_pie(shares, size, **kwargs)
    h = kwargs.get("pull", 0.0) * _radius_mm(ax) + kwargs["gap"] / 2.0
    insets = _expected_insets_mm(ax, h)
    thin = ax._publiplots_wedge_meta.wedges[-1]
    radius = thin.r_outer * _radius_mm(ax)
    half = math.radians(thin.theta1 - thin.theta0) / 2.0
    chord, angle = _rim_mm(ax, thin.patch)
    assert chord >= radius * math.sin(half) - 0.01
    assert radius * angle >= radius * half - 0.01
    # The 3.8% wedge only needs the cap once the explode adds to the
    # gap; the two slivers need it at gap=1 alone. A capped rim keeps
    # exactly half its span, not merely somewhere above it.
    capped = insets[thin.path[0]] < h
    assert capped == (shares[-1] < 1.0 or "pull" in kwargs)
    if capped:
        assert radius * angle == pytest.approx(radius * half, abs=0.01)


@pytest.mark.parametrize("kwargs", INSETS)
@pytest.mark.parametrize("shares, size", THIN)
def test_the_gap_beside_a_capped_wedge_is_h_plus_its_inset(
    shares, size, kwargs,
):
    """``h + h_i`` beside the capped wedge; every pair of uncapped
    neighbours keeps the full ``2h``."""
    ax = _thin_pie(shares, size, **kwargs)
    h = kwargs.get("pull", 0.0) * _radius_mm(ax) + kwargs["gap"] / 2.0
    insets = _expected_insets_mm(ax, h)
    gaps = _neighbour_gaps_mm(ax)
    seen = {"full": 0, "capped": 0}
    for (a, b), measured in gaps.items():
        h_a, h_b = insets[a], insets[b]
        if h_a == h and h_b == h:
            expected, kind = 2.0 * h, "full"
        else:
            # h + h_i beside one capped wedge, h_i + h_j between two.
            expected, kind = h_a + h_b, "capped"
        seen[kind] += 1
        assert measured == pytest.approx([expected] * 4, abs=0.01), (a, b)
    assert seen["full"] >= 1
    assert seen["capped"] >= (2 if insets[ax._publiplots_wedge_meta
                                         .wedges[-1].path[0]] < h else 0)


@pytest.mark.parametrize("kwargs", [
    pytest.param(dict(pull=0.1), id="pull=0.1"),
    pytest.param(dict(gap=1.5), id="gap=1.5"),
])
def test_a_long_tail_keeps_every_wedge_and_every_gap_by_the_rule(kwargs):
    """Shares 40/30/20/5/3/1/0.5/0.5: the full gap between uncapped
    neighbours, h + h_i beside a capped one and h_i + h_j between the
    two capped 0.5% wedges, and every wedge drawn."""
    frame = pd.DataFrame({"kind": list("abcdefgh"),
                          "n": [40, 30, 20, 5, 3, 1, 0.5, 0.5]})
    ax = _pie(frame, axes_size=(40, 40), **kwargs)
    h = kwargs.get("pull", 0.0) * _radius_mm(ax) + kwargs.get("gap", 0) / 2
    insets = _expected_insets_mm(ax, h)
    capped = sorted(k for k, v in insets.items() if v < h)
    assert {"g", "h"} <= set(capped) and {"a", "b", "c"}.isdisjoint(capped)
    gaps = _neighbour_gaps_mm(ax)
    assert len(gaps) == 8
    for (a, b), measured in gaps.items():
        assert measured == pytest.approx([insets[a] + insets[b]] * 4,
                                         abs=0.01), (a, b)
    for record in ax._publiplots_wedge_meta.wedges:
        chord, _ = _rim_mm(ax, record.patch)
        assert chord > 0.0, record.path


def test_a_wedge_too_thin_for_the_gap_still_draws_ink(sliver):
    palette = {"a": "#0000ff", "b": "#00ff00", "c": "#ff0000"}

    def red_pixels(gap):
        ax = _pie(sliver, axes_size=(40, 40), gap=gap, alpha=1.0,
                  hatch="kind", palette=palette)
        buf = io.BytesIO()
        ax.figure.savefig(buf, format="png", dpi=600)
        buf.seek(0)
        img = np.asarray(Image.open(buf).convert("RGB")).astype(int)
        return int(((img[..., 0] > 150) & (img[..., 1] < 100)
                    & (img[..., 2] < 100)).sum())

    assert red_pixels(0.0) > 1000, "the sliver is visible without a gap"
    assert red_pixels(2.0) > 100


def test_a_capped_wedge_keeps_its_record_label_and_legend(sliver):
    fig, ax = pp.subplots(axes_size=(40, 40))
    pp.pieplot(data=sliver, x="kind", y="n", gap=2.0, ax=ax,
               annotate=True)
    records = ax._publiplots_wedge_meta.wedges
    assert [r.path[0] for r in records] == ["a", "b", "c"]
    capped = [r.patch.get_drawn_inset() < r.patch.get_inset()
              for r in records]
    assert capped == [False, False, True]
    assert ax.patches.count(records[2].patch) == 1
    assert "0.1%" in [t.get_text() for t in ax.texts]
    from publiplots.utils.legend_entries import get_entries
    (entry,) = get_entries(ax)
    assert list(entry.labels) == ["a", "b", "c"]
    for fmt in ("png", "pdf", "svg"):
        fig.savefig(io.BytesIO(), format=fmt)


def test_a_reflex_wedge_with_nothing_left_to_gap_from_is_drawn_whole():
    """Everything else in the ring is too thin for the full gap, so the
    big wedge is drawn as a lone wedge would be rather than split by a
    slot running past its rim; the thin one is capped and still drawn,
    clear of it by its own inset."""
    data = pd.DataFrame({"kind": ["a", "b"], "n": [99.95, 0.05]})
    ax = _pie(data, axes_size=(40, 40), gap=2.0)
    big, thin = ax._publiplots_wedge_meta.wedges
    assert 0.0 < thin.patch.get_drawn_inset() < thin.patch.get_inset()
    patch = big.patch
    assert patch.get_drawn_inset() == 0.0
    plain = Wedge(patch.center, patch.r, patch.theta1, patch.theta2,
                  width=patch.width)
    assert np.array_equal(patch.get_path().vertices,
                          plain.get_path().vertices)


@pytest.mark.parametrize("kwargs", [dict(pull=0.1), dict(pull=0.1, gap=1.0),
                                    dict(pull=0.3)])
def test_a_capped_wedge_stays_outside_the_hole(kwargs):
    """Capped insets are smaller than t, but a capped wedge still ends
    no nearer the centre than r_inner + t."""
    frame = pd.DataFrame({"kind": list("abcdefgh"),
                          "n": [40, 30, 20, 5, 3, 1, 0.5, 0.5]})
    ax = _pie(frame, axes_size=(40, 40), **kwargs)
    ax.figure.canvas.draw()
    t = kwargs["pull"] * _radius_mm(ax)
    records = ax._publiplots_wedge_meta.wedges
    assert any(r.patch.get_drawn_inset() < r.patch.get_inset()
               for r in records)
    for record in records:
        assert _curve_radii_mm(ax, record.patch).min() >= t - 0.01, \
            record.path


# ----------------------------------------------------------- validation


@pytest.mark.parametrize("bad", [-0.5, float("nan"), float("inf")])
def test_a_negative_or_non_finite_gap_is_refused(df, bad):
    with pytest.raises(ValueError, match=r"gap="):
        _pie(df, gap=bad)


@pytest.mark.parametrize("bad", ["1", True, [1.0]])
def test_a_non_numeric_gap_is_refused(df, bad):
    with pytest.raises(TypeError, match=r"gap="):
        _pie(df, gap=bad)


def test_a_bad_rcparam_gap_is_refused_too(df):
    previous = pp.rcParams["pie.gap"]
    pp.rcParams["pie.gap"] = -1.0
    try:
        with pytest.raises(ValueError, match=r"gap="):
            _pie(df)
    finally:
        pp.rcParams["pie.gap"] = previous


# ------------------------------------------- per-edge and radial insets


def _edge_distance(patch, angle_deg, vertices):
    """Perpendicular distances from the line of the ray at angle_deg to
    the given path vertices."""
    a = np.radians(angle_deg)
    n = np.array([-np.sin(a), np.cos(a)])
    return np.abs((vertices - np.asarray(patch.center)) @ n)


def _first_corner(path):
    """Index of the first vertex after the outer arc."""
    return int(np.flatnonzero(path.codes == Path.LINETO)[0])


def test_unequal_edge_insets_are_drawn_per_edge():
    """Each edge's straight segment, from the rim to its half's inner
    end, runs at that edge's own inset from its ray."""
    w = GappedWedge((0, 0), 1.0, 10.0, 80.0, width=1.0)
    w.set_insets(0.02, 0.05)
    assert w.get_drawn_insets() == (0.02, 0.05)
    path = w.get_path()
    k = _first_corner(path)
    assert _edge_distance(w, 10.0, path.vertices[[0, -2]]) == \
        pytest.approx([0.02, 0.02], abs=1e-9)
    assert _edge_distance(w, 80.0, path.vertices[[k - 1, k]]) == \
        pytest.approx([0.05, 0.05], abs=1e-9)


def test_equal_insets_reproduce_set_inset_exactly():
    a = GappedWedge((0, 0), 1.0, 10.0, 80.0, width=1.0)
    a.set_inset(0.03)
    b = GappedWedge((0, 0), 1.0, 10.0, 80.0, width=1.0)
    b.set_insets(0.03, 0.03)
    assert a.get_path().vertices.tolist() == b.get_path().vertices.tolist()


def test_radial_insets_move_the_band():
    w = GappedWedge((0, 0), 1.0, 0.0, 90.0, width=0.5)
    w.set_radial_insets(0.01, 0.02)
    assert w.r == pytest.approx(0.98)
    assert w.r - w.width == pytest.approx(0.51)


def test_radial_insets_redraw_the_path_on_the_new_radii():
    w = GappedWedge((0.3, -0.2), 1.0, 0.0, 90.0, width=0.5)
    w.set_insets(0.02, 0.05)
    w.get_path()
    w.set_radial_insets(0.01, 0.02)
    path = w.get_path()
    k = int(np.flatnonzero(path.codes == Path.LINETO)[0])
    # Outer arc start and end, inner arc start and end: on the new radii.
    corners = path.vertices[[0, k - 1, k, -2]] - np.asarray(w.center)
    assert np.linalg.norm(corners, axis=1) == pytest.approx(
        [0.98, 0.98, 0.51, 0.51], rel=1e-12)


def test_each_edge_is_capped_on_its_own():
    """An 8-degree wedge keeps a small inset on one edge and is capped
    on the other, each by the rule of a lone wedge."""
    w = GappedWedge((0, 0), 1.0, 10.0, 18.0, width=1.0)
    w.set_insets(0.001, 0.5)
    cap = math.sin(math.radians(8.0) / 4.0)
    assert w.get_drawn_insets() == (0.001, pytest.approx(cap, abs=1e-15))
    assert w.get_drawn_inset() == pytest.approx(cap, abs=1e-15)


def test_the_cap_reads_the_current_outer_radius():
    """A caller caps against r after its radial insets; the self-cap must
    agree, so it reads the radius as it is now."""
    w = GappedWedge((0, 0), 1.0, 10.0, 18.0, width=0.5)
    w.set_radial_insets(0.0, 0.2)
    w.set_insets(0.5, 0.5)
    cap = 0.8 * math.sin(math.radians(8.0) / 4.0)
    assert w.get_drawn_insets() == (pytest.approx(cap, abs=1e-15),) * 2


def test_the_requested_inset_is_the_larger_edge():
    w = GappedWedge((0, 0), 1.0, 10.0, 80.0, width=1.0)
    w.set_insets(0.05, 0.02)
    assert w.get_inset() == 0.05
    assert w.inset == 0.05
    assert w.get_drawn_inset() == 0.05


def test_a_ring_insets_each_arc_end_by_its_own_edge():
    """With room for the inner arc, both arcs end h0 from the theta1 ray
    and h1 from the theta2 ray."""
    centre = np.array([0.3, -0.2])
    w = GappedWedge(tuple(centre), 1.0, 10.0, 80.0, width=0.5)
    w.set_insets(0.02, 0.05)
    path = w.get_path()
    k = int(np.flatnonzero(path.codes == Path.LINETO)[0])
    # Outer arc start and end, inner arc start (theta2 side) and end.
    corners = [path.vertices[i] - centre for i in (0, k - 1, k, -2)]
    radii = [1.0, 1.0, 0.5, 0.5]
    edges = [10.0, 80.0, 80.0, 10.0]
    insets = [0.02, 0.05, 0.05, 0.02]
    for corner, radius, theta, h in zip(corners, radii, edges, insets):
        ray = np.array([math.cos(math.radians(theta)),
                        math.sin(math.radians(theta))])
        assert np.linalg.norm(corner) == pytest.approx(radius, abs=1e-12)
        assert abs(_cross(ray, corner)) == pytest.approx(h, abs=1e-12)
    assert path.codes[-2] == Path.CURVE4


def test_unequal_halves_end_on_the_mid_ray_each_at_its_own_apex():
    """Without a hole each half ends where its offset edge meets the
    mid-ray, at h_k / sin(half); a radial step joins the two."""
    centre = np.array([0.3, -0.2])
    w = GappedWedge(tuple(centre), 1.0, 10.0, 80.0, width=1.0)
    w.set_insets(0.02, 0.05)
    path = w.get_path()
    k = _first_corner(path)
    assert list(path.codes[k:]) == [Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    half = math.radians(35.0)
    ray = np.array([math.cos(math.radians(45.0)),
                    math.sin(math.radians(45.0))])
    assert path.vertices[k] - centre == pytest.approx(
        0.05 / math.sin(half) * ray, abs=1e-12)
    assert path.vertices[k + 1] - centre == pytest.approx(
        0.02 / math.sin(half) * ray, abs=1e-12)


@pytest.mark.parametrize("width", [1.0, 0.5])
def test_a_full_circle_ignores_per_edge_insets(width):
    plain = Wedge((0.0, 0.0), 1.0, 90.0, 450.0, width=width)
    gapped = GappedWedge((0.0, 0.0), 1.0, 90.0, 450.0, width=width)
    gapped.set_insets(0.1, 0.2)
    assert gapped.get_drawn_insets() == (0.0, 0.0)
    assert np.array_equal(gapped.get_path().vertices,
                          plain.get_path().vertices)


def test_a_reflex_wedge_whose_rim_vanishes_under_either_edge_is_uninset():
    """The vanishing-rim rule reads the larger edge: an inset past the
    outer radius on one edge alone leaves no rim to keep a gap from."""
    plain = Wedge((0.0, 0.0), 1.0, 0.0, 350.0, width=1.0)
    w = GappedWedge((0.0, 0.0), 1.0, 0.0, 350.0, width=1.0)
    w.set_insets(0.0, 1.2)
    assert w.get_drawn_insets() == (0.0, 0.0)
    assert np.array_equal(w.get_path().vertices, plain.get_path().vertices)


def _mid_ray(patch):
    mid = math.radians((patch.theta1 + patch.theta2) / 2.0)
    return np.array([math.cos(mid), math.sin(mid)])


def test_each_half_decides_its_inner_end_by_its_own_inset():
    """A hole between the two insets: the half inset less ends in the
    inner arc, from its edge to the mid-ray; the half inset more ends in
    its corner on the mid-ray, outside the hole."""
    centre = np.array([0.3, -0.2])
    w = GappedWedge(tuple(centre), 1.0, 10.0, 350.0, width=0.97)
    w.set_insets(0.02, 0.05)
    path = w.get_path()
    k = _first_corner(path)
    half = math.radians(170.0)
    corner = path.vertices[k] - centre
    assert corner == pytest.approx(0.05 / math.sin(half) * _mid_ray(w),
                                   abs=1e-12)
    # Then down the mid-ray to the hole, and round it to the theta1 edge.
    assert path.vertices[k + 1] - centre == pytest.approx(
        0.03 * _mid_ray(w), abs=1e-12)
    assert path.codes[-2] == Path.CURVE4
    end = path.vertices[-2] - centre
    assert np.linalg.norm(end) == pytest.approx(0.03, abs=1e-12)
    assert _edge_distance(w, 10.0, path.vertices[[-2]]) == \
        pytest.approx([0.02], abs=1e-12)


def test_a_half_whose_inner_arc_would_vanish_ends_in_its_corner():
    """The inner arc of a half keeps a positive extent only if its edge's
    shift at r_in is short of the half-span: here the theta2 half's is
    not (asin(0.75) > 20 degrees), the theta1 half's is."""
    centre = np.array([0.3, -0.2])
    w = GappedWedge(tuple(centre), 1.0, 10.0, 50.0, width=0.8)
    w.set_insets(0.01, 0.15)
    assert w.get_drawn_insets() == (0.01, 0.15)
    path = w.get_path()
    k = _first_corner(path)
    assert path.vertices[k] - centre == pytest.approx(
        0.15 / math.sin(math.radians(20.0)) * _mid_ray(w), abs=1e-12)
    assert path.codes[-2] == Path.CURVE4
    assert np.linalg.norm(path.vertices[-2] - centre) == pytest.approx(
        0.2, abs=1e-12)


# ------------------------- closing an unequally inset wedge, two halves


def _dense_outline(patch, samples=200):
    """The outline as a dense polyline, relative to the centre: each arc
    segment evaluated along its length (its control points are off the
    curve), each straight segment subdivided."""
    points = []
    for bezier, code in patch.get_path().iter_bezier():
        u = np.linspace(0.0, 1.0, samples)[1:]
        if code == Path.CURVE4:
            points.extend(bezier(u))
        elif code == Path.MOVETO:
            points.append(bezier.control_points[0])
        else:
            p0, p1 = bezier.control_points[0], bezier.control_points[-1]
            points.extend(p0 + (p1 - p0) * u[:, None])
    return np.asarray(points) - np.asarray(patch.center)


def _line_distances(patch):
    """The nearest and farthest any straight segment of the outline comes
    to the centre, exactly."""
    centre = np.asarray(patch.center)
    nearest, farthest = np.inf, 0.0
    for bezier, code in patch.get_path().iter_bezier():
        if code in (Path.LINETO, Path.CLOSEPOLY):
            p0 = bezier.control_points[0] - centre
            p1 = bezier.control_points[-1] - centre
            d = p1 - p0
            u = 0.0 if not d.any() else min(max(-(p0 @ d) / (d @ d), 0.0),
                                             1.0)
            nearest = min(nearest, float(np.linalg.norm(p0 + u * d)))
            farthest = max(farthest, float(np.linalg.norm(p0)),
                           float(np.linalg.norm(p1)))
    return nearest, farthest


def _polyline_distance(a, b):
    """Symmetric max distance from the points of one polyline to the
    segments of the other."""
    def one_way(points, line):
        p0, d = line[:-1], np.diff(line, axis=0)
        dd = np.maximum((d * d).sum(axis=1), 1e-300)
        rel = points[:, None, :] - p0[None, :, :]
        u = np.clip((rel * d[None]).sum(axis=2) / dd, 0.0, 1.0)
        gap = np.linalg.norm(rel - u[..., None] * d[None], axis=2)
        return gap.min(axis=1).max()
    return max(one_way(a, b), one_way(b, a))


def _normals(patch):
    a, b = math.radians(patch.theta1), math.radians(patch.theta2)
    return (np.array([-math.sin(a), math.cos(a)]),
            np.array([math.sin(b), -math.cos(b)]))


_R = 1.0 / 3.0
_SPANS = [20.0, 60.0, 120.0, 160.0, 170.0, 176.0, 178.0, 180.0, 182.0,
          190.0, 200.0, 340.0]
_ORDERS = [(0.002, 0.04), (0.04, 0.002)]
_HOLES = [0.0, 0.00212, 0.02, 0.039, 0.05]
_GRID = pytest.mark.parametrize("span", _SPANS)
_ORDER = pytest.mark.parametrize("insets", _ORDERS)
_HOLE = pytest.mark.parametrize("r_in", _HOLES)


def _wedge(span, insets, r_in, *, r=_R, theta1=-63.0):
    w = GappedWedge((0.3, -0.2), r, theta1, theta1 + span, width=r - r_in)
    w.set_insets(*insets)
    return w


@_HOLE
@_ORDER
@_GRID
def test_unequal_halves_stay_inside_the_annulus(span, insets, r_in):
    """Within the rim and outside the hole: straight segments exactly; the
    arcs to within matplotlib's own Bezier arc error, which dips up to
    2.9e-5 (relative) inside the radius on any Wedge."""
    w = _wedge(span, insets, r_in)
    nearest, farthest = _line_distances(w)
    assert farthest <= _R * (1.0 + 1e-9)
    assert nearest >= r_in * (1.0 - 1e-9)
    radii = np.linalg.norm(_dense_outline(w), axis=1)
    assert radii.max() <= _R * (1.0 + 1e-9)
    assert radii.min() >= r_in * (1.0 - 5e-5)


@_HOLE
@_ORDER
@_GRID
def test_unequal_halves_stay_inside_the_sector(span, insets, r_in):
    w = _wedge(span, insets, r_in)
    n0, n1 = _normals(w)
    points = _dense_outline(w)
    if span <= 180.0:
        assert (points @ n0).min() >= -1e-12
        assert (points @ n1).min() >= -1e-12
    else:
        # Not inside the excluded cone, behind both edges at once.
        assert not np.any((points @ n0 < -1e-12) & (points @ n1 < -1e-12))


@_HOLE
@_ORDER
@_GRID
def test_each_unequal_edge_keeps_its_inset_from_the_rim_in(span, insets,
                                                          r_in):
    """Edge k runs on its offset line, h_k from its ray, from the rim in
    to its half's inner end: its corner on the mid-ray, or the start of
    its inner arc."""
    w = _wedge(span, insets, r_in)
    h0, h1 = w.get_drawn_insets()
    assert h0 != h1
    centre = np.asarray(w.center)
    path = w.get_path()
    k = _first_corner(path)
    n0, n1 = _normals(w)
    a, b = math.radians(w.theta1), math.radians(w.theta2)
    d0 = np.array([math.cos(a), math.sin(a)])
    d1 = np.array([math.cos(b), math.sin(b)])
    rim_end, first = path.vertices[[k - 1, k]] - centre
    last, rim_start = path.vertices[[-2, 0]] - centre
    assert rim_end @ n1 == pytest.approx(h1, abs=1e-9)
    assert first @ n1 == pytest.approx(h1, abs=1e-9)
    assert first @ d1 <= rim_end @ d1
    assert last @ n0 == pytest.approx(h0, abs=1e-9)
    assert rim_start @ n0 == pytest.approx(h0, abs=1e-9)
    assert last @ d0 <= rim_start @ d0


@pytest.mark.parametrize("span, r, r_in, insets", [
    (200.0, _R, 0.00212, (0.002, 0.04)),
    (200.0, _R, 0.00212, (0.04, 0.002)),
    (340.0, 1.0, 0.02, (0.002, 0.04)),
    (300.0, 1.0, 0.039, (0.002, 0.04)),
])
def test_a_reflex_wedge_keeps_its_smaller_edge_out_of_the_hole(
        span, r, r_in, insets):
    """The cases where the smaller-inset edge, closed past its foot, ran
    through a hole between the two insets."""
    w = _wedge(span, insets, r_in, r=r)
    nearest, _ = _line_distances(w)
    assert nearest >= r_in * (1.0 - 1e-9)


@_HOLE
@_ORDER
@pytest.mark.parametrize("delta", [-1e-6, 1e-6])
def test_the_closure_is_continuous_through_180_degrees(delta, insets, r_in):
    at = _wedge(180.0, insets, r_in)
    near = _wedge(180.0 + delta, insets, r_in)
    assert _polyline_distance(_dense_outline(at),
                              _dense_outline(near)) < 1e-5


@_HOLE
@pytest.mark.parametrize("delta", [-1e-7, 1e-7])
@_GRID
def test_unequal_insets_tend_to_the_equal_inset_wedge(span, delta, r_in):
    """As h1 tends to h0 the two halves' inner ends meet and the outline
    tends to the equal-inset path."""
    h = 0.03
    equal = _wedge(span, (h, h), r_in)
    unequal = _wedge(span, (h, h + delta), r_in)
    assert _polyline_distance(_dense_outline(equal),
                              _dense_outline(unequal)) < 1e-5
