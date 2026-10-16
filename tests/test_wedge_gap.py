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


@pytest.mark.parametrize("kwargs", [
    pytest.param(dict(pull=0.05), id="pie"),
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
    expected = 2.0 * t + kwargs.get("gap", 0.0)
    assert len(gaps) == 4
    for pair, measured in gaps.items():
        assert measured == pytest.approx([expected] * 4, abs=0.01), pair


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
    h, theta1 = 0.05, 17.0
    centre = np.array([0.3, -0.2])
    patch = GappedWedge(tuple(centre), 1.0, theta1, theta1 + span,
                        width=1.0, inset=h)
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


# ------------------------------------------------- too thin for the gap


@pytest.fixture
def sliver():
    """A 0.1% wedge: on a 40mm pie its rim is 0.04mm wide."""
    return pd.DataFrame({"kind": ["a", "b", "c"], "n": [60.0, 39.9, 0.1]})


def test_a_wedge_too_thin_for_the_gap_draws_no_ink(sliver):
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
    assert red_pixels(2.0) == 0


def test_a_collapsed_wedge_keeps_its_record_label_and_legend(sliver):
    fig, ax = pp.subplots(axes_size=(40, 40))
    pp.pieplot(data=sliver, x="kind", y="n", gap=2.0, ax=ax,
               annotate=True)
    records = ax._publiplots_wedge_meta.wedges
    assert [r.path[0] for r in records] == ["a", "b", "c"]
    assert [r.patch.is_collapsed() for r in records] == [False, False, True]
    assert ax.patches.count(records[2].patch) == 1
    assert "0.1%" in [t.get_text() for t in ax.texts]
    from publiplots.utils.legend_entries import get_entries
    (entry,) = get_entries(ax)
    assert list(entry.labels) == ["a", "b", "c"]
    for fmt in ("png", "pdf", "svg"):
        fig.savefig(io.BytesIO(), format=fmt)


def test_a_reflex_wedge_with_nothing_left_to_gap_from_is_drawn_whole():
    """Everything else in the ring is too thin to survive the gap, so
    the big wedge is drawn as a lone wedge would be rather than split by
    a slot running past its rim."""
    data = pd.DataFrame({"kind": ["a", "b"], "n": [99.95, 0.05]})
    ax = _pie(data, axes_size=(40, 40), gap=2.0)
    big, thin = ax._publiplots_wedge_meta.wedges
    assert thin.patch.is_collapsed()
    patch = big.patch
    plain = Wedge(patch.center, patch.r, patch.theta1, patch.theta2,
                  width=patch.width)
    assert np.array_equal(patch.get_path().vertices,
                          plain.get_path().vertices)


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
