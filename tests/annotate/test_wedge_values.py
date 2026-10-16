"""Value labels for pie and donut wedges."""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import publiplots as pp


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def df():
    return pd.DataFrame({"kind": list("ABCD"), "n": [50.0, 25.0, 15.0, 10.0]})


def _pie(df, size=(80.0, 80.0), **kw):
    """A pie big enough that every label fits inside under ``auto``.

    80mm is deliberate: at the default 40mm the 10% wedge is too narrow
    for ``"10.0%"`` at the themed 7pt font, so ``auto`` pushes it out.
    Tests that are not about the fit decision should not be sitting on
    that boundary.
    """
    fig, ax = pp.subplots(axes_size=size)
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax, **kw)
    return ax


def _radius(text):
    return math.hypot(*text.get_position())


MM_PER_INCH = 25.4


def _settle(fig):
    """Draw, so the layout has sized the figure, and return the renderer.

    ``apply_aspect`` afterwards because the layout repositions each axes
    from a draw callback, which resets an ``aspect="equal"`` axes to its
    full cell until the next draw; matplotlib re-applies the aspect at the
    start of every draw, so this reads the box that is actually rendered.
    """
    fig.canvas.draw()
    for ax in fig.axes:
        ax.apply_aspect()
    return fig.canvas.get_renderer()


def _px_to_mm(fig):
    return MM_PER_INCH / fig.dpi


def _mid_ray(record):
    """The wedge's unit mid-ray direction."""
    mid = math.radians((record.theta0 + record.theta1) / 2.0)
    return math.cos(mid), math.sin(mid)


def _box_distance_mm(ax, renderer, text, point):
    """The visible gap from ``point`` to the label's rendered box, in mm.

    The shortest straight-line distance from a display point to the
    nearest point of the box -- what the eye reads as the space between
    a leader's end and its label. Zero when the point is inside the box.
    """
    box = text.get_window_extent(renderer)
    dx = max(box.x0 - point[0], 0.0, point[0] - box.x1)
    dy = max(box.y0 - point[1], 0.0, point[1] - box.y1)
    return math.hypot(dx, dy) * _px_to_mm(ax.figure)


def _rim_display(ax, record):
    """The display point where the wedge's mid-ray crosses its rim."""
    cx, cy = record.patch.center
    dx, dy = _mid_ray(record)
    return ax.transData.transform((cx + record.r_outer * dx,
                                   cy + record.r_outer * dy))


def _outside_records(ax):
    """``(record, text)`` for every label placed outside its wedge."""
    return [(record, text)
            for record, text in zip(ax._publiplots_wedge_meta.wedges,
                                    ax.texts)
            if math.dist(text.get_position(), record.patch.center)
            > record.r_outer]


def _rim_gaps_mm(ax):
    """Rendered ``offset``, in millimetres, for every outside label.

    Measured in display space, so it reads what is drawn under either
    framing rather than assuming the two data scales agree. With a
    leader it is the leader's length, rim to end; with ``leader=None``,
    the visible gap from the rim point to the label's box.
    """
    renderer = _settle(ax.figure)
    pairs = _outside_records(ax)
    if ax.lines:
        assert len(ax.lines) == len(pairs)
        return [math.dist(*_display(ax, line.get_xdata(),
                                    line.get_ydata())[:2])
                * _px_to_mm(ax.figure)
                for line in ax.lines]
    return [_box_distance_mm(ax, renderer, text, _rim_display(ax, record))
            for record, text in pairs]


def _end_gaps_mm(ax):
    """Visible gap, in mm, from each radial leader's end to its label."""
    renderer = _settle(ax.figure)
    pairs = _outside_records(ax)
    assert len(ax.lines) == len(pairs)
    return [_box_distance_mm(
                ax, renderer, text,
                _display(ax, line.get_xdata(), line.get_ydata())[-1])
            for (_, text), line in zip(pairs, ax.lines)]


def _display(ax, xs, ys):
    return ax.transData.transform(list(zip(xs, ys)))


def _disc_diameter_mm(ax):
    """Rendered diameter of the chart's disc, read off the y axis."""
    _settle(ax.figure)
    r_outer = max(w.r_outer for w in ax._publiplots_wedge_meta.wedges)
    (_, y0), (_, y1) = ax.transData.transform([(0.0, 0.0), (0.0, r_outer)])
    return 2.0 * abs(y1 - y0) * _px_to_mm(ax.figure)


def _decorations(ax):
    """Every label and leader line the strategy added to ``ax``."""
    return list(ax.texts) + list(ax.lines)


def _boxes_overlap(a, b):
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


# -----------------------------------------------------------------------------
# Label text
# -----------------------------------------------------------------------------

def test_default_label_is_a_percentage(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values")
    assert [t.get_text() for t in ax.texts] == ["50.0%", "25.0%", "15.0%",
                                                "10.0%"]


def test_fmt_returns_the_raw_value(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", fmt=",.0f")
    assert [t.get_text() for t in ax.texts] == ["50", "25", "15", "10"]


def test_fmt_template_formats_the_value_not_the_share(df):
    """An explicit spec switches the label from share to value.

    Guards the default: if the sentinel branch were dropped and the
    percentage computed unconditionally, these would read "50.0%".
    """
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", fmt="{:.1f}")
    assert [t.get_text() for t in ax.texts] == ["50.0", "25.0", "15.0", "10.0"]


def test_one_label_per_wedge(df):
    ax = _pie(df)
    labels = pp.annotate(ax, "wedge_values")
    assert len(labels) == len(ax._publiplots_wedge_meta.wedges)


def test_labels_are_returned_in_wedge_order(df):
    ax = _pie(df)
    labels = pp.annotate(ax, "wedge_values", fmt=",.0f")
    values = [str(int(w.value)) for w in ax._publiplots_wedge_meta.wedges]
    assert [t.get_text() for t in labels] == values


# -----------------------------------------------------------------------------
# Placement
# -----------------------------------------------------------------------------

def test_labels_sit_inside_their_own_wedge(df):
    """Each label's position must fall within its own record's angular span."""
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="inside")
    records = ax._publiplots_wedge_meta.wedges
    for record, text in zip(records, ax.texts):
        tx, ty = text.get_position()
        angle = math.degrees(math.atan2(ty, tx)) % 360.0
        lo, hi = record.theta0 % 360.0, record.theta1 % 360.0
        inside = (lo <= angle <= hi) if lo <= hi else (angle >= lo or angle <= hi)
        assert inside, f"{record.path} label at {angle} outside [{lo}, {hi}]"


def test_inside_labels_sit_within_the_rim(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="inside")
    assert all(_radius(t) < 1.0 for t in ax.texts)


def test_inside_anchor_never_pushes_a_thin_wedge_out():
    """An explicit anchor is an instruction, not a hint.

    The 0.5% wedge cannot hold its label, but `inside` was asked for.
    """
    thin = pd.DataFrame({"kind": list("AB"), "n": [99.5, 0.5]})
    ax = _pie(thin)
    pp.annotate(ax, "wedge_values", anchor="inside")
    assert all(_radius(t) < 1.0 for t in ax.texts)
    assert len(ax.lines) == 0


def test_outside_anchor_places_labels_beyond_the_rim(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside")
    assert all(_radius(t) > 1.0 for t in ax.texts)


def test_outside_labels_sit_on_the_wedge_mid_ray(df):
    """The label leaves the disc along its own wedge's mid-ray."""
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside", offset=3.0)
    for record, text in zip(ax._publiplots_wedge_meta.wedges, ax.texts):
        mid = (record.theta0 + record.theta1) / 2.0
        tx, ty = text.get_position()
        assert math.degrees(math.atan2(ty, tx)) % 360.0 == pytest.approx(
            mid % 360.0, abs=1e-6)
        assert math.hypot(tx, ty) > record.r_outer


@pytest.mark.parametrize("leader", [None, "radial"])
def test_the_outside_gap_is_the_requested_millimetres_at_any_chart_size(
    df, leader,
):
    """`offset` is millimetres on the page, not data units, and not a
    fraction that drifts with the chart.

    A data unit is worth a different number of millimetres on every chart
    size and on every cell shape, so a gap converted any other way would
    come out right on at most one of these. The non-square cells matter
    because ``aspect="cell"`` stretches the limits on the longer side.
    With no leader the label's nearest point is ``offset`` from the rim;
    a radial leader is ``offset`` long, with ``leader_gap`` between its
    end and the label.
    """
    for size in [(25.0, 25.0), (40.0, 40.0), (80.0, 80.0),
                 (70.0, 40.0), (40.0, 70.0)]:
        ax = _pie(df, size=size)
        pp.annotate(ax, "wedge_values", anchor="outside", offset=3.0,
                    leader=leader)
        gaps = _rim_gaps_mm(ax)
        assert len(gaps) == 4, size
        for gap in gaps:
            assert gap == pytest.approx(3.0, abs=1e-3), size
        if leader:
            for gap in _end_gaps_mm(ax):
                assert gap == pytest.approx(pp.rcParams["pie.leader_gap"],
                                            abs=1e-3), size


def test_the_outside_gap_tracks_the_requested_millimetres(df):
    leader_gap = pp.rcParams["pie.leader_gap"]
    for offset in [0.5, 2.0, 5.0]:
        ax = _pie(df)
        pp.annotate(ax, "wedge_values", anchor="outside", offset=offset)
        for gap in _rim_gaps_mm(ax):
            assert gap == pytest.approx(offset, abs=1e-3)
        for gap in _end_gaps_mm(ax):
            assert gap == pytest.approx(leader_gap, abs=1e-3)


def test_a_bigger_offset_pushes_outside_labels_further(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside", offset=1.0)
    near = [_radius(t) for t in ax.texts]
    ax2 = _pie(df)
    pp.annotate(ax2, "wedge_values", anchor="outside", offset=6.0)
    far = [_radius(t) for t in ax2.texts]
    assert all(f > n for f, n in zip(far, near))


def test_elbow_labels_face_away_from_the_centre(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside", leader="elbow")
    for text in ax.texts:
        tx, _ = text.get_position()
        expected = "left" if tx >= 0 else "right"
        assert text.get_ha() == expected
        assert text.get_va() == "center"


@pytest.mark.parametrize("leader", ["radial", None])
def test_radial_and_unled_labels_are_centre_aligned(df, leader):
    """Centred on the mid-ray, not hung off one side of it."""
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside", leader=leader)
    for text in ax.texts:
        assert (text.get_ha(), text.get_va()) == ("center", "center")


def test_outside_anchor_draws_a_leader_line(df):
    ax = _pie(df)
    before = len(ax.lines)
    pp.annotate(ax, "wedge_values", anchor="outside")
    assert len(ax.lines) == before + len(ax._publiplots_wedge_meta.wedges)
    assert ax.lines[-1].get_linewidth() == pytest.approx(
        pp.rcParams["edgewidth"])


def test_a_leader_line_spans_rim_to_offset_on_the_mid_ray(df):
    """Each radial leader starts on its wedge's rim, runs ``offset`` mm out
    along the mid-ray, and stops short of its label.

    It used to end exactly at the label's anchor point, touching the text.
    """
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside", offset=3.0)
    renderer = _settle(ax.figure)
    records = ax._publiplots_wedge_meta.wedges
    for record, text, line in zip(records, ax.texts, ax.lines):
        xs, ys = line.get_xdata(), line.get_ydata()
        assert len(xs) == 2
        assert math.hypot(xs[0], ys[0]) == pytest.approx(record.r_outer)
        start, end = ax.transData.transform(list(zip(xs, ys)))
        assert math.dist(start, end) * _px_to_mm(ax.figure) == pytest.approx(
            3.0, abs=1e-3)
        mid = (record.theta0 + record.theta1) / 2.0
        assert math.degrees(math.atan2(ys[1], xs[1])) % 360.0 == pytest.approx(
            mid % 360.0, abs=1e-6)
        assert not text.get_window_extent(renderer).contains(*end)


def test_a_leader_line_takes_its_own_wedges_edge_colour(df):
    """A leader continues its wedge's outline, so it matches that edge.

    Reading the themed ``edgecolor`` instead would hand matplotlib a
    ``None`` under the default theme, and each leader would come off the
    property cycle in a colour belonging to no wedge on the chart.
    """
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside")
    for record, line in zip(ax._publiplots_wedge_meta.wedges, ax.lines):
        assert tuple(line.get_color()) == pytest.approx(
            tuple(record.patch.get_edgecolor()))


def test_leader_lines_do_not_all_share_one_colour(df):
    """The default palette gives each wedge its own edge, so leaders differ."""
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside")
    colours = {tuple(line.get_color()) for line in ax.lines}
    assert len(colours) == len(ax.lines)


def test_an_explicit_edgecolor_reaches_the_leader_lines(df):
    from matplotlib.colors import to_rgba

    ax = _pie(df, edgecolor="#123456")
    pp.annotate(ax, "wedge_values", anchor="outside")
    for line in ax.lines:
        assert tuple(line.get_color()) == pytest.approx(to_rgba("#123456"))


def test_inside_labels_draw_no_leader_line(df):
    ax = _pie(df)
    before = len(ax.lines)
    pp.annotate(ax, "wedge_values", anchor="inside")
    assert len(ax.lines) == before


def test_auto_pushes_a_thin_wedge_outside():
    thin = pd.DataFrame({"kind": list("AB"), "n": [99.5, 0.5]})
    ax = _pie(thin)
    pp.annotate(ax, "wedge_values", anchor="auto")
    radii = [_radius(t) for t in ax.texts]
    assert radii[0] < 1.0, "the wide wedge keeps its label inside"
    assert radii[1] > 1.0, "the thin wedge is pushed outside"


def test_auto_draws_a_leader_only_for_the_wedge_it_moved():
    thin = pd.DataFrame({"kind": list("AB"), "n": [99.5, 0.5]})
    ax = _pie(thin)
    before = len(ax.lines)
    pp.annotate(ax, "wedge_values", anchor="auto")
    assert len(ax.lines) == before + 1


def test_offset_reaches_a_label_auto_pushed_out():
    """The flip path must honour `offset`, not a constant of its own."""
    thin = pd.DataFrame({"kind": list("AB"), "n": [99.5, 0.5]})
    near = _pie(thin)
    pp.annotate(near, "wedge_values", anchor="auto", offset=1.0)
    far = _pie(thin)
    pp.annotate(far, "wedge_values", anchor="auto", offset=6.0)
    assert _radius(far.texts[1]) > _radius(near.texts[1])


def test_auto_is_the_default_anchor():
    thin = pd.DataFrame({"kind": list("AB"), "n": [99.5, 0.5]})
    ax = _pie(thin)
    pp.annotate(ax, "wedge_values")
    radii = [_radius(t) for t in ax.texts]
    assert radii[0] < 1.0
    assert radii[1] > 1.0


def test_auto_keeps_a_label_inside_that_a_bigger_chart_can_hold():
    """The fit decision follows the chart's physical size, not its angles.

    The same 36-degree wedge holds "10.0%" at 80mm and cannot at 25mm.
    A threshold on the angular span alone could not tell these apart.
    """
    frame = pd.DataFrame({"kind": list("ABCD"), "n": [50.0, 25.0, 15.0, 10.0]})
    big = _pie(frame, size=(80.0, 80.0))
    pp.annotate(big, "wedge_values", anchor="auto")
    small = _pie(frame, size=(25.0, 25.0))
    pp.annotate(small, "wedge_values", anchor="auto")
    assert _radius(big.texts[-1]) < 1.0
    assert _radius(small.texts[-1]) > 1.0


def test_auto_keeps_a_short_label_inside_a_wedge_a_long_one_leaves():
    """The fit decision follows the label's length too."""
    frame = pd.DataFrame({"kind": list("ABCD"), "n": [50.0, 25.0, 15.0, 10.0]})
    short = _pie(frame, size=(40.0, 40.0))
    pp.annotate(short, "wedge_values", anchor="auto", fmt=".0f")
    long = _pie(frame, size=(40.0, 40.0))
    pp.annotate(long, "wedge_values", anchor="auto", fmt=",.4f")
    assert _radius(short.texts[-1]) < 1.0
    assert _radius(long.texts[-1]) > 1.0


# -----------------------------------------------------------------------------
# Outside labels are decorations
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(38.0, 38.0), (40.0, 70.0), (70.0, 40.0)])
def test_the_disc_is_the_same_size_whichever_anchor_its_labels_take(size):
    """The same data draws the same pie, wherever its labels went.

    The gallery's comparison: one figure, the same frame labelled
    ``inside``, ``outside`` and ``auto`` beside a bare copy. Labels that
    leave the disc are decorations the layout makes room for, so none of
    them may shrink the chart to keep itself inside the frame. Tall and
    wide cells too, since ``aspect="cell"`` frames those by stretching
    the limits on the longer side.
    """
    thin = pd.DataFrame({"kind": list("ABCD"), "n": [80.0, 12.0, 6.0, 2.0]})
    fig, axes = pp.subplots(ncols=4, axes_size=size)
    for ax in axes:
        pp.pieplot(data=thin, x="kind", y="n", legend=False, ax=ax)
    for ax, anchor in zip(axes[1:], ["inside", "outside", "auto"]):
        pp.annotate(ax, "wedge_values", anchor=anchor)

    inside, outside, auto = [sum(_radius(t) > 1.0 for t in ax.texts)
                             for ax in axes[1:]]
    assert (inside, outside) == (0, 4)
    assert 0 < auto < 4, "auto must move some labels, not all"

    bare = axes[0]
    for ax in axes[1:]:
        assert ax.get_xlim() == pytest.approx(bare.get_xlim(), abs=1e-12)
        assert ax.get_ylim() == pytest.approx(bare.get_ylim(), abs=1e-12)
    diameters = [_disc_diameter_mm(ax) for ax in axes]
    for diameter in diameters[1:]:
        assert diameter == pytest.approx(diameters[0], abs=1e-6)


def test_aspect_equal_limits_are_left_alone_too(df):
    fig, axes = pp.subplots(ncols=2, axes_size=(70.0, 40.0))
    for ax in axes:
        pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax,
                   aspect="equal")
    pp.annotate(axes[1], "wedge_values", anchor="outside", offset=4.0)
    assert axes[1].get_xlim() == pytest.approx(axes[0].get_xlim(), abs=1e-12)
    assert axes[1].get_ylim() == pytest.approx(axes[0].get_ylim(), abs=1e-12)


def test_the_axes_keeps_its_size_and_the_figure_holds_every_label(df):
    """The layout grows the figure, never the axes, to fit the labels.

    A non-square cell as well as a square one, since ``aspect="cell"``
    frames those two differently.
    """
    for size in [(40.0, 40.0), (60.0, 40.0)]:
        fig, ax = pp.subplots(axes_size=size)
        pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
        pp.annotate(ax, "wedge_values", anchor="outside", offset=3.0)
        renderer = _settle(fig)
        frame = ax.get_window_extent()
        to_mm = _px_to_mm(fig)
        assert (frame.width * to_mm, frame.height * to_mm) == pytest.approx(
            size, abs=1e-6)
        page = fig.bbox
        reaching_out = 0
        for artist in _decorations(ax):
            box = artist.get_window_extent(renderer)
            assert page.x0 <= box.x0 and box.x1 <= page.x1, (size, artist)
            assert page.y0 <= box.y0 and box.y1 <= page.y1, (size, artist)
            reaching_out += not (frame.x0 <= box.x0 and box.x1 <= frame.x1
                                 and frame.y0 <= box.y0 and box.y1 <= frame.y1)
        assert reaching_out > 0, "nothing reached past the frame to test"


def test_an_extreme_offset_leaves_the_limits_alone(df):
    """A gap and labels far wider than the chart grow the figure instead.

    Under limit-growing framing this request had no finite answer -- a
    10mm gap plus nine-character labels on a 25mm chart -- and needed a
    ceiling to stop it. As decorations it is just a wider figure: the
    limits are what ``pieplot`` set, the gap is exact, and every label is
    on the page.
    """
    fig, axes = pp.subplots(ncols=2, axes_size=(25.0, 25.0))
    for ax in axes:
        pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax)
    ax = axes[1]
    pp.annotate(ax, "wedge_values", anchor="outside", offset=10.0,
                fmt=",.2f")
    # Each cell frames itself from its own millimetre ratio, so two cells
    # of one size can disagree in the last bit or two.
    assert ax.get_xlim() == pytest.approx(axes[0].get_xlim(), abs=1e-12)
    assert ax.get_ylim() == pytest.approx(axes[0].get_ylim(), abs=1e-12)
    for gap in _rim_gaps_mm(ax):
        assert gap == pytest.approx(10.0, abs=1e-3)
    for gap in _end_gaps_mm(ax):
        assert gap == pytest.approx(pp.rcParams["pie.leader_gap"], abs=1e-3)
    renderer = _settle(fig)
    page = fig.bbox
    for artist in _decorations(ax):
        box = artist.get_window_extent(renderer)
        assert page.x0 <= box.x0 and box.x1 <= page.x1
        assert page.y0 <= box.y0 and box.y1 <= page.y1


def test_leader_lines_and_labels_are_not_clipped_at_the_frame():
    """A leader runs from the rim, inside the frame, to a label beyond it.

    ``Line2D`` clips to the axes by default, which would cut every leader
    off at the frame. Checked on the rendered pixels: with the leader in
    pure red, red ink must appear between the frame and the label.
    """
    import numpy as np

    halves = pd.DataFrame({"kind": list("AB"), "n": [50.0, 50.0]})
    fig, ax = pp.subplots(axes_size=(40.0, 40.0))
    pp.pieplot(data=halves, x="kind", y="n", legend=False, ax=ax,
               edgecolor="#ff0000")
    pp.annotate(ax, "wedge_values", anchor="outside", offset=3.0)
    for artist in _decorations(ax):
        assert artist.get_clip_on() is False, artist

    renderer = _settle(fig)
    # The first draw renders before the layout resizes the figure, so its
    # buffer does not match the settled positions; the second one does.
    fig.canvas.draw()
    frame = ax.get_window_extent()
    label = ax.texts[0].get_window_extent(renderer)   # A, at three o'clock
    assert label.x0 > frame.x1 + 2, "the label must sit past the frame"
    pixels = np.asarray(fig.canvas.buffer_rgba())
    height = pixels.shape[0]
    row = height - int(round((label.y0 + label.y1) / 2.0))
    strip = pixels[row - 2:row + 3,
                   int(math.ceil(frame.x1)) + 1:int(math.floor(label.x0)) - 1]
    red = ((strip[..., 0] > 200) & (strip[..., 1] < 80)
           & (strip[..., 2] < 80))
    assert red.any(), "the leader was cut off at the axes frame"


def test_a_row_of_pies_keeps_outside_labels_off_their_neighbours():
    """The layout reserves the room between cells, not just at the edges.

    Every outside label and leader in a 1x3 row must stay clear of every
    *other* axes -- and of the other cells' labels. ``offset=4`` so the
    labels reach sideways well past the 3mm the layout leaves between
    cells that have no decorations: 3.94mm past the label's own frame.
    The default 2mm offset reaches only 2.40mm, and a 1mm one 1.63mm,
    inside that 3mm, where the row would pass with no reservation at all.
    """
    thin = pd.DataFrame({"kind": list("ABCD"), "n": [80.0, 12.0, 6.0, 2.0]})
    fig, axes = pp.subplots(ncols=3, axes_size=(38.0, 38.0))
    for ax in axes:
        pp.pieplot(data=thin, x="kind", y="n", legend=False, ax=ax)
        pp.annotate(ax, "wedge_values", anchor="outside", offset=4.0)
    renderer = _settle(fig)
    to_mm = _px_to_mm(fig)
    overhang = max(
        max(own.x0 - box.x0, box.x1 - own.x1) * to_mm
        for ax in axes
        for own in [ax.get_window_extent()]
        for box in (a.get_window_extent(renderer) for a in _decorations(ax))
    )
    assert overhang > 3.0, "labels must reach past the bare inter-cell gap"
    for i, ax in enumerate(axes):
        mine = [a.get_window_extent(renderer) for a in _decorations(ax)]
        for j, other in enumerate(axes):
            if i == j:
                continue
            frame = other.get_window_extent()
            theirs = [a.get_window_extent(renderer)
                      for a in _decorations(other)]
            for box in mine:
                assert not _boxes_overlap(box, frame), (i, j)
                for their in theirs:
                    assert not _boxes_overlap(box, their), (i, j)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="The per-axes legend is pinned 2mm from the axes frame rather "
           "than past the decorations on that side, so an outside label on "
           "the right of the pie overprints it.",
)
def test_outside_labels_do_not_overlap_the_default_legend():
    """`legend=True` puts the legend to the right of the axes, where the
    labels of wedges pointing right now are.

    Measured after a full ``savefig``, which settles the layout, so a fix
    that needs more than one draw to place the legend is still seen. The
    setup check uses ``pytest.fail`` rather than ``assert`` so that only
    the overlap assertion can be the expected failure.
    """
    import io

    import matplotlib.legend

    frame = pd.DataFrame({"kind": list("AB"), "n": [30.0, 70.0]})
    fig, ax = pp.subplots(axes_size=(40.0, 40.0))
    pp.pieplot(data=frame, x="kind", y="n", ax=ax,
               annotate={"anchor": "outside"})
    fig.savefig(io.BytesIO(), format="png")
    renderer = fig.canvas.get_renderer()
    legends = fig.findobj(matplotlib.legend.Legend)
    if len(legends) != 1:
        pytest.fail(f"expected one legend, found {len(legends)}")
    legend = legends[0].get_window_extent(renderer)
    for artist in _decorations(ax):
        assert not _boxes_overlap(artist.get_window_extent(renderer),
                                  legend), artist


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="matplotlib places an axes title against the axes frame rather "
           "than above the decorations, so an outside label near twelve "
           "o'clock runs into it.",
)
def test_outside_labels_do_not_overlap_the_axes_title():
    """The top label of a pie whose last wedge ends at twelve o'clock
    climbs into the title as ``offset`` grows.

    The 2% wedge sits just left of twelve o'clock; at ``offset=3`` its
    label, centred on the mid-ray past a radial leader, runs 3.62mm into
    the title on a 38x38mm cell. Measured after a
    full ``savefig``, as the legend case is, and the setup check uses
    ``pytest.fail`` so that only the overlap assertion can be the
    expected failure.
    """
    import io

    frame = pd.DataFrame({"kind": list("ABCD"), "n": [80.0, 12.0, 6.0, 2.0]})
    fig, ax = pp.subplots(axes_size=(38.0, 38.0))
    pp.pieplot(data=frame, x="kind", y="n", legend=False, ax=ax,
               annotate={"anchor": "outside", "offset": 3})
    ax.set_title("outside")
    fig.savefig(io.BytesIO(), format="png")
    renderer = fig.canvas.get_renderer()
    if len(ax.texts) != 4 or not ax.get_title():
        pytest.fail("expected four labels and a title")
    title = ax.title.get_window_extent(renderer)
    for artist in _decorations(ax):
        assert not _boxes_overlap(artist.get_window_extent(renderer),
                                  title), artist


def test_a_narrow_ring_cannot_hold_its_labels(df):
    """The radial half of the fit test carries its own weight.

    A wide ring holds every label; a ring thinner than the label is tall
    holds none -- including the 50% wedge, which has all the angular room
    in the world and is rejected on ring width alone. Nothing in the
    tangential half of the test can tell these two charts apart.
    """
    wide = _pie(df, donut=0.6)
    pp.annotate(wide, "wedge_values", anchor="auto")
    assert all(_radius(t) < 1.0 for t in wide.texts)
    assert len(wide.lines) == 0

    narrow = _pie(df, donut=0.9)
    pp.annotate(narrow, "wedge_values", anchor="auto")
    assert all(_radius(t) > 1.0 for t in narrow.texts)
    assert len(narrow.lines) == len(narrow.texts)


def test_a_donut_labels_the_middle_of_its_ring(df):
    ax = _pie(df, donut=0.6)
    pp.annotate(ax, "wedge_values", anchor="inside")
    for text in ax.texts:
        assert _radius(text) == pytest.approx(0.8)


def test_a_pulled_wedge_is_labelled_at_its_pulled_centre(df):
    ax = _pie(df, pull={"A": 0.3})
    pp.annotate(ax, "wedge_values", anchor="inside")
    record = ax._publiplots_wedge_meta.wedges[0]
    mid = math.radians((record.theta0 + record.theta1) / 2.0)
    cx, cy = record.patch.center
    expected = (cx + 0.5 * math.cos(mid), cy + 0.5 * math.sin(mid))
    assert ax.texts[0].get_position() == pytest.approx(expected)


# -----------------------------------------------------------------------------
# aspect="equal"
# -----------------------------------------------------------------------------

def test_aspect_equal_honours_offset_in_both_orientations(df):
    """Under ``aspect="equal"`` the drawn box is a square inside the cell.

    matplotlib only shrinks it on the first draw, so a scale read before
    then is the full cell's -- right on a cell wider than tall, and off by
    the cell's own ratio on one taller than wide, where a 3mm gap would
    render at 3 * 40/70mm. The strategy applies the aspect before it
    converts, so both orientations come out exact.
    """
    for size in [(70.0, 40.0), (40.0, 70.0)]:
        ax = _pie(df, size=size, aspect="equal")
        pp.annotate(ax, "wedge_values", anchor="outside", offset=3.0)
        gaps = _rim_gaps_mm(ax)
        assert len(gaps) == 4, size
        for gap in gaps:
            assert gap == pytest.approx(3.0, abs=1e-3), size
        for gap in _end_gaps_mm(ax):
            assert gap == pytest.approx(pp.rcParams["pie.leader_gap"],
                                        abs=1e-3), size


def test_aspect_equal_on_a_tall_cell_keeps_labels_on_the_page_and_off_neighbours(
    df,
):
    """The tall ``aspect="equal"`` cell used to put a label box outside
    the frame and over whatever sat beside it. As decorations, those
    labels are reserved for like any other, in a row of such cells too.
    """
    fig, axes = pp.subplots(ncols=3, axes_size=(40.0, 70.0))
    for ax in axes:
        pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax,
                   aspect="equal")
        pp.annotate(ax, "wedge_values", anchor="outside", offset=2.0)
    renderer = _settle(fig)
    page = fig.bbox
    for i, ax in enumerate(axes):
        for artist in _decorations(ax):
            box = artist.get_window_extent(renderer)
            assert page.x0 <= box.x0 and box.x1 <= page.x1
            assert page.y0 <= box.y0 and box.y1 <= page.y1
            for j, other in enumerate(axes):
                if i != j:
                    assert not _boxes_overlap(box, other.get_window_extent())


def test_aspect_equal_is_labelled_like_a_cell_framed_pie(df):
    """The other framing gets labelled too, and lands in the same places.

    A square chart only: there the two scales are equal by construction,
    so this is a smoke test of the framing, not of the geometry.
    ``test_aspect_equal_honours_offset_in_both_orientations`` covers that.
    """
    equal = _pie(df, aspect="equal")
    pp.annotate(equal, "wedge_values", anchor="inside")
    cell = _pie(df, aspect="cell")
    pp.annotate(cell, "wedge_values", anchor="inside")
    assert [t.get_text() for t in equal.texts] == [t.get_text()
                                                   for t in cell.texts]
    for a, b in zip(equal.texts, cell.texts):
        assert a.get_position() == pytest.approx(b.get_position())


def test_aspect_equal_outside_labels_clear_the_rim_and_get_leaders(df):
    ax = _pie(df, aspect="equal")
    pp.annotate(ax, "wedge_values", anchor="outside")
    assert all(_radius(t) > 1.0 for t in ax.texts)
    assert len(ax.lines) == len(ax.texts)


# -----------------------------------------------------------------------------
# Leader lines
# -----------------------------------------------------------------------------
#
# Every geometry check here reads the rendered figure after a full
# `savefig`, in millimetres, so it sees what the page shows once the layout
# has settled -- not the data-unit arithmetic the strategy did.

GAP_TOL_MM = 0.01

# The gallery's first figure: two wide wedges at three and nine o'clock and
# two thin ones either side of twelve, the case reported as hanging off to
# one side of its wedge.
GALLERY = pd.DataFrame({"kind": ["symptom", "admin", "lay_dx", "none"],
                        "n": [52.2, 41.8, 6.0, 4.0]})


@pytest.fixture
def restore_pie_rcparams():
    """Put back every ``pie.*`` leader rcParam a test changes."""
    keys = ["pie.leader", "pie.label_offset", "pie.leader_gap",
            "pie.elbow_length"]
    saved = {key: pp.rcParams[key] for key in keys}
    yield
    for key, value in saved.items():
        pp.rcParams[key] = value


def _saved_pie(frame=GALLERY, size=(45.0, 45.0), **annotate_kws):
    """A pie with every label outside, saved, with the renderer it drew."""
    import io

    fig, ax = pp.subplots(axes_size=size)
    pp.pieplot(data=frame, x="kind", y="n", legend=False, ax=ax)
    annotate_kws.setdefault("anchor", "outside")
    pp.annotate(ax, "wedge_values", **annotate_kws)
    fig.savefig(io.BytesIO(), format="png")
    return ax, _settle(fig)


def _mm(ax, px):
    return px * _px_to_mm(ax.figure)


ROTATIONS = [0.0, "tangential", "radial", 30.0]


# Mid-ray angles, anticlockwise from three o'clock: twelve exactly and just
# off it, half past one, three, half past four, and one in each other
# quadrant. `_pointing` centres a wedge on each, and the other wedge lands
# opposite it, so both halves of the dial are covered.
ANGLES = [90.0, 84.0, 45.0, 0.0, -45.0, 135.0, 200.0, 250.0]


def _pointing(angle, size=(40.0, 40.0), **annotate_kws):
    """A two-wedge pie whose 10% wedge is centred on ``angle``, saved."""
    import io

    frame = pd.DataFrame({"kind": list("AB"), "n": [10.0, 90.0]})
    fig, ax = pp.subplots(axes_size=size)
    # Clockwise from the start: the 36-degree wedge's mid-ray is 18
    # degrees short of where it starts.
    pp.pieplot(data=frame, x="kind", y="n", legend=False, ax=ax,
               start_angle=angle + 18.0)
    annotate_kws.setdefault("anchor", "outside")
    pp.annotate(ax, "wedge_values", **annotate_kws)
    fig.savefig(io.BytesIO(), format="png")
    record = ax._publiplots_wedge_meta.wedges[0]
    mid = ((record.theta0 + record.theta1) / 2.0 - angle) % 360.0
    assert min(mid, 360.0 - mid) == pytest.approx(0.0, abs=1e-9)
    return ax, _settle(fig)


def _assert_end_gaps(ax, renderer, expected):
    pairs = _outside_records(ax)
    assert pairs and len(pairs) == len(ax.lines)
    for (record, text), line in zip(pairs, ax.lines):
        end = _display(ax, line.get_xdata(), line.get_ydata())[-1]
        gap = _box_distance_mm(ax, renderer, text, end)
        assert gap == pytest.approx(expected, abs=GAP_TOL_MM), (
            text.get_text(), (record.theta0 + record.theta1) / 2.0)


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("leader_gap", [None, 0.3, 2.5])
def test_a_radial_leader_ends_leader_gap_short_of_its_label_at_any_angle(
    angle, leader_gap,
):
    """The visible gap from the leader's end to the label is ``leader_gap``.

    The shortest straight-line distance to the rendered box, at every
    angle. Placing the box by its reach along the mid-ray instead left
    labels off the axes -- half past one, say -- visibly further from
    their lines than those at three o'clock; and the leader used to end
    on the label's anchor point, a gap of zero.
    """
    ax, renderer = _pointing(angle, leader="radial", leader_gap=leader_gap)
    _assert_end_gaps(ax, renderer, pp.rcParams["pie.leader_gap"]
                     if leader_gap is None else leader_gap)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_rotated_radial_label_keeps_the_gap_to_its_rotated_box(rotation):
    """The gap is to the box around the rotated label, on the gallery frame."""
    ax, renderer = _saved_pie(leader="radial", rotation=rotation)
    _assert_end_gaps(ax, renderer, pp.rcParams["pie.leader_gap"])


@pytest.mark.parametrize("angle", ANGLES)
def test_no_leader_puts_the_label_offset_from_the_rim_at_any_angle(angle):
    ax, renderer = _pointing(angle, leader=None, offset=3.0)
    assert len(ax.lines) == 0
    pairs = _outside_records(ax)
    assert len(pairs) == 2
    for record, text in pairs:
        gap = _box_distance_mm(ax, renderer, text, _rim_display(ax, record))
        assert gap == pytest.approx(3.0, abs=GAP_TOL_MM), text.get_text()


@pytest.mark.parametrize("direction", [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0),
                                       (0.0, -1.0)])
@pytest.mark.parametrize("gap", [0.0, 0.8])
def test_centre_distance_on_an_axis_is_the_half_size_plus_the_gap(
    direction, gap,
):
    """Exactly along an axis one component is zero, and must not divide."""
    from publiplots.annotate._wedge import _centre_distance

    half_w, half_h = 3.0, 1.0
    expected = (half_w if direction[0] else half_h) + gap
    assert _centre_distance(half_w, half_h, direction, gap) == pytest.approx(
        expected, abs=1e-12)


@pytest.mark.parametrize("degrees", [1.0, 10.0, 18.4, 30.0, 45.0, 60.0,
                                     71.6, 85.0, 120.0, 225.0, 300.0])
@pytest.mark.parametrize("gap", [0.0, 0.3, 0.8, 5.0])
def test_centre_distance_puts_the_box_gap_from_the_point(degrees, gap):
    """The closed form against the definition, over side and corner cases.

    18.4 and 71.6 degrees sit near the switch between a side and the
    corner for a 3:1 box at these gaps.
    """
    from publiplots.annotate._wedge import _centre_distance

    half_w, half_h = 3.0, 1.0
    dx, dy = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    t = _centre_distance(half_w, half_h, (dx, dy), gap)

    def dist(t):
        return math.hypot(max(abs(t * dx) - half_w, 0.0),
                          max(abs(t * dy) - half_h, 0.0))

    assert dist(t) == pytest.approx(gap, abs=1e-9)
    # The smallest such t: any nearer and the box is closer than the gap
    # (or, for a zero gap, still covers the point).
    nearer = dist(t - 1e-6)
    assert nearer < gap or (gap == 0.0 and nearer == 0.0)


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("leader", ["radial", None])
def test_radial_and_unled_label_boxes_are_centred_on_the_mid_ray(
    leader, rotation,
):
    """The rendered box's centre lies on its wedge's mid-ray.

    The old layout hung each label off to the left or right of its point,
    so a label near twelve o'clock sat beside its wedge with the line
    meeting it at a corner.
    """
    ax, renderer = _saved_pie(leader=leader, rotation=rotation)
    for record, text in _outside_records(ax):
        box = text.get_window_extent(renderer)
        cx, cy = ax.transData.transform(record.patch.center)
        dx, dy = _mid_ray(record)
        bx, by = (box.x0 + box.x1) / 2.0 - cx, (box.y0 + box.y1) / 2.0 - cy
        assert _mm(ax, abs(-bx * dy + by * dx)) == pytest.approx(
            0.0, abs=GAP_TOL_MM), text.get_text()
        assert bx * dx + by * dy > 0, "on the ray, not behind the centre"


@pytest.mark.parametrize("rotation", [0.0, "tangential"])
def test_no_leader_puts_the_label_offset_from_the_rim_and_draws_nothing(
    rotation,
):
    ax, renderer = _saved_pie(leader=None, offset=3.0, rotation=rotation)
    assert len(ax.lines) == 0
    pairs = _outside_records(ax)
    assert len(pairs) == 4
    for record, text in pairs:
        gap = _box_distance_mm(ax, renderer, text, _rim_display(ax, record))
        assert gap == pytest.approx(3.0, abs=GAP_TOL_MM), text.get_text()


def test_leader_false_is_leader_none():
    ax_false, _ = _saved_pie(leader=False)
    ax_none, _ = _saved_pie(leader=None)
    assert len(ax_false.lines) == 0
    for a, b in zip(ax_false.texts, ax_none.texts):
        assert a.get_position() == pytest.approx(b.get_position())
        assert (a.get_ha(), a.get_va()) == (b.get_ha(), b.get_va())


def test_leader_false_through_pieplot_annotate_dict():
    fig, ax = pp.subplots(axes_size=(40.0, 40.0))
    pp.pieplot(data=GALLERY, x="kind", y="n", legend=False, ax=ax,
               annotate={"anchor": "outside", "leader": False})
    assert len(ax.texts) == 4
    assert len(ax.lines) == 0


@pytest.mark.parametrize("elbow_length", [None, 0.5, 4.0])
def test_an_elbow_runs_out_then_horizontally_for_elbow_length(elbow_length):
    """One polyline: rim, elbow ``offset`` out on the mid-ray, then a
    horizontal tail of ``elbow_length`` toward the label's side."""
    ax, renderer = _saved_pie(leader="elbow", offset=2.5,
                              elbow_length=elbow_length)
    expected = (pp.rcParams["pie.elbow_length"] if elbow_length is None
                else elbow_length)
    pairs = _outside_records(ax)
    assert len(pairs) == len(ax.lines) == 4
    for (record, text), line in zip(pairs, ax.lines):
        rim, elbow, tail = _display(ax, line.get_xdata(), line.get_ydata())
        assert rim == pytest.approx(_rim_display(ax, record))
        assert _mm(ax, math.dist(rim, elbow)) == pytest.approx(
            2.5, abs=GAP_TOL_MM)
        dx, dy = _mid_ray(record)
        assert (elbow - rim) @ (dx, dy) > 0, "the first leg is radial"
        assert _mm(ax, abs(tail[1] - elbow[1])) == pytest.approx(
            0.0, abs=1e-9), "the tail is horizontal"
        assert _mm(ax, abs(tail[0] - elbow[0])) == pytest.approx(
            expected, abs=GAP_TOL_MM)
        side = 1.0 if text.get_ha() == "left" else -1.0
        assert (tail[0] - elbow[0]) * side > 0, "the tail runs toward the label"


@pytest.mark.parametrize("rotation", [0.0, "tangential", "radial"])
@pytest.mark.parametrize("leader_gap", [None, 1.5])
def test_an_elbow_label_sits_leader_gap_beside_its_tail(leader_gap, rotation):
    """Left or right of the tail's end by ``leader_gap``, centred on it.

    Rotated labels too: matplotlib aligns the rotated box, so its near
    side keeps the gap and its middle stays level with the tail.
    """
    ax, renderer = _saved_pie(leader="elbow", leader_gap=leader_gap,
                              rotation=rotation)
    expected = (pp.rcParams["pie.leader_gap"] if leader_gap is None
                else leader_gap)
    sides = set()
    for (record, text), line in zip(_outside_records(ax), ax.lines):
        _, _, (tx, ty) = _display(ax, line.get_xdata(), line.get_ydata())
        box = text.get_window_extent(renderer)
        if text.get_ha() == "left":
            gap = box.x0 - tx
        else:
            assert text.get_ha() == "right"
            gap = tx - box.x1
        sides.add(text.get_ha())
        assert _mm(ax, gap) == pytest.approx(expected, abs=GAP_TOL_MM)
        assert _mm(ax, (box.y0 + box.y1) / 2.0 - ty) == pytest.approx(
            0.0, abs=GAP_TOL_MM)
    assert sides == {"left", "right"}, "both sides must be exercised"


def test_an_elbow_on_a_vertical_mid_ray_turns_right():
    """Twelve and six o'clock both turn right.

    ``cos`` of six o'clock is -1.8e-16 rather than 0, and without a
    tolerance that tail would turn left.
    """
    halves = pd.DataFrame({"kind": list("AB"), "n": [50.0, 50.0]})
    fig, ax = pp.subplots(axes_size=(40.0, 40.0))
    pp.pieplot(data=halves, x="kind", y="n", legend=False, ax=ax,
               start_angle=0)
    pp.annotate(ax, "wedge_values", anchor="outside", leader="elbow")
    mids = sorted(round(((w.theta0 + w.theta1) / 2.0) % 360.0)
                  for w in ax._publiplots_wedge_meta.wedges)
    assert mids == [90, 270]
    for text, line in zip(ax.texts, ax.lines):
        xs = line.get_xdata()
        assert text.get_ha() == "left"
        assert xs[2] > xs[1]


def test_an_elbow_is_one_line_with_round_joins_and_butt_caps():
    """Set on the line itself, not inherited: matplotlib's default join is
    already round, so a theme that changes it must not reach the leader."""
    with plt.rc_context({"lines.solid_joinstyle": "miter",
                         "lines.solid_capstyle": "projecting"}):
        ax, _ = _saved_pie(leader="elbow")
    assert len(ax.lines) == 4
    for line in ax.lines:
        assert len(line.get_xdata()) == 3
        assert line.get_solid_joinstyle() == "round"
        assert line.get_solid_capstyle() == "butt"
        assert line.get_clip_on() is False


def test_leaders_keep_their_wedges_colour_width_and_zorder_in_every_style():
    for leader in ["radial", "elbow"]:
        ax, _ = _saved_pie(leader=leader)
        for record, line in zip(ax._publiplots_wedge_meta.wedges, ax.lines):
            assert tuple(line.get_color()) == pytest.approx(
                tuple(record.patch.get_edgecolor()))
            assert line.get_linewidth() == pytest.approx(
                pp.rcParams["edgewidth"])
            assert line.get_zorder() == record.patch.get_zorder() + 1


@pytest.mark.parametrize("leader", ["radial", "elbow", None])
def test_a_pulled_wedge_keeps_its_outside_label(df, leader):
    """Rim, leader and label all move out with the pulled wedge."""
    ax = _pie(df, pull={"A": 0.3})
    pp.annotate(ax, "wedge_values", anchor="outside", leader=leader)
    record = ax._publiplots_wedge_meta.wedges[0]
    assert record.patch.center != pytest.approx((0.0, 0.0))
    cx, cy = record.patch.center
    dx, dy = _mid_ray(record)
    if leader is not None:
        xs, ys = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
        assert (xs[0], ys[0]) == pytest.approx(
            (cx + record.r_outer * dx, cy + record.r_outer * dy))
    if leader != "elbow":
        tx, ty = ax.texts[0].get_position()
        assert -(tx - cx) * dy + (ty - cy) * dx == pytest.approx(0.0,
                                                                 abs=1e-12)


def test_auto_places_a_pushed_out_label_with_the_chosen_leader():
    thin = pd.DataFrame({"kind": list("AB"), "n": [99.5, 0.5]})
    ax = _pie(thin)
    pp.annotate(ax, "wedge_values", anchor="auto", leader="elbow")
    assert len(ax.lines) == 1
    assert len(ax.lines[0].get_xdata()) == 3
    assert ax.texts[0].get_ha() == "center", "the inside label is unchanged"
    assert ax.texts[1].get_ha() in {"left", "right"}


# --- defaults ---------------------------------------------------------------

def test_the_leader_defaults_to_the_rcparam(restore_pie_rcparams):
    ax, _ = _saved_pie()
    assert [len(line.get_xdata()) for line in ax.lines] == [2] * 4
    pp.rcParams["pie.leader"] = "elbow"
    ax, _ = _saved_pie()
    assert [len(line.get_xdata()) for line in ax.lines] == [3] * 4
    pp.rcParams["pie.leader"] = None
    ax, _ = _saved_pie()
    assert len(ax.lines) == 0


def test_a_per_call_leader_overrides_the_rcparam(restore_pie_rcparams):
    pp.rcParams["pie.leader"] = "elbow"
    ax, _ = _saved_pie(leader="radial")
    assert [len(line.get_xdata()) for line in ax.lines] == [2] * 4
    ax, _ = _saved_pie(leader=None)
    assert len(ax.lines) == 0


def _leader_length_mm(ax, line):
    rim, end = _display(ax, line.get_xdata(), line.get_ydata())[:2]
    return _mm(ax, math.dist(rim, end))


def test_offset_defaults_to_the_label_offset_rcparam(restore_pie_rcparams):
    ax, _ = _saved_pie()
    for line in ax.lines:
        assert _leader_length_mm(ax, line) == pytest.approx(2.0,
                                                            abs=GAP_TOL_MM)
    pp.rcParams["pie.label_offset"] = 3.5
    ax, _ = _saved_pie()
    for line in ax.lines:
        assert _leader_length_mm(ax, line) == pytest.approx(3.5,
                                                            abs=GAP_TOL_MM)
    ax, _ = _saved_pie(offset=1.5)
    for line in ax.lines:
        assert _leader_length_mm(ax, line) == pytest.approx(1.5,
                                                            abs=GAP_TOL_MM)


def test_offset_default_reaches_pieplot_annotate_true(restore_pie_rcparams):
    pp.rcParams["pie.label_offset"] = 4.0
    fig, ax = pp.subplots(axes_size=(40.0, 40.0))
    pp.pieplot(data=GALLERY, x="kind", y="n", legend=False, ax=ax,
               annotate=True)
    _settle(fig)
    assert ax.lines, "the thin wedges must be pushed out"
    for line in ax.lines:
        assert _leader_length_mm(ax, line) == pytest.approx(4.0,
                                                            abs=GAP_TOL_MM)


def test_leader_gap_defaults_to_the_rcparam(restore_pie_rcparams):
    pp.rcParams["pie.leader_gap"] = 1.7
    for leader_gap, expected in [(None, 1.7), (0.4, 0.4)]:
        ax, renderer = _saved_pie(leader_gap=leader_gap)
        _assert_end_gaps(ax, renderer, expected)


def test_elbow_length_defaults_to_the_rcparam(restore_pie_rcparams):
    pp.rcParams["pie.elbow_length"] = 3.3
    for elbow_length, expected in [(None, 3.3), (1.1, 1.1)]:
        ax, _ = _saved_pie(leader="elbow", elbow_length=elbow_length)
        for line in ax.lines:
            _, elbow, tail = _display(ax, line.get_xdata(), line.get_ydata())
            assert _mm(ax, abs(tail[0] - elbow[0])) == pytest.approx(
                expected, abs=GAP_TOL_MM)


# --- validation and wiring --------------------------------------------------

@pytest.mark.parametrize("leader", ["curved", True, "Radial"])
def test_an_unknown_leader_raises(df, leader):
    ax = _pie(df)
    with pytest.raises(ValueError, match="leader"):
        pp.annotate(ax, "wedge_values", leader=leader)


def test_an_unknown_leader_rcparam_raises(df, restore_pie_rcparams):
    pp.rcParams["pie.leader"] = "curved"
    ax = _pie(df)
    with pytest.raises(ValueError, match="leader"):
        pp.annotate(ax, "wedge_values")


@pytest.mark.parametrize("name", ["leader_gap", "elbow_length"])
@pytest.mark.parametrize("value", [-0.5, float("nan"), "wide"])
def test_a_bad_length_raises(df, name, value):
    ax = _pie(df)
    with pytest.raises(ValueError, match=name):
        pp.annotate(ax, "wedge_values", **{name: value})


@pytest.mark.parametrize("option", [
    {"leader": "radial"}, {"leader": None}, {"leader": False},
    {"leader_gap": 1.0}, {"elbow_length": 1.0},
])
def test_leader_options_are_rejected_for_other_kinds(option):
    from publiplots.annotate._dispatcher import _STRATEGIES, _WEDGE_KINDS

    (name,) = option
    for kind in sorted(set(_STRATEGIES) - _WEDGE_KINDS):
        fig, ax = pp.subplots()
        with pytest.raises(TypeError, match=f"{name}="):
            pp.annotate(ax, kind, **option)


def test_other_kinds_still_default_offset_and_pad_to_one_mm(monkeypatch):
    """Only the wedge kinds take their offset from an rcParam.

    Every other strategy must still receive ``offset=1.0`` and, through
    the ``pad = offset`` fallback, ``pad=1.0``; an explicit ``offset``
    still becomes the pad.
    """
    from publiplots.annotate import _dispatcher

    others = sorted(set(_dispatcher._STRATEGIES) - _dispatcher._WEDGE_KINDS)
    seen = {}

    def spy(kind):
        def strategy(ax, *, offset, pad, **kws):
            seen[kind] = (offset, pad)
            return []
        return strategy

    for kind in others:
        monkeypatch.setitem(_dispatcher._STRATEGIES, kind, spy(kind))
    fig, ax = pp.subplots()
    for kind in others:
        pp.annotate(ax, kind)
    assert seen == {kind: (1.0, 1.0) for kind in others}
    for kind in others:
        pp.annotate(ax, kind, offset=2.5)
    assert seen == {kind: (2.5, 2.5) for kind in others}


def test_wedge_values_receives_no_offset_when_none_is_passed(monkeypatch):
    """The dispatcher leaves the wedge default to the strategy's rcParam."""
    from publiplots.annotate import _dispatcher

    seen = {}

    def strategy(ax, *, offset, **kws):
        seen["offset"] = offset
        return []

    monkeypatch.setitem(_dispatcher._STRATEGIES, "wedge_values", strategy)
    fig, ax = pp.subplots()
    pp.annotate(ax, "wedge_values")
    assert seen == {"offset": None}


# -----------------------------------------------------------------------------
# basis
# -----------------------------------------------------------------------------

def test_basis_parent_equals_basis_total_on_a_pie(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", basis="parent")
    parent = [t.get_text() for t in ax.texts]
    for text in list(ax.texts):
        text.remove()
    pp.annotate(ax, "wedge_values", basis="total")
    assert parent == [t.get_text() for t in ax.texts]


def test_basis_selects_the_record_field(df):
    """A pie cannot tell the two bases apart, so build a case that can.

    Overwriting `fraction_of_parent` on the cached records is the only
    way to see which field the strategy reads until the sunburst lands.
    """
    from dataclasses import replace

    ax = _pie(df)
    meta = ax._publiplots_wedge_meta
    meta.wedges = [replace(w, fraction_of_parent=0.25) for w in meta.wedges]
    pp.annotate(ax, "wedge_values", basis="parent")
    assert [t.get_text() for t in ax.texts] == ["25.0%"] * 4
    for text in list(ax.texts):
        text.remove()
    pp.annotate(ax, "wedge_values", basis="total")
    assert [t.get_text() for t in ax.texts] == ["50.0%", "25.0%", "15.0%",
                                                "10.0%"]


def test_an_unknown_basis_raises(df):
    ax = _pie(df)
    with pytest.raises(ValueError, match="basis"):
        pp.annotate(ax, "wedge_values", basis="siblings")


def test_basis_is_rejected_for_other_kinds():
    bars = pd.DataFrame({"k": list("AB"), "v": [1.0, 2.0]})
    fig, ax = pp.subplots()
    pp.barplot(data=bars, x="k", y="v", legend=False, ax=ax)
    with pytest.raises(TypeError, match="basis="):
        pp.annotate(ax, "bar_values", basis="total")


# -----------------------------------------------------------------------------
# rotation
# -----------------------------------------------------------------------------

def test_rotation_tangential_is_accepted(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", rotation="tangential")
    assert any(t.get_rotation() % 180 != 0 for t in ax.texts)


def test_rotation_tangential_follows_each_wedges_arc(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", rotation="tangential")
    for record, text in zip(ax._publiplots_wedge_meta.wedges, ax.texts):
        mid = (record.theta0 + record.theta1) / 2.0
        # matplotlib normalises rotation into [0, 360); the label may also
        # have been flipped upright, which is a 180-degree turn.
        assert (text.get_rotation() - (mid - 90.0)) % 180.0 == pytest.approx(
            0.0, abs=1e-6)


def test_rotation_radial_follows_each_wedges_mid_ray(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", rotation="radial")
    for record, text in zip(ax._publiplots_wedge_meta.wedges, ax.texts):
        mid = (record.theta0 + record.theta1) / 2.0
        assert (text.get_rotation() - mid) % 180.0 == pytest.approx(
            0.0, abs=1e-6)


def test_radial_and_tangential_differ_by_a_right_angle(df):
    ax = _pie(df)
    tangential = pp.annotate(ax, "wedge_values", rotation="tangential")
    ax2 = _pie(df)
    radial = pp.annotate(ax2, "wedge_values", rotation="radial")
    for a, b in zip(tangential, radial):
        assert (b.get_rotation() - a.get_rotation()) % 180.0 == pytest.approx(
            90.0, abs=1e-6)


def test_derived_rotations_are_never_upside_down(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", rotation="radial")
    for text in ax.texts:
        angle = (text.get_rotation() + 180.0) % 360.0 - 180.0
        assert -90.0 <= angle <= 90.0


def test_rotation_still_accepts_a_float(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", rotation=45.0)
    assert all(t.get_rotation() == pytest.approx(45.0) for t in ax.texts)


def test_an_explicit_rotation_is_not_flipped_upright(df):
    """120 degrees reads upside-down, and is still the caller's choice."""
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", rotation=120.0)
    assert all(t.get_rotation() == pytest.approx(120.0) for t in ax.texts)


def test_rotation_rejects_an_unknown_string(df):
    ax = _pie(df)
    with pytest.raises(ValueError, match="rotation"):
        pp.annotate(ax, "wedge_values", rotation="sideways")


def test_named_rotations_are_rejected_for_other_kinds():
    bars = pd.DataFrame({"k": list("AB"), "v": [1.0, 2.0]})
    fig, ax = pp.subplots()
    pp.barplot(data=bars, x="k", y="v", legend=False, ax=ax)
    with pytest.raises(ValueError, match="wedge-only"):
        pp.annotate(ax, "bar_values", rotation="tangential")


def test_a_non_finite_rotation_is_still_rejected_for_wedges(df):
    ax = _pie(df)
    with pytest.raises(ValueError, match="finite"):
        pp.annotate(ax, "wedge_values", rotation=float("nan"))


# -----------------------------------------------------------------------------
# colour
# -----------------------------------------------------------------------------

def test_a_literal_colour_is_used_verbatim(df):
    from matplotlib.colors import to_rgba

    ax = _pie(df)
    pp.annotate(ax, "wedge_values", color="#ff0000")
    assert all(t.get_color() == to_rgba("#ff0000") for t in ax.texts)


def test_colour_hue_inherits_each_wedges_fill(df):
    from matplotlib.colors import to_rgba

    ax = _pie(df)
    pp.annotate(ax, "wedge_values", color="hue")
    for record, text in zip(ax._publiplots_wedge_meta.wedges, ax.texts):
        assert text.get_color() == to_rgba(record.hue_color)


def test_a_label_auto_pushed_out_stops_taking_the_fills_contrast():
    """Contrast against a fill is meaningless where there is no fill.

    The fill has to be dark and opaque for the two branches to differ at
    all: at the themed ``alpha`` of 0.1 every wedge composites light over
    the page, so ``'auto'`` returns the plain text colour either way and
    a test built on the default palette proves nothing.
    """
    from matplotlib.colors import to_rgba

    thin = pd.DataFrame({"kind": list("AB"), "n": [99.5, 0.5]})
    ax = _pie(thin, alpha=1.0, palette={"A": "#101010", "B": "#101010"})
    pp.annotate(ax, "wedge_values", anchor="auto", color="auto")
    inside, pushed_out = ax.texts
    assert inside.get_color() == to_rgba("#ffffff")
    assert pushed_out.get_color() == to_rgba(plt.rcParams["text.color"])
    assert inside.get_color() != pushed_out.get_color()


def test_outside_labels_take_the_outside_colour():
    """`anchor="outside"` labels sit on the page, not on a fill.

    Same dark opaque fill as above, so the inside contrast colour would be
    white; an outside label drawn white on the page would be invisible.
    """
    from matplotlib.colors import to_rgba

    frame = pd.DataFrame({"kind": list("AB"), "n": [60.0, 40.0]})
    ax = _pie(frame, alpha=1.0, palette={"A": "#101010", "B": "#101010"})
    pp.annotate(ax, "wedge_values", anchor="outside", color="auto")
    for text in ax.texts:
        assert text.get_color() == to_rgba(plt.rcParams["text.color"])

    inside = _pie(frame, alpha=1.0, palette={"A": "#101010", "B": "#101010"})
    pp.annotate(inside, "wedge_values", anchor="inside", color="auto")
    for text in inside.texts:
        assert text.get_color() == to_rgba("#ffffff"), "the fill must matter"


# -----------------------------------------------------------------------------
# Wiring
# -----------------------------------------------------------------------------

def test_a_foreign_axes_raises_and_names_pieplot():
    fig, ax = pp.subplots()
    ax.pie([1, 2, 3])
    with pytest.raises(ValueError, match="pp.pieplot"):
        pp.annotate(ax, "wedge_values")


def test_an_unknown_anchor_raises(df):
    ax = _pie(df)
    with pytest.raises(ValueError, match="anchor"):
        pp.annotate(ax, "wedge_values", anchor="rim")


def test_annotate_true_on_the_plot_call_labels_every_wedge(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax, annotate=True)
    assert len(ax.texts) == 4


def test_annotate_dict_on_the_plot_call_is_forwarded(df):
    fig, ax = pp.subplots()
    pp.pieplot(data=df, x="kind", y="n", legend=False, ax=ax,
               annotate={"fmt": ",.0f"})
    assert [t.get_text() for t in ax.texts] == ["50", "25", "15", "10"]


def test_a_negative_pad_is_not_rejected_for_wedge_values(df):
    """`pad` has no effect here, so it is not validated here either."""
    ax = _pie(df)
    labels = pp.annotate(ax, "wedge_values", anchor="outside", pad=-2.0)
    assert len(labels) == 4


def test_a_negative_pad_is_still_rejected_for_every_other_kind():
    from publiplots.annotate._dispatcher import _STRATEGIES, _WEDGE_KINDS

    others = sorted(set(_STRATEGIES) - _WEDGE_KINDS)
    assert others, "nothing to check"
    for kind in others:
        fig, ax = pp.subplots()
        with pytest.raises(ValueError, match="pad must be >= 0"):
            pp.annotate(ax, kind, pad=-0.5)


def test_text_kws_reach_the_artists(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", fontweight="bold")
    assert all(t.get_fontweight() == "bold" for t in ax.texts)
