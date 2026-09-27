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
    pp.annotate(ax, "wedge_values", anchor="outside", offset=3.0, pad=0.0)
    for record, text in zip(ax._publiplots_wedge_meta.wedges, ax.texts):
        mid = (record.theta0 + record.theta1) / 2.0
        tx, ty = text.get_position()
        assert math.degrees(math.atan2(ty, tx)) % 360.0 == pytest.approx(
            mid % 360.0, abs=1e-6)
        assert math.hypot(tx, ty) > record.r_outer


def test_a_bigger_offset_pushes_outside_labels_further(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside", offset=1.0, pad=0.0)
    near = [_radius(t) for t in ax.texts]
    ax2 = _pie(df)
    pp.annotate(ax2, "wedge_values", anchor="outside", offset=6.0, pad=0.0)
    far = [_radius(t) for t in ax2.texts]
    assert all(f > n for f, n in zip(far, near))


def test_outside_labels_face_away_from_the_centre(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside")
    for text in ax.texts:
        tx, _ = text.get_position()
        expected = "left" if tx >= 0 else "right"
        assert text.get_ha() == expected


def test_outside_anchor_draws_a_leader_line(df):
    ax = _pie(df)
    before = len(ax.lines)
    pp.annotate(ax, "wedge_values", anchor="outside")
    assert len(ax.lines) == before + len(ax._publiplots_wedge_meta.wedges)
    assert ax.lines[-1].get_linewidth() == pytest.approx(
        pp.rcParams["edgewidth"])


def test_a_leader_line_spans_rim_to_label(df):
    """Each leader must start on its wedge's rim and end at its label."""
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside")
    records = ax._publiplots_wedge_meta.wedges
    for record, text, line in zip(records, ax.texts, ax.lines):
        xs, ys = line.get_xdata(), line.get_ydata()
        assert math.hypot(xs[0], ys[0]) == pytest.approx(record.r_outer)
        assert (xs[1], ys[1]) == pytest.approx(text.get_position())


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
    pp.annotate(near, "wedge_values", anchor="auto", offset=1.0, pad=0.0)
    far = _pie(thin)
    pp.annotate(far, "wedge_values", anchor="auto", offset=6.0, pad=0.0)
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


def test_outside_labels_are_framed_inside_the_axes(df):
    """`pad` earns its name: the framing grows to keep labels on-canvas."""
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", anchor="outside", offset=2.0, pad=1.0)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for text in ax.texts:
        tx, ty = text.get_position()
        assert x0 < tx < x1
        assert y0 < ty < y1


def test_the_framing_stays_square_when_it_grows(df):
    """Growing the limits must not squash the disc into an ellipse."""
    ax = _pie(df)
    before = ((ax.get_xlim()[1] - ax.get_xlim()[0])
              / (ax.get_ylim()[1] - ax.get_ylim()[0]))
    pp.annotate(ax, "wedge_values", anchor="outside", offset=6.0)
    after = ((ax.get_xlim()[1] - ax.get_xlim()[0])
             / (ax.get_ylim()[1] - ax.get_ylim()[0]))
    assert after == pytest.approx(before)


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


def test_text_kws_reach_the_artists(df):
    ax = _pie(df)
    pp.annotate(ax, "wedge_values", fontweight="bold")
    assert all(t.get_fontweight() == "bold" for t in ax.texts)
