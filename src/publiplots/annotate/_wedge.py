"""The ``wedge_values`` annotate strategy: labels for pie and donut wedges.

Placement here is decided by *measuring* the rendered label against the
room its wedge actually has, not by a threshold on the wedge's angular
span. A fixed degree threshold is the wrong shape for this problem: the
room a label needs depends on the chart's physical size, on how long the
label string is, and on where the wedge points. At the themed 7pt font,
across the cases measured in :func:`_label_fits`, the span a label needs
to sit inside runs from 14.3 degrees (``"50.0%"`` at three o'clock on a
40mm chart) to 134.2 degrees (``"1,234,567"`` at twelve o'clock on a
25mm one).

Labels that end up outside their wedge are *decorations*, like a title
or a tick label: they reach past the axes frame, and the layout grows
the figure to hold them rather than the strategy shrinking the chart to
fit them in. So nothing here touches the axis limits, and the disc is
the same size whichever anchor its labels took.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate._cache import WedgeMeta
from publiplots.annotate._color import resolve_color
from publiplots.annotate._positioning import mm_to_data
from publiplots.annotate._shared import (
    ensure_renderer as _ensure_renderer,
)
from publiplots.annotate._shared import (
    format_value as _format_value,
)
from publiplots.themes.rcparams import resolve_param

_VALID_ANCHORS = {"inside", "outside", "auto"}
_DEFAULT_ANCHOR = "auto"
_VALID_BASES = {"total", "parent"}
_DEFAULT_BASIS = "total"
_VALID_ROTATIONS = {"tangential", "radial"}

# Sentinel the dispatcher hands back when the caller passed no `fmt`. A
# real format spec cannot collide with it, and it is what distinguishes
# "label the share as a percentage" (the default) from "format the value
# with this spec".
_WEDGE_DEFAULT_FMT = "__wedge_percent__"


def _meta(ax: Axes) -> WedgeMeta:
    meta = getattr(ax, "_publiplots_wedge_meta", None)
    if meta is None:
        raise ValueError(
            "annotate(kind='wedge_values') requires an axes drawn by "
            "pp.pieplot. There is no introspection fallback for a foreign "
            "ax.pie(): recovering wedges from the axes is the step that "
            "produced the annotate pairing bugs this design avoids."
        )
    return meta


def _data_per_pixel(ax: Axes) -> float:
    """Data units spanned by one display pixel, read off the y axis.

    One scalar describes both directions only when the x and y scales
    agree. ``aspect="cell"`` makes them agree by construction: it matches
    the data limits to the axes' millimetre ratio, which is what makes the
    disc render round there.

    ``aspect="equal"`` makes them agree only once ``apply_aspect`` has
    shrunk the drawn box to a square, and matplotlib otherwise does that
    on the first draw -- after the labels are placed. Until then the box
    is still the full cell against square limits, so the y scale is the
    right one on a cell wider than tall and wrong by the cell's own
    ratio on one taller than wide. The strategy therefore calls
    ``ax.apply_aspect()`` before reading this. Measured on the frame
    ``tests/annotate/test_wedge_values.py::df`` uses (``kind=ABCD``,
    ``n=[50, 25, 15, 10]``) with ``legend=False`` and
    ``annotate={"anchor": "outside", "offset": 3.0}``: without that call
    the rim-to-label gap renders at 3.000mm on a 70x40mm cell and at
    1.714mm -- three times 40/70 -- on a 40x70mm one; with it, at
    3.000mm on both. The call changes nothing under ``aspect="cell"``,
    whose drawn box already is the whole cell.
    """
    inverse = ax.transData.inverted()
    y0 = inverse.transform((0.0, 0.0))[1]
    y1 = inverse.transform((0.0, 1.0))[1]
    return abs(y1 - y0)


def _upright(angle: float) -> float:
    """Keep derived text readable by flipping anything upside-down."""
    angle = (angle + 180.0) % 360.0 - 180.0
    if angle > 90.0:
        return angle - 180.0
    if angle < -90.0:
        return angle + 180.0
    return angle


def _rotation_for(rotation, mid_deg: float) -> float:
    """Resolve the requested rotation for a wedge at ``mid_deg``.

    Flipping upside-down text applies only to the derived orientations.
    An explicit ``rotation=120`` is the caller's choice and is passed
    through untouched.
    """
    if rotation == "tangential":
        return _upright(mid_deg - 90.0)
    if rotation == "radial":
        return _upright(mid_deg)
    return float(rotation)


def _inside_point(record, mid: float) -> Tuple[float, float, str]:
    """Anchor at the middle of the wedge's radial band."""
    cx, cy = record.patch.center
    radius = (record.r_inner + record.r_outer) / 2.0
    return cx + radius * math.cos(mid), cy + radius * math.sin(mid), "center"


def _outside_point(
    record, mid: float, gap: float,
) -> Tuple[float, float, str]:
    """Anchor ``gap`` data units beyond the wedge's rim, along its mid-ray."""
    cx, cy = record.patch.center
    radius = record.r_outer + gap
    ha = "left" if math.cos(mid) >= 0.0 else "right"
    return cx + radius * math.cos(mid), cy + radius * math.sin(mid), ha


def _label_fits(text: Text, record, renderer, scale: float, mid: float) -> bool:
    """Does the rendered label fit inside its own wedge?

    The label is an axis-aligned rectangle once rendered (``get_window_extent``
    already folds in any rotation), so its reach along a unit direction
    ``d`` is the rectangle's support function, ``|w/2 * d.x| + |h/2 * d.y|``.
    Two directions matter: tangential, where the wedge's two edge rays
    close in, and radial, where the ring's inner and outer radii do.

    At the label radius ``r`` the perpendicular distance from the mid-ray
    to either edge ray is ``r * sin(span / 2)``. Past a half-span of 90
    degrees the wedge stops being convex about its mid-ray and the edges
    no longer bound the label at all, so the disc itself is the limit.

    The tangential bound is **approximate, and its error is not small**.
    It resolves the label's reach along the mid-ray's tangent, whereas
    the half-planes that actually contain the wedge have normals
    perpendicular to its two *edge* rays, which the tangent only matches
    as the span goes to zero. In most directions that makes it
    optimistic: it accepts a label with a corner outside its wedge, by an
    amount that grows with the label's width and shrinks as the chart
    gets bigger. Swept over every span it accepts (0.5-degree steps) at
    every whole-degree direction, for a pie with no hole at the default
    rotation, the worst overhang is 0.63mm for ``"50.0%"`` on a 40x40mm
    chart, 1.35mm for ``"1,234,567"`` on that chart, and 1.98mm for
    ``"1,234,567"`` on a 25x25mm one, at a 64-degree span on a mid-ray
    22 degrees off horizontal (22, 158, 202 or 338 degrees, measured
    anticlockwise from three o'clock). That last is the largest over
    ``"5.0%"``, ``"50.0%"``, ``"1,234.57"`` and ``"1,234,567"`` on 25,
    40 and 80mm charts. At three o'clock exactly, the same label on the same chart
    is accepted from a 23.0-degree span and only fits from 136.8; and
    ``"50.0%"`` on 40mm is accepted from 14.3 against a geometric 22.3.
    Only on mid-rays within about 30 degrees of vertical -- 59 to 121
    degrees and 239 to 301 for that label on 25mm, so around twelve and
    six o'clock -- does a long label err the other way: ``"1,234,567"``
    on 25mm is rejected at both twelve and six o'clock below 134.2
    degrees although it fits from 98.0. So an ``auto`` label left inside
    can visibly cross its wedge's edge -- by up to about two millimetres
    with long labels on a small chart -- and ``anchor="outside"`` or a
    shorter ``fmt`` is the remedy there.

    The span a label needs is *not* a single number. Against
    this test at the themed 7pt font, ``"50.0%"`` on a 40x40mm chart
    needs 43.2 degrees at twelve o'clock and 14.3 at three o'clock, where
    the label lies along the radius rather than across it; the same
    label at twelve o'clock needs 72.2 degrees on a 25mm chart and 21.2
    on an 80mm one; and ``"1,234,567"`` at twelve o'clock on that 25mm
    chart needs 134.2.
    That is the range a single constant would have had to stand for,
    which is why this measures instead.
    """
    box = text.get_window_extent(renderer)
    half_w = box.width * scale / 2.0
    half_h = box.height * scale / 2.0

    tangential = (-math.sin(mid), math.cos(mid))
    radial = (math.cos(mid), math.sin(mid))
    reach_t = abs(half_w * tangential[0]) + abs(half_h * tangential[1])
    reach_r = abs(half_w * radial[0]) + abs(half_h * radial[1])

    r_label = (record.r_inner + record.r_outer) / 2.0
    half_span = math.radians(record.theta1 - record.theta0) / 2.0
    if half_span >= math.pi / 2.0:
        room_t = record.r_outer
    else:
        room_t = r_label * math.sin(half_span)
    if reach_t > room_t:
        return False
    return 2.0 * reach_r <= (record.r_outer - record.r_inner)


def _wedge_values_strategy(
    ax: Axes,
    *,
    fmt: str,
    anchor,
    offset: float,
    color,
    pad: float,
    rotation=0.0,
    basis: Optional[str] = None,
    **text_kws,
) -> List[Text]:
    """Label every wedge on ``ax`` with its share, or its formatted value.

    ``anchor="inside"`` centres each label in its wedge's radial band;
    ``"outside"`` puts it ``offset`` millimetres beyond the rim along the
    wedge's mid-ray, joined to the rim by a leader line in the wedge's
    own edge colour at ``edgewidth``; ``"auto"`` measures each label
    inside (:func:`_label_fits`) and moves only the ones that do not fit.

    The axis limits are never changed here -- they are whatever
    :func:`publiplots.pieplot` framed, room for ``pull`` included -- so
    ``inside``, ``outside`` and ``auto`` all draw the same disc. Outside
    labels and leaders are unclipped and sit beyond the axes frame, and
    the layout grows the figure around them the way it does for any
    other decoration while the axes keeps its ``axes_size``. Because the
    axes' millimetre size is fixed, ``offset`` is converted to data units
    once and renders as exactly that many millimetres.

    ``pad`` is accepted for signature parity with the other strategies
    and has **no effect** here: it is a margin between labels and the
    axes edge, and these labels are not kept inside that edge.
    """
    meta = _meta(ax)
    anchor = _DEFAULT_ANCHOR if anchor is None else anchor
    if anchor not in _VALID_ANCHORS:
        raise ValueError(
            f"wedge_values anchor must be one of {sorted(_VALID_ANCHORS)}; "
            f"got {anchor!r}"
        )
    basis = _DEFAULT_BASIS if basis is None else basis
    if basis not in _VALID_BASES:
        raise ValueError(
            f"basis must be one of {sorted(_VALID_BASES)}; got {basis!r}"
        )
    if isinstance(rotation, str) and rotation not in _VALID_ROTATIONS:
        raise ValueError(
            f"rotation must be a finite number of degrees or one of "
            f"{sorted(_VALID_ROTATIONS)}; got {rotation!r}"
        )
    # Defensive: the dispatcher always passes `rotation=` explicitly, so
    # text_kws cannot currently carry one and this pop never fires. It
    # stays because the top-level parameter must win if a future caller
    # reaches this function directly -- `ax.text` would raise on the
    # duplicate keyword rather than pick one.
    text_kws.pop("rotation", None)

    if not meta.wedges:
        return []

    renderer = _ensure_renderer(ax)

    # Both conversions are made once, for the whole call. Nothing here
    # touches the axis limits and the layout fixes the axes box in
    # millimetres, so a data unit is worth the same number of millimetres
    # now as when the figure is drawn: an `offset` converted here renders
    # as exactly that many millimetres, with no framing to settle first.
    # `apply_aspect` first, so that under `aspect="equal"` the box these
    # read is the square matplotlib will draw rather than the full cell
    # -- see `_data_per_pixel`.
    ax.apply_aspect()
    scale = _data_per_pixel(ax)
    gap = mm_to_data(offset, ax, "y")
    # A leader outlines the gap between rim and label rather than being
    # data itself, so it takes `edgewidth`, not `lines.linewidth`.
    edgewidth = resolve_param("edgewidth", None)
    # Outside labels and their leaders are decorations: they reach past
    # the axes frame, and the layout grows the figure to hold them the
    # way it does for a title or a tick label. A clipped label would be
    # cut at the frame instead, so neither may clip. `ax.text` already
    # defaults to unclipped; saying so keeps that from being an accident.
    text_kws.setdefault("clip_on", False)

    texts: List[Text] = []
    for record in meta.wedges:
        share = (record.fraction_of_total if basis == "total"
                 else record.fraction_of_parent)
        # Percentage is the default; an explicit spec formats the value.
        if fmt == _WEDGE_DEFAULT_FMT:
            label = format(share * 100.0, ".1f") + "%"
        else:
            label = _format_value(record.value, fmt)

        mid = math.radians((record.theta0 + record.theta1) / 2.0)
        place_outside = anchor == "outside"
        if place_outside:
            x, y, ha = _outside_point(record, mid, gap)
        else:
            x, y, ha = _inside_point(record, mid)

        text = ax.text(
            x, y, label,
            ha=ha, va="center",
            rotation=_rotation_for(rotation, math.degrees(mid)),
            # resolve_color reads `.patch` and `.hue_color` off the record
            # and branches on the anchor, so a WedgeRecord works unchanged.
            # Pass the anchor this label actually uses, not the one that
            # was requested: under `auto` they differ per wedge, and the
            # contrast calculation is only correct for a label drawn over
            # a fill.
            color=resolve_color(
                record,
                color,
                "outside" if place_outside else "inside",
                ax,
                hue_active=meta.hue_active,
            ),
            **text_kws,
        )
        texts.append(text)

        # Under `auto` the label is measured where it was born, inside its
        # wedge, and moved out only if it does not fit there. One pass is
        # enough: the disc no longer shrinks when a label leaves it, so a
        # label that fitted cannot be unseated by another one moving out.
        if anchor == "auto" and not _label_fits(text, record, renderer,
                                                 scale, mid):
            place_outside = True
            x, y, ha = _outside_point(record, mid, gap)
            text.set_position((x, y))
            text.set_ha(ha)
            text.set_color(resolve_color(
                record, color, "outside", ax, hue_active=meta.hue_active,
            ))

        if not place_outside:
            continue
        cx, cy = record.patch.center
        ax.plot(
            [cx + record.r_outer * math.cos(mid), x],
            [cy + record.r_outer * math.sin(mid), y],
            # The wedge's own edge, not the themed `edgecolor`: that
            # rcParam is None by default, which means "match the fill"
            # for a wedge outline and would mean "take the next colour
            # off the property cycle" here -- leaders in colours from no
            # wedge at all. Reading the drawn patch honours an explicit
            # edgecolor too, since the patch already carries it.
            color=record.patch.get_edgecolor(),
            linewidth=edgewidth,
            solid_capstyle="butt",
            zorder=record.patch.get_zorder() + 1,
            # Line2D clips to the axes by default, which would cut a
            # leader off at the frame on its way to a label beyond it.
            clip_on=False,
        )

    return texts
