"""The ``wedge_values`` annotate strategy: labels for pie and donut wedges.

Placement here is decided by *measuring* the rendered label against the
room its wedge actually has, not by a threshold on the wedge's angular
span. A fixed degree threshold is the wrong shape for this problem: the
room a label needs depends on the chart's physical size, on how long the
label string is, and on where the wedge points. Measured at the themed
7pt font with the default ``.1f`` percentage, the span a label needs to
sit inside runs from 14 degrees to 134 degrees across ordinary inputs --
see :func:`_label_fits` for the numbers and the geometry.
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

MM_PER_INCH = 25.4

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

# How many times the place -> measure -> expand cycle may run. Pushing a
# label outside grows the axis limits, which shrinks the disc, which can
# in turn stop a label that had fitted from fitting. Both moves are
# monotone -- a label never comes back inside and the limits never
# shrink -- so the cycle settles; this only bounds the work.
_MAX_FIT_PASSES = 3


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
    """Data units spanned by one display pixel.

    Both wedge framings -- ``aspect="cell"``, which matches the data
    limits to the axes' millimetre ratio, and ``aspect="equal"`` -- leave
    the x and y scales equal, which is what makes the disc render round.
    One scalar therefore describes both directions.
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

    Measured against this test at the themed 7pt font, the span a label
    needs to sit inside a pie wedge is not one number: ``"50.0%"`` on the
    default 40x40mm axes needs 43 degrees at twelve o'clock but only 14
    degrees at three o'clock, because the label lies along the radius
    there; the same label needs 72 degrees on a 25mm chart and 21 on an
    80mm one; and ``"1,234,567"`` on that 25mm chart needs 134 degrees.
    That is the span of values a single constant would have had to stand
    for, which is why this measures instead.
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


def _expand_limits(ax: Axes, texts: List[Text], renderer, pad_mm: float) -> bool:
    """Grow the framing until every label clears the axis edge by ``pad``.

    Both limits scale by the same factor about the chart's centre, so the
    disc stays round. Because the whole picture -- disc and labels alike
    -- lives in data coordinates, growing the limits shrinks it uniformly
    rather than moving a label relative to its wedge.

    A label's anchor sits at a fixed data coordinate ``a`` and its box
    reaches a fixed *pixel* distance from that anchor, so after scaling
    the limit radius ``r`` by ``f`` the label reaches
    ``|a| + reach_px * scale * f`` and must clear ``r * f``. Solving for
    ``f`` gives the factor below in closed form; no iteration is needed.

    Returns True when the limits changed.
    """
    if not texts:
        return False
    pad_px = pad_mm * ax.figure.dpi / MM_PER_INCH
    scale = _data_per_pixel(ax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0

    factor = 1.0
    for text in texts:
        ax_x, ax_y = text.get_position()
        px, py = ax.transData.transform((ax_x, ax_y))
        box = text.get_window_extent(renderer)
        reach_x = max(abs(box.x0 - px), abs(box.x1 - px)) + pad_px
        reach_y = max(abs(box.y0 - py), abs(box.y1 - py)) + pad_px
        room_x = rx - reach_x * scale
        room_y = ry - reach_y * scale
        # A non-positive denominator means the label is wider than the
        # whole cell; no finite framing helps, so leave it be rather than
        # blowing the limits up to infinity.
        if room_x > 0.0:
            factor = max(factor, abs(ax_x - cx) / room_x)
        if room_y > 0.0:
            factor = max(factor, abs(ax_y - cy) / room_y)

    if factor <= 1.0:
        return False
    ax.set_xlim(cx - rx * factor, cx + rx * factor)
    ax.set_ylim(cy - ry * factor, cy + ry * factor)
    return True


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
    # Top-level `rotation` wins over one smuggled via text_kws; pop silently.
    text_kws.pop("rotation", None)

    if not meta.wedges:
        return []

    renderer = _ensure_renderer(ax)
    gap = mm_to_data(offset, ax, "y")

    texts: List[Text] = []
    mids: List[float] = []
    outside: List[bool] = []

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

        texts.append(ax.text(
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
        ))
        mids.append(mid)
        outside.append(place_outside)

    # Settle placement and framing together: a label that does not fit is
    # pushed out, which widens the limits, which shrinks the disc and can
    # unseat another label.
    for _ in range(_MAX_FIT_PASSES):
        scale = _data_per_pixel(ax)
        moved = False
        if anchor == "auto":
            for index, record in enumerate(meta.wedges):
                if outside[index]:
                    continue
                if _label_fits(texts[index], record, renderer, scale,
                               mids[index]):
                    continue
                x, y, ha = _outside_point(record, mids[index],
                                          mm_to_data(offset, ax, "y"))
                texts[index].set_position((x, y))
                texts[index].set_ha(ha)
                texts[index].set_color(resolve_color(
                    record, color, "outside", ax, hue_active=meta.hue_active,
                ))
                outside[index] = True
                moved = True
        grew = _expand_limits(ax, texts, renderer, pad)
        if not moved and not grew:
            break

    # Leader lines last, so they are drawn against the settled framing.
    # A leader outlines the gap between rim and label rather than being
    # data itself, so it takes `edgewidth`, not `lines.linewidth`.
    edgewidth = resolve_param("edgewidth", None)
    for index, record in enumerate(meta.wedges):
        if not outside[index]:
            continue
        cx, cy = record.patch.center
        mid = mids[index]
        x0 = cx + record.r_outer * math.cos(mid)
        y0 = cy + record.r_outer * math.sin(mid)
        x1, y1 = texts[index].get_position()
        ax.plot(
            [x0, x1], [y0, y1],
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
        )

    return texts
